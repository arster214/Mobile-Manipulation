"""Grasp-first planning: solve grasp, retreat with fixed base, then connect start."""
from copy import deepcopy
from dataclasses import dataclass, replace
import time
import logging
import numpy as np
from scipy.spatial.transform import Rotation
from .interfaces import Status, PlannedTrajectory
from .model import state_vector
from .selection import path_cost, side_allowed


@dataclass
class ArmCandidate:
    arm: str
    target: object
    pregrasp: object
    search_stats: dict
    approach: object = None
    approach_reason: str = ''
    task_suffix: object = None

    @property
    def complete(self):
        return (self.pregrasp.validation_status == Status.PASSED and self.approach is not None
                and self.approach.validation_status == Status.PASSED)


def contact_environment(environment, target):
    """Exclude only the grasped object, never robot self collision or other obstacles."""
    obstacles = deepcopy(environment.obstacles)
    name = target.task_constraints.get('collision_object_name', 'Cup')
    del obstacles['cuboid'][name]
    return replace(environment, obstacles=obstacles)


def state_at(reference, names, q):
    state = replace(reference, joint_positions=reference.joint_positions.copy(),
                    joint_velocities=np.zeros_like(reference.joint_velocities),
                    base_pose=reference.base_pose.copy(), base_twist=np.zeros(2))
    for name,value in zip(names,q):
        if name in ('base_x','base_y','base_yaw'):
            if name=='base_yaw':
                state.base_pose[3:]=Rotation.from_euler('z',value).as_quat(scalar_first=True)
            else:
                state.base_pose[0 if name=='base_x' else 1]=value
        else:
            state.joint_positions[state.joint_names.index(name)]=value
    return state


def reverse_retreat(retreat, pregrasp_state):
    """Time reversal keeps the same configurations, speeds and accelerations bounds."""
    return replace(retreat, positions=retreat.positions[::-1].copy(),
                   velocities=-retreat.velocities[::-1].copy(),
                   accelerations=retreat.accelerations[::-1].copy(),
                   times=retreat.times[-1]-retreat.times[::-1], start_state=pregrasp_state,
                   base_poses=retreat.base_poses[::-1].copy(),
                   base_controls=-retreat.base_controls[::-1].copy(),
                   validation_status=Status.NOT_RUN, execution_status=Status.NOT_RUN)


def _plan_grasp_tier(state, environment, target, planner_factory, settings, scope, planners=None, seed_hint=None, continuation=None):
    side=target.tool_frame[0]
    started=time.perf_counter()
    logging.getLogger(__name__).info('Search %s %s rank=%s', side, scope, target.task_constraints.get('candidate_rank'))
    contact_world=contact_environment(environment,target)
    grasp_target=replace(target,target_type='grasp',target_pose=np.array(target.task_constraints['grasp_pose']))
    if planners is None:
        planners = {}
    key = (side, target.task_constraints.get('collision_object_name', 'Cup'))
    if key not in planners:
        planners[key] = (planner_factory(state,contact_world,False,side,state),
                         planner_factory(state,environment,False,side,state))
    template,mobile_planner = planners[key]
    grasp_planner = template.grasp_solver(scope)
    # Reuse initialization, while retaining the previous per-height seed sequence.
    grasp_planner.rng = np.random.default_rng(0)
    q0=state_vector(state if seed_hint is None else seed_hint,grasp_planner.names)
    stats={'motion_scope':scope,'grasp_ik_success':0,'grasp_constraints_pass':0,'retreat_pass':0,'connection_pass':0}
    best=None;best_cost=np.inf;last_reason='No grasp IK satisfying all constraints'
    # The optional warm refinement tests the known mobile branch once; the
    # preceding fixed-tier search already exhausted the full retry budget.
    attempts = settings['max_attempts'] if seed_hint is None else 1
    if scope != 'mobile' and not side_allowed(state_vector(state,['base_x','base_y','base_yaw']),target.object_pose,side,settings):
        attempts = 0
        last_reason = 'Fixed base heading violates grasp-arm side constraint'
    for _ in range(attempts):
        stats['ik_batches'] = stats.get('ik_batches', 0) + 1
        solutions = grasp_planner.ik_candidates(grasp_target,q0)
        if scope == 'mobile':
            from .trajectory import wrapped
            origin = state_vector(state, ['base_x','base_y','base_yaw'])
            weights = settings['path_cost_weights']
            def base_effort(q):
                return (weights['base_distance_m']*np.linalg.norm(q[:2]-origin[:2])
                        + weights['base_rotation_rad']*abs(wrapped(q[2]-origin[2])))
            solutions.sort(key=base_effort)
        for grasp_q in solutions:
            stats['grasp_ik_success']+=1
            grasp_state=state_at(state,grasp_planner.names,grasp_q)
            grasp_base=state_vector(grasp_state,['base_x','base_y','base_yaw'])
            if (not side_allowed(grasp_base,target.object_pose,side,settings)
                    or not grasp_planner.posture_metrics(grasp_q[None])['passed']):
                continue
            stats['grasp_constraints_pass']+=1
            grasp_state=state_at(state,grasp_planner.names,grasp_q)
            arm_planner=planner_factory(grasp_state,contact_world,True,side,state)
            actual_grasp=arm_planner.tcp_pose(state_vector(grasp_state,arm_planner.names)[None])[0]
            try:
                retreat=arm_planner.plan_approach(replace(target,target_type='retreat'),actual_grasp)
            except RuntimeError as error:
                last_reason=str(error)
                continue
            if retreat.validation_status!=Status.PASSED:
                last_reason=retreat.reason
                continue
            stats['retreat_pass']+=1
            pregrasp_state=state_at(grasp_state,arm_planner.names,retreat.positions[-1])
            pregrasp_q=state_vector(pregrasp_state,mobile_planner.names)
            # The connection retains Cup collision checking. Use this exact retreated posture.
            mobile=mobile_planner.plan_mobile(target,goal_configs=[pregrasp_q])
            if mobile.validation_status!=Status.PASSED:
                last_reason=mobile.reason
                continue
            approach=reverse_retreat(retreat,pregrasp_state)
            approach.ignored_environment_objects=(target.task_constraints.get('collision_object_name','Cup'),)
            if not arm_planner.validate(approach):
                last_reason=approach.reason
                continue
            stats['connection_pass']+=1
            cost=path_cost(mobile,side,settings)[0]+path_cost(approach,side,settings)[0]
            if cost<best_cost:
                best_cost=cost
                best=ArmCandidate(side,target,mobile,stats.copy(),approach)
                if continuation is not None:
                    try:
                        best.task_suffix = continuation(best)
                    except RuntimeError as error:
                        last_reason = str(error)
                        logging.getLogger(__name__).info('Reject %s %s task suffix: %s', side, scope, last_reason)
                        stats['task_rejections'] = stats.get('task_rejections', 0) + 1
                        best = None
                        best_cost = np.inf
                        continue
                best.pregrasp.planning_time=time.perf_counter()-started
                return best
    if best is None:
        failed=PlannedTrajectory([],np.empty((0,0)),np.empty((0,0)),np.empty((0,0)),np.array([0.]),
                                  state,environment.version,target.version,
                                  ik_status=Status.PASSED if stats['grasp_constraints_pass'] else Status.FAILED,
                                  planning_status=Status.FAILED,validation_status=Status.FAILED,reason=last_reason)
        best=ArmCandidate(side,target,failed,stats,approach_reason=last_reason)
    best.search_stats=stats.copy()
    best.pregrasp.planning_time=time.perf_counter()-started
    return best


GRASP_TIERS = ('arm_only', 'arm_lift', 'mobile')


def plan_arm_from_grasp(state, environment, target, planner_factory, settings):
    """Escalate only after the complete grasp/retreat/connection search fails."""
    history = []
    planners = {}
    for scope in GRASP_TIERS:
        item = _plan_grasp_tier(state, environment, target, planner_factory, settings, scope, planners)
        history.append(dict(item.search_stats, reason=item.approach_reason))
        if item.complete:
            break
    item.search_stats['priority_trials'] = history
    return item


def _candidate_rank(item, side, settings, scope):
    pregrasp_cost, terms = path_cost(item.pregrasp, side, settings)
    total = pregrasp_cost + path_cost(item.approach, side, settings)[0]
    if getattr(item, 'task_suffix', None) is not None:
        total += sum(path_cost(plan, side, settings)[0]
                     for plan, _ in item.task_suffix['stages'].values())
    weights = settings['path_cost_weights']
    base = (weights['base_distance_m']*terms['base_distance_m']
            + weights['base_rotation_rad']*terms['base_rotation_rad']) if scope == 'mobile' else 0.
    return base, total


def compare_arms(state, environment, targets, planner_factory, settings, continuation=None, allowed_arm='auto'):
    """Search all arm/height choices in a tier before allowing more motion."""
    history = []
    planners = {}
    for scope in GRASP_TIERS:
        candidates = {}
        for side in (('l', 'r') if allowed_arm == 'auto' else (allowed_arm,)):
            choices = targets[side] if isinstance(targets[side], list) else [targets[side]]
            best = None; best_score = None; attempts = []
            for target in choices:
                options = {} if continuation is None else {'continuation': continuation}
                item = _plan_grasp_tier(state, environment, target, planner_factory, settings, scope, planners, **options)
                attempts.append({'candidate_rank':target.task_constraints.get('candidate_rank'),
                                 'stats':item.search_stats.copy(), 'reason':item.approach_reason})
                if item.complete:
                    score = _candidate_rank(item, side, settings, scope)
                    if best_score is None or score < best_score:
                        best, best_score = item, score
                    # Full-task feasibility is the expensive acceptance gate.
                    # Keep one complete candidate per arm in the first feasible
                    # tier; do not re-plan the entire suffix for every angle.
                    if continuation is not None:
                        break
                elif best is None:
                    best = item
            best.search_stats['height_candidates'] = attempts
            candidates[side] = best
        history.append({'scope':scope, 'arms':{side:deepcopy(item.search_stats) for side,item in candidates.items()}})
        complete = [(_candidate_rank(item, side, settings, scope), side)
                    for side,item in candidates.items() if item.complete]
        if complete:
            selected = min(complete)[1]
            if scope == 'mobile':
                mobile = candidates[selected]
                hint = state_at(mobile.approach.start_state, mobile.approach.joint_names,
                                mobile.approach.positions[-1])
                for fixed_scope in ('arm_only', 'arm_lift'):
                    refined = _plan_grasp_tier(state, environment, mobile.target, planner_factory,
                                              settings, fixed_scope, planners, seed_hint=hint, continuation=continuation)
                    history.append({'scope':fixed_scope, 'seed_source':'feasible_mobile_grasp',
                                    'arms':{selected:deepcopy(refined.search_stats)},
                                    'reason':refined.approach_reason})
                    if refined.complete:
                        candidates[selected] = refined
                        break
            break
    else:
        selected = None
    for item in candidates.values():
        item.search_stats['priority_trials'] = history
    return candidates, selected, selected
