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


def _plan_grasp_tier(state, environment, target, planner_factory, settings, scope):
    side=target.tool_frame[0]
    started=time.perf_counter()
    logging.getLogger(__name__).info('Search %s %s rank=%s', side, scope, target.task_constraints.get('candidate_rank'))
    contact_world=contact_environment(environment,target)
    grasp_target=replace(target,target_type='grasp',target_pose=np.array(target.task_constraints['grasp_pose']))
    grasp_planner=planner_factory(state,contact_world,False,side,state).grasp_solver(scope)
    mobile_planner=planner_factory(state,environment,False,side,state)
    q0=state_vector(state,grasp_planner.names)
    stats={'motion_scope':scope,'grasp_ik_success':0,'grasp_constraints_pass':0,'retreat_pass':0,'connection_pass':0}
    best=None;best_cost=np.inf;last_reason='No grasp IK satisfying all constraints'
    for _ in range(settings['max_attempts']):
        for grasp_q in grasp_planner.ik_candidates(grasp_target,q0):
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
    for scope in GRASP_TIERS:
        item = _plan_grasp_tier(state, environment, target, planner_factory, settings, scope)
        history.append(dict(item.search_stats, reason=item.approach_reason))
        if item.complete:
            break
    item.search_stats['priority_trials'] = history
    return item


def compare_arms(state, environment, targets, planner_factory, settings):
    """Search all arm/height choices in a tier before allowing more motion."""
    history = []
    for scope in GRASP_TIERS:
        candidates = {}
        for side in ('l', 'r'):
            choices = targets[side] if isinstance(targets[side], list) else [targets[side]]
            best = None; best_score = np.inf; attempts = []
            for target in choices:
                item = _plan_grasp_tier(state, environment, target, planner_factory, settings, scope)
                attempts.append({'candidate_rank':target.task_constraints.get('candidate_rank'),
                                 'stats':item.search_stats.copy(), 'reason':item.approach_reason})
                if item.complete:
                    score = path_cost(item.pregrasp,side,settings)[0] + path_cost(item.approach,side,settings)[0]
                    if score < best_score:
                        best, best_score = item, score
                elif best is None:
                    best = item
            best.search_stats['height_candidates'] = attempts
            candidates[side] = best
        history.append({'scope':scope, 'arms':{side:deepcopy(item.search_stats) for side,item in candidates.items()}})
        complete = [(path_cost(item.pregrasp,side,settings)[0]+path_cost(item.approach,side,settings)[0],side)
                    for side,item in candidates.items() if item.complete]
        if complete:
            selected = min(complete)[1]
            break
    else:
        selected = None
    for item in candidates.values():
        item.search_stats['priority_trials'] = history
    return candidates, selected, selected
