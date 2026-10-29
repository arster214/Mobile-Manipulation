"""Backtrack over grasp IK branches until the entire pick/place task is feasible.

No scene IDs, successful-angle lists, robot teleports or execution commands.
The caller supplies current state, measured targets, world geometry and factory.
"""
from dataclasses import dataclass
import numpy as np
from .arm_comparison import compare_arms, contact_environment, state_at
from .carried_object import CarriedObject, pose_matrix, matrix_pose
from .interfaces import Status, TaskTarget
from .model import state_vector
from .placement import plan_placement
from .trajectory import candidate


@dataclass
class TaskSolution:
    grasp: object
    stages: dict
    attachment: CarriedObject
    object_goal_pose: np.ndarray
    search: dict


def released_object_audit(planner, positions, object_pose, attachment):
    """The released object is stationary; only intended gripper contacts are exempt."""
    transform = pose_matrix(object_pose)
    kin = planner.checker.get_kinematics(planner.tensor(positions[:, planner.to_internal])[None])
    spheres = kin.robot_spheres[0].detach().cpu().numpy()
    params = planner.checker.kinematics.config.kinematics_config
    names = [params.all_link_names[i] for i in params.link_sphere_idx_map.cpu().tolist()]
    local = (spheres[:, :, :3]-transform[:3, 3]) @ transform[:3, :3]
    gaps = np.linalg.norm(np.maximum(abs(local)-attachment.dimensions/2, 0), axis=-1)-spheres[:, :, 3]
    active = np.array([name not in attachment.contact_links for name in names])
    return {'passed': bool(np.all((gaps >= -1e-6) | (~active[None]) | (spheres[:, :, 3] <= 0))),
            'object_state': 'released', 'finger_contact_allowed': True}


def plan_pick_place(state, environment, targets, goal_object_pose, planner_factory,
                    settings, lift_height, withdraw_distance, allowed_arm='auto'):
    """A failed placement/withdrawal rejects the grasp branch, not the whole task.

    planner_factory has the same signature as compare_arms. Stage tuples contain
    (trajectory, planner), allowing the caller to serialize or execute only a
    fully accepted solution. Search budgets remain in the shared configuration.
    """
    def continuation(grasp):
        side = grasp.arm
        world = contact_environment(environment, grasp.target)
        start = state_at(grasp.approach.start_state, grasp.approach.joint_names,
                         grasp.approach.positions[-1])

        def make(snapshot, arm_only, attachment=None):
            planner = planner_factory(snapshot, world, arm_only, side, state)
            if attachment is not None:
                planner.extra_validation = lambda q: attachment.audit(planner, q, world)
            return planner

        first = make(start, False)
        tcp = first.tcp_pose(state_vector(start, first.names)[None])[0]
        attachment = CarriedObject.attach_box(tcp, grasp.target.object_pose,
                                              grasp.target.task_constraints['object_dimensions_m'], side)
        first.extra_validation = lambda q: attachment.audit(first, q, world)
        lifted_q = state_vector(start, first.names)
        lifted_q[first.names.index('platform_joint')] -= lift_height
        lift = candidate(start, lifted_q, {k: v[3:] for k, v in first.limits.items()},
                         settings, world.version, grasp.target.version)
        lift.ik_status = lift.planning_status = Status.PASSED
        if not first.validate(lift):
            raise RuntimeError('lift: '+lift.reason)
        lifted = state_at(start, first.names, lifted_q)
        nominal = state_at(lifted, ['base_x', 'base_y'], lifted.base_pose[:2]
                           + np.asarray(goal_object_pose[:2])-grasp.target.object_pose[:2])
        release = {}

        def check_withdrawal(placed_state, place_planner, lowering):
            actual_object = matrix_pose(attachment.poses(
                place_planner.tcp_pose(lowering.positions[-1:]))[0])
            planner = make(placed_state, True)
            current = planner.tcp_pose(state_vector(placed_state, planner.names)[None])[0]
            # Retreat along the gripper's actual negative approach axis; this
            # remains valid after transporting or rotating the held object.
            direction = -pose_matrix(current)[:3, 2]
            target_pose = current.copy()
            target_pose[:3] += direction*withdraw_distance
            target = TaskTarget(target_pose, actual_object, state.timestamp,
                                tool_frame=f'{side}_grasp_tcp', target_type='withdraw')
            planner.extra_validation = lambda q: released_object_audit(planner, q, actual_object, attachment)
            withdraw = planner.plan_approach(target, current)
            if withdraw.validation_status != Status.PASSED:
                raise RuntimeError('withdraw: '+withdraw.reason)
            release.update(stage=(withdraw, planner), object_pose=actual_object)

        carry, place, _, carry_planner, place_planner, search = plan_placement(
            lifted, nominal, goal_object_pose, attachment,
            lambda snapshot, arm_only: make(snapshot, arm_only, attachment),
            settings, lift_height, accept_placement=check_withdrawal)
        return dict(stages={'lift': (lift, first), 'carry': (carry, carry_planner),
                            'place': (place, place_planner), 'withdraw': release['stage']},
                    attachment=attachment, object_pose=release['object_pose'], search=search)

    candidates, _, selected = compare_arms(state, environment, targets, planner_factory,
                                           settings, continuation=continuation, allowed_arm=allowed_arm)
    if selected is None:
        reasons = {side: item.approach_reason for side, item in candidates.items()}
        raise RuntimeError(f'No complete pick/place solution within search budget: {reasons}')
    grasp = candidates[selected]
    suffix = grasp.task_suffix
    return TaskSolution(grasp, suffix['stages'], suffix['attachment'],
                        suffix['object_pose'], suffix['search'])
