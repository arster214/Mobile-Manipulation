"""Solve placement first, retreat upwards, connect carry, then reverse to lower."""
from dataclasses import replace
from time import perf_counter
import numpy as np
from .interfaces import TaskTarget, Status
from .model import state_vector
from .arm_comparison import state_at, reverse_retreat, GRASP_TIERS
from .carried_object import pose_matrix, matrix_pose


def plan_placement(lifted_state, nominal_state, goal_object_pose, attachment, make_planner,
                   settings, clearance):
    """Nominal base is a destination seed, never a teleported execution state.

    All returned motion begins at lifted_state. Final lowering uses measured locks
    at the selected placement base/lift state. Every trial includes carry validation.
    """
    timers = dict(final_ik_s=0., upward_retreat_s=0., carry_s=0., lowering_validation_s=0.)
    trials = []
    target_pose = matrix_pose(pose_matrix(goal_object_pose) @ np.linalg.inv(attachment.tcp_from_object))
    target = TaskTarget(target_pose, np.array(goal_object_pose), 0., tool_frame='r_grasp_tcp',
                        task_constraints={'approach_direction':[0.,1.,0.]})
    carry_planner = make_planner(lifted_state, False)
    nominal_planner = make_planner(nominal_state, False)
    for scope in GRASP_TIERS:
        begin = perf_counter()
        solver = nominal_planner.grasp_solver(scope)
        seed = state_vector(nominal_state, solver.names)
        timers['final_ik_s'] += perf_counter()-begin
        record = {'scope':scope, 'ik_solutions':0, 'retreat_pass':0, 'carry_pass':0}
        trials.append(record)
        for attempt in range(settings['max_attempts']):
            begin = perf_counter()
            solutions = solver.ik_candidates(target, seed)
            timers['final_ik_s'] += perf_counter()-begin
            for q in solutions:
                record['ik_solutions'] += 1
                if not solver.posture_metrics(q[None])['passed']:
                    continue
                placed_state = state_at(nominal_state, solver.names, q)
                begin = perf_counter()
                arm_planner = make_planner(placed_state, True)
                actual = arm_planner.tcp_pose(state_vector(placed_state,arm_planner.names)[None])[0]
                upper = actual.copy(); upper[2] += clearance
                try:
                    retreat = arm_planner.plan_approach(replace(target,target_pose=upper,target_type='preplace_retreat'),actual)
                except RuntimeError as error:
                    record['last_failure'] = str(error)
                    timers['upward_retreat_s'] += perf_counter()-begin
                    continue
                timers['upward_retreat_s'] += perf_counter()-begin
                if retreat.validation_status != Status.PASSED:
                    record['last_failure'] = retreat.reason
                    continue
                record['retreat_pass'] += 1
                preplace_state = state_at(placed_state,arm_planner.names,retreat.positions[-1])
                begin = perf_counter()
                carry = carry_planner.plan_mobile(replace(target,target_pose=upper),
                           goal_configs=[state_vector(preplace_state,carry_planner.names)])
                timers['carry_s'] += perf_counter()-begin
                if carry.validation_status != Status.PASSED:
                    record['last_failure'] = carry.reason
                    continue
                begin = perf_counter()
                lowering = reverse_retreat(retreat,preplace_state)
                valid = arm_planner.validate(lowering)
                timers['lowering_validation_s'] += perf_counter()-begin
                if valid:
                    record['carry_pass'] += 1
                    return carry,lowering,placed_state,carry_planner,arm_planner,{'timers':timers,'trials':trials,'scope':scope}
    raise RuntimeError(f'No complete placement/carry candidate: {trials}')
