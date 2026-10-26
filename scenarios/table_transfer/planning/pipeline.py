"""Orchestrate validated pregrasp and measured-locked approach segments."""
from dataclasses import replace
import numpy as np
from .interfaces import RunState, Status
from .replanning import ReplanningMonitor
from .arm_comparison import plan_arm_from_grasp, contact_environment


class ManipulationPipeline:
    """Planner factory receives state, environment, arm_only, arm and reference_state.

    The controller owns braking and feedback. A stopped-state call to plan_pregrasp
    also handles replanning; no moving-state splice or automatic grasp is implied.
    """

    def __init__(self, executor, planner_factory, tcp_pose, settings=None):
        if settings is None:
            import json
            from pathlib import Path
            settings = json.loads(Path(__file__).with_name('config.json').read_text())
        self.settings = settings
        self.prepared_approach = None
        self.executor = executor
        self.planner_factory = planner_factory
        self.tcp_pose = tcp_pose
        self.monitor = ReplanningMonitor(executor)
        self.target = self.environment = None
        self.phase = 'idle'

    def plan_pregrasp(self, target, environment, timestamp):
        state = self.executor.controller.read_state()
        self.executor.state = RunState.GLOBAL_PLANNING
        self.reference_state = state
        def factory(snapshot,world,arm_only,arm,reference_state):
            return self.planner_factory(snapshot,world,arm_only,arm=arm,reference_state=reference_state)
        item = plan_arm_from_grasp(state,environment,target,factory,self.settings)
        plan = item.pregrasp
        self.prepared_approach = item.approach
        if plan.validation_status != Status.PASSED:
            self.executor.stop(plan.reason)
            return plan
        self.target, self.environment = target, environment
        self.phase = 'pregrasp'
        self.executor.start(plan, timestamp, environment.version, target.version)
        return plan

    def plan_approach(self, timestamp):
        if self.phase != 'pregrasp_reached':
            raise ValueError('Approach requires completed pregrasp execution')
        state = self.executor.controller.read_state()
        target = replace(self.target, target_type='approach',
                         target_pose=np.asarray(self.target.task_constraints['grasp_pose']))
        world = contact_environment(self.environment,self.target)
        planner = self.planner_factory(state, world, True,
                                       arm=target.tool_frame[0], reference_state=self.reference_state)
        try:
            prepared = self.prepared_approach
            expected = prepared.start_state if prepared is not None else None
            matches = expected is not None and np.allclose(state.base_pose,expected.base_pose,atol=1e-6,rtol=0) and np.allclose(
                state.ordered(expected.joint_names)[0],expected.joint_positions,atol=1e-6,rtol=0)
            if matches:
                plan = replace(prepared,start_state=state,validation_status=Status.NOT_RUN)
                planner.validate(plan)
            else:
                # Actual feedback differs: solve again with freshly measured fixed platform values.
                plan = planner.plan_approach(target, self.tcp_pose(state))
            plan.ignored_environment_objects=(self.target.task_constraints.get('collision_object_name','Cup'),)
        except RuntimeError:
            self.executor.stop('Approach IK failed: replan pregrasp', RunState.REPLANNING)
            self.phase = 'needs_pregrasp'
            raise
        if plan.validation_status != Status.PASSED:
            self.executor.stop('Approach validation failed: replan pregrasp', RunState.REPLANNING)
            self.phase = 'needs_pregrasp'
            return plan
        self.phase = 'approach'
        self.executor.start(plan, timestamp, self.environment.version, target.version)
        return plan

    def tick(self, timestamp, target, environment):
        request = self.monitor.update(target, environment, timestamp)
        if request is not None:
            self.phase = 'needs_pregrasp'
            return request
        if self.executor.tick(timestamp):
            self.phase = 'pregrasp_reached' if self.phase == 'pregrasp' else 'approach_reached'
        return None
