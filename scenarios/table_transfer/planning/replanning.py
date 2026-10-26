"""Modules D/E: versioned invalidation and measured-state replan requests."""
from __future__ import annotations
from .interfaces import EnvironmentState, ReplanningRequest, RunState, TaskTarget
from .execution import TrajectoryExecutor

class ReplanningMonitor:
    """Baseline stops before replanning; continuous moving handoff is deferred."""

    def __init__(self, executor: TrajectoryExecutor):
        self.executor = executor

    def update(self, target: TaskTarget, environment: EnvironmentState,
               timestamp: float) -> ReplanningRequest | None:
        plan = self.executor.plan
        if plan is None or self.executor.state != RunState.EXECUTING:
            return None
        if not target.valid or not environment.valid:
            self.executor.stop('Invalid perception snapshot')
            return None
        if target.version == plan.target_version and environment.version == plan.environment_version:
            return None
        reason = 'target_updated' if target.version != plan.target_version else 'environment_updated'
        self.executor.stop(reason, RunState.REPLANNING)
        # This is the actual feedback after issuing hold. Planner must await a measured stop.
        return ReplanningRequest(self.executor.controller.read_state(), target, environment,
                                 reason, timestamp-self.executor.started_at)
