"""Module D: synchronized dispatch and feedback; no hardware-specific commands."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Protocol
import numpy as np
from .interfaces import RobotState, PlannedTrajectory, Status, RunState

@dataclass
class Command:
    joint_names: list[str]
    positions: np.ndarray
    velocities: np.ndarray
    base_control: np.ndarray
    timestamp: float

class Controller(Protocol):
    def read_state(self) -> RobotState: ...
    def send(self, command: Command) -> None: ...
    def hold(self) -> None:
        """Stop/hold using controller-specific braking, not position teleportation."""


class TrajectoryExecutor:
    def __init__(self, controller: Controller, completion_check: Callable[[RobotState], bool]):
        self.controller = controller
        self.completion_check = completion_check
        self.plan: PlannedTrajectory | None = None
        self.started_at = 0.
        self.state = RunState.IDLE
        self.tracking_error = np.empty(0)

    def start(self, plan: PlannedTrajectory, timestamp: float, environment_version: int,
              target_version: int) -> None:
        if plan.validation_status != Status.PASSED or plan.base_validation_status != Status.PASSED:
            raise ValueError('Only validated joint and base trajectories may execute')
        if (plan.environment_version, plan.target_version) != (environment_version, target_version):
            raise ValueError('Planning snapshot changed; replan from current feedback')
        self.plan = plan
        self.started_at = timestamp
        self.state = RunState.EXECUTING

    def stop(self, reason: str, state: RunState = RunState.SAFE_STOP) -> None:
        self.controller.hold()
        if self.plan is not None:
            self.plan.execution_status = Status.FAILED
            self.plan.reason = reason
        self.state = state

    def tick(self, timestamp: float) -> bool:
        """Return segment completion, never claim that a grasp succeeded."""
        if self.state != RunState.EXECUTING:
            return False
        plan = self.plan
        elapsed = min(max(timestamp-self.started_at, 0.), plan.times[-1])
        q = np.array([np.interp(elapsed, plan.times, plan.positions[:,j]) for j in range(len(plan.joint_names))])
        v = np.array([np.interp(elapsed, plan.times, plan.velocities[:,j]) for j in range(len(plan.joint_names))])
        interval = min(np.searchsorted(plan.times, elapsed, side='right')-1, len(plan.base_controls)-1)
        base = plan.base_controls[interval] if elapsed < plan.times[-1] else np.zeros(2)
        # Send locked joints as holds; approach cannot accidentally unlock the lift/left arm.
        names = plan.start_state.joint_names
        full_q = plan.start_state.joint_positions.copy()
        full_v = np.zeros_like(full_q)
        for name, position, velocity in zip(plan.joint_names, q, v):
            idx = names.index(name)
            full_q[idx], full_v[idx] = position, velocity
        self.controller.send(Command(names, full_q, full_v, base.copy(), timestamp))
        measured = self.controller.read_state()
        actual, _ = measured.ordered(plan.joint_names)
        self.tracking_error = actual-q
        if elapsed == plan.times[-1] and self.completion_check(measured):
            plan.execution_status = Status.PASSED
            self.state = RunState.IDLE
            return True
        return False
