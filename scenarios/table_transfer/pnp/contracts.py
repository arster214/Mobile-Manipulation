"""Runtime PnP contracts. Poses are metres, world <- frame, quaternion wxyz.

Observation timestamps and Context.time() share a clock. Total deadlines use
monotonic wall time, so paused simulation or a stalled controller cannot hide them.
"""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Pose:
    xyz: tuple
    wxyz: tuple = (1., 0., 0., 0.)
    frame: str = 'world'
    timestamp: float = 0.
    valid: bool = True
    source: str = 'simulation_truth'


@dataclass(frozen=True)
class Surface:
    identity: str
    size_xy: tuple
    # Pose origin is TOP of the support, not table/tray centre.
    pose: Pose


@dataclass(frozen=True)
class Goal:
    request_id: str
    object_id: str
    dimensions: tuple
    source_id: str
    destination_id: str
    placement_xy: tuple = (0., 0.)  # In destination support frame.
    placement_yaw: float = 0.      # Relative to destination frame, radians.
    arm: str = 'auto'
    timeout_s: float = 1800.
    max_retries: int = 1


@dataclass
class Snapshot:
    object_id: str
    object_pose: Pose
    tcp_pose: Pose
    surfaces: dict[str, Surface]
    linear_velocity: tuple
    angular_velocity: tuple
    gripper: dict  # measured opening_m, action_complete, released (bool or None)
    robot_state: dict = field(default_factory=dict)


class Fault(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class Context(Protocol):
    """One exclusive owner of base/lift/arms/gripper per acquired lease.

    start is NONBLOCKING, step advances I/O and returns True on measured
    completion. Planning adapters must run cancellable work out of process.
    hold stops active commands/planners but NEVER opens a held gripper.
    Snapshot object feedback must not be synthesized from planning attachment.
    """
    mode: str
    def acquire(self, run_id: str) -> bool: ...
    def release(self, run_id: str) -> None: ...
    def time(self) -> float: ...
    def observe(self, goal: Goal) -> Snapshot: ...
    def start(self, stage: str, goal: Goal, geometry: dict) -> None: ...
    def step(self) -> bool: ...
    def hold(self) -> bool: ...
