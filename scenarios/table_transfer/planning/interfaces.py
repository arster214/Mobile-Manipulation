"""Shared SI-unit interfaces; world poses use quaternion order wxyz."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

class RunState(str, Enum):
    IDLE = 'IDLE'
    GLOBAL_PLANNING = 'GLOBAL_PLANNING'
    EXECUTING = 'EXECUTING'
    LOCAL_CORRECTION = 'LOCAL_CORRECTION'  # Reserved; disabled in baseline.
    REPLANNING = 'REPLANNING'
    TASK_COMPLETED = 'TASK_COMPLETED'
    SAFE_STOP = 'SAFE_STOP'

class Status(str, Enum):
    NOT_RUN = 'not_run'
    PASSED = 'passed'
    FAILED = 'failed'

@dataclass
class RobotState:
    joint_names: list[str]
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    timestamp: float
    base_pose: np.ndarray  # [x,y,z,w,x,y,z], world <- URDF base.
    frame_id: str = 'world'
    base_twist: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [forward m/s, yaw rad/s]

    def ordered(self, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        indices = [self.joint_names.index(name) for name in names]
        return self.joint_positions[indices], self.joint_velocities[indices]

@dataclass
class TaskTarget:
    target_pose: np.ndarray  # TCP world pose, not object pose.
    object_pose: np.ndarray
    timestamp: float
    version: int = 1
    target_type: str = 'pregrasp'
    target_frame: str = 'world'
    tool_frame: str = 'r_grasp_tcp'
    valid: bool = True
    confidence: float = 1.0
    task_constraints: dict = field(default_factory=dict)

@dataclass
class EnvironmentState:
    obstacles: dict
    timestamp: float
    version: int = 1
    frame_id: str = 'world'
    valid: bool = True

@dataclass
class PlannedTrajectory:
    joint_names: list[str]
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    times: np.ndarray
    start_state: RobotState
    environment_version: int
    target_version: int
    ik_status: Status = Status.NOT_RUN
    planning_status: Status = Status.NOT_RUN
    validation_status: Status = Status.NOT_RUN
    execution_status: Status = Status.NOT_RUN
    planning_time: float = 0.0
    reason: str = ''
    base_poses: np.ndarray | None = None  # [N,3]: x,y,URDF yaw on the same times.
    base_controls: np.ndarray | None = None  # [N-1,2]: forward speed, yaw rate.
    base_validation_status: Status = Status.NOT_RUN
    constraint_metrics: dict = field(default_factory=dict)
    cost: float | None = None
    cost_terms: dict = field(default_factory=dict)
    object_metrics: dict = field(default_factory=dict)
    ignored_environment_objects: tuple[str, ...] = ()

@dataclass
class ReplanningRequest:
    current_state: RobotState
    target: TaskTarget
    environment: EnvironmentState
    reason: str
    execution_elapsed: float
