"""Module F remains disabled until the cuRobo baseline is verified."""
from typing import Protocol
from .interfaces import RobotState, TaskTarget
import numpy as np

class LocalCorrection(Protocol):
    def joint_velocity(self, state: RobotState, target: TaskTarget) -> np.ndarray:
        """Return a candidate joint velocity; caller must validate it."""


def create_local_controller(enabled: bool = False) -> None:
    if enabled:
        raise NotImplementedError('Pink closed-loop correction is not enabled in the baseline')
    return None
