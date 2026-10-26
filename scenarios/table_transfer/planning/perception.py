"""Module B: versioned simulation truth and object-to-TCP target conversion."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from .interfaces import EnvironmentState, TaskTarget

@dataclass
class ObjectObservation:
    pose: np.ndarray  # world <- object, xyz + wxyz
    timestamp: float
    version: int
    valid: bool = True
    confidence: float = 1.0


def compose_pose(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(parent[3:], scalar_first=True)
    child_rotation = Rotation.from_quat(child[3:], scalar_first=True)
    return np.r_[parent[:3] + rotation.apply(child[:3]), (rotation*child_rotation).as_quat(scalar_first=True)]


class SimulationPerception:
    """Receives timestamped simulator observations; owns no vision model."""

    def __init__(self, static_obstacles: dict, radius: float, height: float):
        self.static = deepcopy(static_obstacles)
        self.radius = radius
        self.height = height

    def environment(self, observation: ObjectObservation) -> EnvironmentState:
        obstacles = deepcopy(self.static)
        # Floor support is excluded only for this fixed-height mobile simulation.
        obstacles['cuboid'].pop('Floor', None)
        obstacles['cuboid']['Cup'] = {
            'pose': observation.pose.tolist(),
            'dims': [2*self.radius, 2*self.radius, self.height],
        }
        return EnvironmentState(obstacles, observation.timestamp, observation.version, valid=observation.valid)

    def targets(self, observation: ObjectObservation, table: dict, policy: dict, arm: str = 'r') -> list[TaskTarget]:
        # Reuse the existing center-out sampler, with current observation instead of saved pose.
        from scripts.grasp_candidates import generate
        cup = {'center_m': observation.pose[:3].tolist(), 'radius_m': self.radius,
               'height_m': self.height, 'pose_source': 'simulation_ground_truth'}
        object_up = Rotation.from_quat(observation.pose[3:], scalar_first=True).apply([0, 0, 1])
        if not np.allclose(object_up, [0, 0, 1], atol=1e-6):
            raise ValueError('Current candidate generator supports upright cylinders only')
        candidates = generate(table, cup, policy)
        conversion = Rotation.from_euler('z', policy['robot_grasp_tcp']['candidate_to_tcp_rotation_local_z_rad'])
        targets = []
        for candidate in candidates['candidates']:
            rotation = Rotation.from_matrix(candidate['rotation_world_from_grasp']) * conversion
            targets.append(TaskTarget(
                target_pose=np.r_[candidate['pregrasp_center_world_m'], rotation.as_quat(scalar_first=True)],
                object_pose=observation.pose.copy(), timestamp=observation.timestamp, tool_frame=f'{arm}_grasp_tcp',
                version=observation.version, valid=observation.valid, confidence=observation.confidence,
                task_constraints={'arm': arm, 'grasp_pose': np.r_[candidate['grasp_center_world_m'], rotation.as_quat(scalar_first=True)].tolist(),
                                  'approach_direction': candidates['approach_direction_world'],
                                  'candidate_rank': candidate['rank']},
            ))
        return targets
