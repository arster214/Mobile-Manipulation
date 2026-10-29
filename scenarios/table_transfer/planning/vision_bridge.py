"""Calibrated optical-camera observations and box targets for manipulation."""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from .perception import ObjectObservation, SimulationPerception
from .carried_object import matrix_pose


def world_observation(record, world_from_camera, version):
    """Camera extrinsics must come from calibration or the simulator, never guessed."""
    if not record.get('world_extrinsics_valid', False):
        raise ValueError('World-camera calibration is missing; camera-only estimates cannot drive planning')
    if record['units'] != 'metres' or record['camera_frame'] != 'optical x-right y-down z-forward':
        raise ValueError('Expected metric optical camera pose')
    camera_pose = np.asarray(record['T_object_to_camera'], dtype=float)
    extrinsic = np.asarray(world_from_camera, dtype=float)
    for transform in (camera_pose, extrinsic):
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError('Invalid rigid transform')
        r = transform[:3, :3]
        if not np.allclose(transform[3], [0, 0, 0, 1]) or not np.allclose(r.T @ r, np.eye(3), atol=1e-4) or np.linalg.det(r) < 0:
            raise ValueError('Invalid rigid transform')
    return ObjectObservation(matrix_pose(extrinsic @ camera_pose), record['timestamp'], version,
                             confidence=record['confidence'])


def approach_angles(count):
    """Nested coverage of the side-to-top approach interval, independent of scene."""
    values = [0.]
    for index in range(1, count):
        number, fraction, scale = index, 0., .5
        while number:
            fraction += scale*(number % 2)
            number //= 2
            scale *= .5
        values.append(90.*fraction)
    return values


class BoxPerception(SimulationPerception):
    """Table-side grasp sampling of a box, including cube orientation symmetry."""
    def __init__(self, static_obstacles, dimensions, object_name='ManipulatedCube'):
        self.static = deepcopy(static_obstacles)
        self.dimensions = np.asarray(dimensions, dtype=float)
        self.object_name = object_name

    def environment(self, observation):
        from .interfaces import EnvironmentState
        obstacles = deepcopy(self.static)
        obstacles['cuboid'].pop('Floor', None)
        obstacles['cuboid'][self.object_name] = dict(pose=observation.pose.tolist(), dims=self.dimensions.tolist())
        return EnvironmentState(obstacles, observation.timestamp, observation.version, valid=observation.valid)

    def targets(self, observation, table, policy, arm='r'):
        from .interfaces import TaskTarget
        from scripts.grasp_candidates import generate
        rotation = Rotation.from_quat(observation.pose[3:], scalar_first=True).as_matrix()
        # Choose the object axis nearest vertical; cube-24 rotations must not reject identical geometry.
        up_index = np.argmax(abs(rotation[2]))
        up = rotation[:, up_index] * np.sign(rotation[2, up_index])
        horizontal = [i for i in range(3) if i != up_index]
        proxy = dict(center_m=observation.pose[:3].tolist(),
                     radius_m=float(np.linalg.norm(self.dimensions[horizontal])/2),
                     height_m=float(self.dimensions[up_index]), pose_source='observation')
        geometric = generate(table, proxy, policy)
        toward = np.array(geometric['approach_direction_world'])
        axes = [rotation[:, i]*sign for i in horizontal for sign in (-1, 1)]
        approach = max(axes, key=lambda axis: axis @ toward)
        closing = np.cross(up, approach)
        grasp_rotation = Rotation.from_matrix(np.column_stack([closing, up, approach]))
        conversion = Rotation.from_euler('z', policy['robot_grasp_tcp']['candidate_to_tcp_rotation_local_z_rad'])
        quat = (grasp_rotation * conversion).as_quat(scalar_first=True)
        targets = []
        for pitch_deg in policy.get('box_approach_pitch_deg', approach_angles(policy.get('orientation_samples', 1))):
            tilt = Rotation.from_rotvec(closing*np.deg2rad(pitch_deg))
            direction = tilt.apply(approach)
            quat = (tilt * grasp_rotation * conversion).as_quat(scalar_first=True)
            for candidate in geometric['candidates']:
                offset = candidate['grasp_center_world_m'][2] - observation.pose[2]
                center = observation.pose[:3] + up*offset
                pre = center - direction*policy['pregrasp_backoff_m']
                targets.append(TaskTarget(np.r_[pre, quat], observation.pose.copy(), observation.timestamp,
                    tool_frame=f'{arm}_grasp_tcp', version=observation.version, valid=observation.valid,
                    confidence=observation.confidence, task_constraints={
                        'arm': arm, 'grasp_pose': np.r_[center, quat].tolist(),
                        'approach_direction': direction.tolist(), 'candidate_rank': len(targets),
                        'approach_pitch_deg': pitch_deg,
                        'collision_object_name': self.object_name, 'object_dimensions_m': self.dimensions.tolist()}))
        return targets


def rest_box_on_table(observation, dimensions, table_height):
    """Apply an explicit known resting-face constraint, retaining measured XY and yaw.

    Only use for a static box known to rest flat on this support; raw observations
    remain available for error evaluation. This is not a collision-ignore rule.
    """
    from dataclasses import replace
    rotation = Rotation.from_quat(observation.pose[3:], scalar_first=True)
    basis = rotation.as_matrix()
    axis = int(np.argmax(abs(basis[2])))
    up = basis[:, axis]*np.sign(basis[2, axis])
    cross = np.cross(up, [0.,0.,1.])
    sine = np.linalg.norm(cross)
    correction = Rotation.identity() if sine < 1e-12 else Rotation.from_rotvec(cross/sine*np.arctan2(sine,up[2]))
    pose = observation.pose.copy()
    pose[2] = table_height + dimensions[axis]/2
    supported = correction*rotation
    if np.allclose(dimensions, dimensions[0]):
        # A textureless cube has 24 equivalent frames. Keep the closest-to-world
        # representative so placement does not demand a fictitious 90-degree turn.
        supported = min((supported*symmetry for symmetry in Rotation.create_group('O')),
                        key=lambda candidate: candidate.magnitude())
    pose[3:] = supported.as_quat(scalar_first=True)
    return replace(observation, pose=pose)
