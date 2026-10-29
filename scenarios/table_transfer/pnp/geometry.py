"""Support-relative geometry; no table numbers or world-axis retreat offsets."""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from .contracts import Fault, Pose


def matrix(pose):
    q = np.asarray(pose.wxyz, dtype=float)
    xyz = np.asarray(pose.xyz, dtype=float)
    if (pose.frame != 'world' or not pose.valid or q.shape != (4,) or xyz.shape != (3,)
            or not np.isfinite([*pose.xyz, *q, pose.timestamp]).all()
            or abs(np.linalg.norm(q)-1)>1e-4):
        raise Fault('invalid_observation', 'Expected valid metric world pose and normalized wxyz')
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(q, scalar_first=True).as_matrix()
    t[:3, 3] = pose.xyz
    return t


def corners(transform, dimensions):
    local = np.array(list(itertools.product((-1., 1.), repeat=3)))*np.asarray(dimensions)/2
    return local@transform[:3, :3].T+transform[:3, 3]


def on_surface(pose, dimensions, surface, margin, tolerance):
    relative = np.linalg.inv(matrix(surface.pose))@matrix(pose)
    vertices = corners(relative, dimensions)
    inside = bool(np.all(abs(vertices[:, :2]) <= np.array(surface.size_xy)/2-margin))
    supported = abs(float(vertices[:, 2].min())) <= tolerance
    return inside, supported, float(vertices[:, 2].min())


def derive(goal, snapshot, config):
    execution = config['execution']
    target = snapshot.surfaces[goal.destination_id]
    source = snapshot.surfaces[goal.source_id]
    dims = np.asarray(goal.dimensions)
    if dims.shape != (3,) or not np.isfinite(dims).all() or not np.allclose(
            dims, config['objects_environment']['cube_dimensions_m'], atol=1e-9, rtol=0):
        raise Fault('unsupported_geometry', 'First version supports the configured 4 cm cube only')
    low, high = config['objects_environment']['table_height_range_m']
    for surface in (source, target):
        size = np.asarray(surface.size_xy, dtype=float)
        if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise Fault('invalid_observation', 'Support must have a finite positive rectangular region')
        transform = matrix(surface.pose)
        if (not low <= transform[2, 3] <= high
                or not np.allclose(transform[:3, 2], [0, 0, 1], atol=1e-5)):
            raise Fault('unsupported_geometry', 'Horizontal support height outside declared range')
    local = np.eye(4)
    local[:3, :3] = Rotation.from_euler('z', goal.placement_yaw).as_matrix()
    local[:3, 3] = [*goal.placement_xy, dims[2]/2]
    vertices = corners(local, dims)
    if np.any(abs(vertices[:, :2]) > np.asarray(target.size_xy)/2-execution['edge_margin_m']):
        raise Fault('invalid_goal', 'Object footprint including yaw and edge margin leaves target region')
    opening = float(max(dims[:2])+execution['opening_clearance_m'])
    if opening > config['robot_calibration']['max_opening_m']:
        raise Fault('unsupported_geometry', 'Object exceeds gripper opening')
    return {'object_goal': (matrix(target.pose)@local).tolist(), 'opening_m': opening,
            'approach_m': float(max(dims)*execution['approach_object_lengths']),
            'lift_m': float(dims[2]*execution['lift_object_heights']),
            'withdraw_m': float(max(dims)*execution['approach_object_lengths'])}
