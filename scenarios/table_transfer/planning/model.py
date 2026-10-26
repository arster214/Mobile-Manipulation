"""Whole-body joint semantics and task-facing base orientation."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .interfaces import RobotState

BASE_JOINTS = ['base_x', 'base_y', 'base_yaw']
UPPER_BODY_JOINTS = ['platform_joint'] + [f'{side}_joint{i}' for side in ('l', 'r') for i in range(1, 8)]
ACTIVE_JOINTS = BASE_JOINTS + UPPER_BODY_JOINTS


def front_angle_deg(state: RobotState, object_pose: np.ndarray) -> float:
    rotation = Rotation.from_quat(state.base_pose[3:], scalar_first=True).as_matrix()
    forward = rotation @ np.array([0., -1., 0.])
    delta = object_pose[:2] - state.base_pose[:2]
    return float(np.degrees(np.arctan2(forward[0]*delta[1]-forward[1]*delta[0], forward[:2] @ delta)))


def state_vector(state: RobotState, names: list[str]) -> np.ndarray:
    """Map measured base and articulation state by name, never array order."""
    yaw = Rotation.from_quat(state.base_pose[3:], scalar_first=True).as_euler('xyz')[2]
    values = dict(zip(state.joint_names, state.joint_positions))
    values.update(zip(BASE_JOINTS, [*state.base_pose[:2], yaw]))
    return np.array([values[name] for name in names])


def build_config(source, state: RobotState, target_directory, settings: dict,
                 arm_only: bool = False, arm: str = 'r', reference_state=None) -> dict:
    """Compose a simulation planning model; approach locks use measured values."""
    from pathlib import Path
    import xml.etree.ElementTree as ET
    import yaml
    source, target_directory = Path(source), Path(target_directory)
    cfg = yaml.safe_load(source.read_text())
    k = cfg['kinematics']
    urdf = Path(k['urdf_path'])
    reference_state = state if reference_state is None else reference_state
    reference = dict(zip(reference_state.joint_names, reference_state.joint_positions))
    tree = ET.parse(urdf)
    robot = tree.getroot()
    for mesh in robot.findall('.//mesh'):
        mesh.set('filename', str((urdf.parent / mesh.attrib['filename']).resolve()))
    for link in ['planning_world', 'planning_x', 'planning_y', 'planning_yaw']:
        ET.SubElement(robot, 'link', name=link)
    links = ['planning_world', 'planning_x', 'planning_y', 'planning_yaw']
    for i, (name, axis, limit, velocity) in enumerate(zip(
            BASE_JOINTS, ['1 0 0', '0 1 0', '0 0 1'], [3.5, 3.5, np.pi],
            [settings['base_max_linear_velocity_m_s']]*2 + [settings['simulation_base_yaw_rate_rad_s']])):
        joint = ET.SubElement(robot, 'joint', name=name, type='prismatic' if i < 2 else 'revolute')
        ET.SubElement(joint, 'parent', link=links[i])
        ET.SubElement(joint, 'child', link=links[i+1])
        ET.SubElement(joint, 'axis', xyz=axis)
        ET.SubElement(joint, 'limit', lower=str(-limit), upper=str(limit), velocity=str(velocity), effort='1000')
    joint = ET.SubElement(robot, 'joint', name='planning_mount', type='fixed')
    ET.SubElement(joint, 'parent', link='planning_yaw')
    ET.SubElement(joint, 'child', link=k['base_link'])
    ET.SubElement(joint, 'origin', xyz=f'0 0 {state.base_pose[2]}', rpy='0 0 0')
    robot.find("joint[@name='platform_joint']/limit").set('upper', str(settings['lift_max_m']))
    for side in ('l', 'r'):
        for i in range(1, 8):
            name = f'{side}_joint{i}'
            limit = robot.find(f"joint[@name='{name}']/limit")
            delta = settings.get('arm_joint_max_change_rad', .8)
            limit.set('lower', str(max(float(limit.get('lower')), reference[name]-delta)))
            limit.set('upper', str(min(float(limit.get('upper')), reference[name]+delta)))
    target_directory.mkdir(parents=True, exist_ok=True)
    model_path = target_directory / ('approach.urdf' if arm_only else 'mobile.urdf')
    tree.write(model_path)
    old = k['cspace']
    old_names = old['joint_names']
    measured = dict(zip(state.joint_names, state.joint_positions))
    measured.update(zip(BASE_JOINTS, state_vector(state, BASE_JOINTS)))
    active = [f'{arm}_joint{i}' for i in range(1, 8)] if arm_only else ACTIVE_JOINTS.copy()
    extras = {
        'default_joint_position': [*state_vector(state, BASE_JOINTS), measured['platform_joint']],
        'null_space_weight': [1., 1., 1., 1.],
        'cspace_distance_weight': [1., 1., 1., 1.],
        'max_acceleration': [1., 1., 1., settings['lift_acceleration_m_s2']],
        'max_jerk': [5., 5., 5., settings['lift_jerk_m_s3']],
    }
    cspace = dict(old)
    for key, value in old.items():
        if key == 'joint_names' or not isinstance(value, list) or len(value) != len(old_names):
            continue
        by_name = dict(zip(old_names, value))
        by_name.update(zip(ACTIVE_JOINTS[:4], extras[key]))
        cspace[key] = [by_name[name] for name in active]
    cspace['joint_names'] = active
    cspace['default_joint_position'] = [float(measured[name]) for name in active]
    for name in list(k['lock_joints']):
        if name in measured:
            k['lock_joints'][name] = float(measured[name])
    for name in ACTIVE_JOINTS:
        if name in active:
            k['lock_joints'].pop(name, None)
        else:
            k['lock_joints'][name] = float(measured[name])
    k.update(cspace=cspace, base_link='planning_world', urdf_path=str(model_path), tool_frames=[f'{arm}_grasp_tcp'])
    return cfg
