"""Offline Pink differential IK from saved posture; no hardware communication."""
import json
import time
from pathlib import Path
import numpy as np
import yaml
import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask
from grasp_candidates import generate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'ik_test'


def main():
    OUT.mkdir(exist_ok=True)
    scene = json.loads((ROOT / 'scene_config.json').read_text())
    posture = json.loads(Path(scene['robot']['initial_posture']).read_text())
    cfg = yaml.safe_load(Path(scene['robot']['curobo_config']).read_text())['kinematics']
    tcp_definition = json.loads(Path(scene['robot']['grasp_tcp_definition']).read_text())
    policy = json.loads((ROOT / 'grasp_policy.json').read_text())
    manifest = json.loads((ROOT / 'scene_manifest.json').read_text())
    full = pin.buildModelFromUrdf(cfg['urdf_path'])
    q_full = pin.neutral(full)
    values = dict(zip(posture['joint_names'], posture['joint_positions_rad']))
    values.update(cfg['lock_joints'])
    # Pinocchio does not enforce mimic joints in this loaded model.
    for side in ('l', 'r'):
        values[f'{side}_Joint_finger2'] = values[f'{side}_Joint_finger1']
    for name, value in values.items():
        j = full.joints[full.getJointId(name)]
        assert j.nq == 1, name
        q_full[j.idx_q] = value
    world_from_base = pin.SE3(pin.rpy.rpyToMatrix(0., 0., np.deg2rad(scene['robot']['yaw_deg'])),
                             np.array([*scene['robot']['spawn_xy_m'], manifest['robot_root_z_m']]))
    table = next(t for t in scene['tables'] if t['name'] == policy['source_table'])
    # Explicit reproducible placement on source tabletop; keep existing cylinder dimensions.
    cup = {**scene['cup'], 'center_m': [-1.05, 1.32, table['size_m'][2] + scene['cup']['height_m']/2],
           'pose_source': 'offline_test_placement_on_tabletop'}
    candidates = generate(table, cup, policy)
    R_conversion = pin.rpy.rpyToMatrix(0., 0., policy['robot_grasp_tcp']['candidate_to_tcp_rotation_local_z_rad'])
    results = []
    for use_lift in (False, True):
        active = [f'r_joint{i}' for i in range(1, 8)] + (['platform_joint'] if use_lift else [])
        locks = [i for i in range(1, full.njoints) if full.names[i] not in active]
        model = pin.buildReducedModel(full, locks, q_full)
        names = list(model.names)[1:]
        q0 = np.array([values[name] for name in names])
        if use_lift:
            # Conservative intersection of URDF and existing real bridge ranges, not a calibration.
            j = model.joints[model.getJointId('platform_joint')]
            model.lowerPositionLimit[j.idx_q] = 0.
            model.upperPositionLimit[j.idx_q] = .79
        frame = scene['robot']['grasp_tcp_frames']['right']
        assert model.existFrame(frame)
        initial = pink.Configuration(model, model.createData(), q0)
        for tcp_name, definition in tcp_definition['frames'].items():
            actual = initial.get_transform_frame_to_world(definition['parent']).inverse() * initial.get_transform_frame_to_world(tcp_name)
            assert np.allclose(actual.translation, definition['xyz_m'], atol=1e-8)
            assert np.allclose(actual.rotation, pin.rpy.rpyToMatrix(*definition['rpy_rad']), atol=1e-8)
        for candidate in candidates['candidates']:
            configuration = pink.Configuration(model, model.createData(), q0.copy())
            target_world = pin.SE3(np.array(candidate['rotation_world_from_grasp']) @ R_conversion,
                                   np.array(candidate['pregrasp_center_world_m']))
            target_base = world_from_base.inverse() * target_world
            task = FrameTask(frame, position_cost=1., orientation_cost=1., lm_damping=1e-3)
            task.set_target(target_base)
            posture_task = PostureTask(cost=1e-6)
            posture_task.set_target(q0)
            trace = [q0.tolist()]
            start = time.perf_counter()
            error_message = None
            for step in range(1200):
                try:
                    velocity = pink.solve_ik(configuration, [task, posture_task], .02, solver='quadprog')
                    configuration.integrate_inplace(velocity, .02)
                except Exception as exc:
                    error_message = repr(exc)
                    break
                trace.append(configuration.q.tolist())
                actual = configuration.get_transform_frame_to_world(frame)
                position_error = np.linalg.norm(actual.translation - target_base.translation)
                rotation_error = np.linalg.norm(pin.log3(target_base.rotation.T @ actual.rotation))
                if position_error < 1e-4 and rotation_error < 1e-3:
                    break
            actual_world = world_from_base * configuration.get_transform_frame_to_world(frame)
            position_error = float(np.linalg.norm(actual_world.translation - target_world.translation))
            rotation_error = float(np.linalg.norm(pin.log3(target_world.rotation.T @ actual_world.rotation)))
            limits_ok = bool(np.all(configuration.q >= model.lowerPositionLimit - 1e-7) and np.all(configuration.q <= model.upperPositionLimit + 1e-7))
            solved = position_error < 1e-4 and rotation_error < 1e-3 and limits_ok
            result = {'mode': 'right_arm_and_lift' if use_lift else 'right_arm_only',
                      'candidate_rank': candidate['rank'], 'success': solved,
                      'iterations': len(trace)-1, 'wall_time_s': time.perf_counter()-start,
                      'position_error_m': position_error, 'orientation_error_rad': rotation_error,
                      'joint_limits_ok': limits_ok, 'joint_names': names,
                      'initial_joint_positions': q0.tolist(), 'joint_positions': configuration.q.tolist(),
                      'target_tcp_world': target_world.homogeneous.tolist(), 'actual_tcp_world': actual_world.homogeneous.tolist(),
                      'exception': error_message, 'iteration_trace': trace}
            results.append(result)
            print(json.dumps({k: result[k] for k in ('mode','candidate_rank','success','iterations','position_error_m','orientation_error_rad')}), flush=True)
    successful = [r for r in results if r['success']]
    preferred = sorted(successful, key=lambda r: (r['candidate_rank'], r['mode'] != 'right_arm_and_lift'))
    selected = preferred[0] if preferred else None
    output = {'pinocchio_version': pin.__version__, 'pink_version': getattr(pink,'__version__','unknown'),
              'robot_config': scene['robot']['curobo_config'], 'urdf': cfg['urdf_path'],
              'initial_posture': scene['robot']['initial_posture'], 'tcp_definition': tcp_definition,
              'base_world_transform': world_from_base.homogeneous.tolist(), 'cup': cup,
              'base_fixed': True, 'pregrasp_backoff_m': policy['pregrasp_backoff_m'],
              'lift_range_m_for_test': [0., .79], 'lift_range_status': 'provisional_intersection_not_hardware_calibrated',
              'collision_checked': False, 'execution_ready': False,
              'scope': 'Pose IK only; iteration trace is NOT a validated motion trajectory. No base navigation, collision avoidance or grasp physics.',
              'results': results, 'selected': selected}
    (OUT / 'result.json').write_text(json.dumps(output, indent=2)+'\n')
    (OUT / 'candidates.json').write_text(json.dumps(candidates, indent=2)+'\n')
    if selected:
        final_values = dict(values)
        final_values.update(zip(selected['joint_names'], selected['joint_positions']))
        q_result = q_full.copy()
        for name, value in final_values.items():
            q_result[full.joints[full.getJointId(name)].idx_q] = value
        data = full.createData()
        pin.framesForwardKinematics(full, data, q_result)
        transforms = {f.name: (world_from_base * data.oMf[i]).homogeneous.tolist() for i,f in enumerate(full.frames)}
        (OUT / 'selected_posture.json').write_text(json.dumps({'joint_positions': final_values, 'frame_transforms_world': transforms, 'cup': cup, 'execution_ready': False}, indent=2)+'\n')
    print('RESULT', OUT / 'result.json', 'SUCCESSES', len(successful), '/', len(results))


if __name__ == '__main__':
    main()
