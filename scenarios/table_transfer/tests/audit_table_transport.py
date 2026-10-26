"""Independent URDF FK check of saved TCP attachment and stage continuity."""
from pathlib import Path
from functools import partial
import json,sys
import numpy as np,yaml,yourdfpy
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from planning.carried_object import pose_matrix
OUT=Path(sys.argv[sys.argv.index('--directory')+1]) if '--directory' in sys.argv else ROOT/('tests/artifacts/arm_placement' if '--placement' in sys.argv else ('tests/artifacts/transport_priority' if '--priority' in sys.argv else 'tests/artifacts/transport'));report=json.loads((OUT/'result.json').read_text());scene=json.loads((ROOT/'scene_config.json').read_text());cfg=yaml.safe_load(Path(scene['robot']['curobo_config']).read_text())['kinematics'];urdf=Path(cfg['urdf_path'])
robot=yourdfpy.URDF.load(urdf,load_meshes=False,build_scene_graph=True,filename_handler=partial(yourdfpy.filename_handler_magic,dir=urdf.parent))
settings=json.loads((ROOT/'planning/config.json').read_text())
arm=sys.argv[sys.argv.index('--arm')+1] if '--arm' in sys.argv else 'r'
fixed=dict(zip(report['start_state']['joint_names'],report['start_state']['positions']));attachment=np.array(report['tcp_from_object']);metrics={};previous=None
for stage in ('lift','carry','place','withdraw'):
 if '--partial' in sys.argv and not (OUT/f'{stage}.npz').exists():continue
 p=np.load(OUT/f'{stage}.npz');errors=[];angles=[]
 for q,b,savedtcp,objectpose in zip(p['positions'],p['base_poses'],p['tcp_poses'],p['object_poses']):
  values=fixed.copy();values.update(zip(p['joint_names'],q));robot.update_cfg(values)
  base=np.eye(4);base[:3,:3]=Rotation.from_euler('z',b[2]).as_matrix();base[:3,3]=[*b[:2],report['start_state']['base_pose'][2]]
  tcp=base@robot.get_transform(f'{arm}_grasp_tcp');errors.append(np.linalg.norm(tcp[:3,3]-savedtcp[:3]));angles.append(Rotation.from_matrix(tcp[:3,:3].T@pose_matrix(savedtcp)[:3,:3]).magnitude())
  if stage!='withdraw':assert np.linalg.norm((tcp@attachment)[:3,3]-objectpose[:3])<1e-5
 assert max(errors)<settings['position_tolerance_m'] and max(angles)<settings['orientation_tolerance_rad']
 if previous is not None:
  assert np.max(abs(previous['positions'][-1]-p['positions'][0]))<1e-6
  assert np.max(abs(previous['base_poses'][-1]-p['base_poses'][0]))<1e-6
 if '--placement' in sys.argv and stage in ('place','withdraw'):
  assert np.max(np.ptp(p['base_poses'],axis=0))<1e-10
  locked=[j for j,n in enumerate(p['joint_names']) if n=='platform_joint' or n.startswith(('l' if arm=='r' else 'r')+'_joint')]
  assert np.max(np.ptp(p['positions'][:,locked],axis=0))<1e-10
 metrics[stage]={'independent_fk_max_position_error_m':float(max(errors)),'independent_fk_max_rotation_error_rad':float(max(angles)),'duration_s':float(p['times'][-1])};previous=p
placed=np.load(OUT/'place.npz')['object_poses'][-1]
metrics['placement_position_error_m']=float(np.linalg.norm(placed[:3]-scene['placement_center_m']))
metrics['placement_orientation_error_rad']=float(Rotation.from_quat(placed[3:],scalar_first=True).magnitude())
assert metrics['placement_position_error_m']<settings['position_tolerance_m']
assert metrics['placement_orientation_error_rad']<settings['orientation_tolerance_rad']
margin=np.array(scene['tray']['outer_size_m'])/2-scene['tray']['rim_thickness_m']-scene['cup']['radius_m']
assert np.all(abs(placed[:2]-scene['tray']['center_xy_m'])<=margin)
metrics['tray_inner_clearance_xy_m']=margin.tolist();metrics['passed']=True
(OUT/'independent_audit.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics),flush=True)
