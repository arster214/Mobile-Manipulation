"""Independent Pinocchio audit of the measured starting arm Jacobians."""
from pathlib import Path
import json,yaml,numpy as np,pinocchio as pin
ROOT=Path(__file__).resolve().parents[1]
scene=json.loads((ROOT/'scene_config.json').read_text())
k=yaml.safe_load(Path(scene['robot']['curobo_config']).read_text())['kinematics']
initial=json.loads(Path(scene['robot']['initial_posture']).read_text())
model=pin.buildModelFromUrdf(k['urdf_path']);data=model.createData();q=pin.neutral(model)
values=dict(zip(initial['joint_names'],initial['joint_positions_rad']));values.update(k['lock_joints'])
for side in ('l','r'):values[f'{side}_Joint_finger2']=values[f'{side}_Joint_finger1']
for name,value in values.items():q[model.joints[model.getJointId(name)].idx_q]=value
for side in ('l','r'):
 j=pin.computeFrameJacobian(model,data,q,model.getFrameId(f'{side}_grasp_tcp'),pin.LOCAL_WORLD_ALIGNED)
 cols=[model.joints[model.getJointId(f'{side}_joint{i}')].idx_v for i in range(1,8)]
 s=np.linalg.svd(j[:,cols],compute_uv=False)
 print(side, 'sigma_min',s[-1],'condition',s[0]/s[-1])
# Audit saved GPU endpoints and the entire joint-change envelope independently.
path=ROOT/'tests/artifacts/arm_comparison/result.json'
if path.exists():
 report=json.loads(path.read_text())
 measured=dict(zip(report['start_state']['joint_names'],report['start_state']['positions']))
 for side,row in report['arms'].items():
  if row['pregrasp_status']!='passed':continue
  trajectory=np.load(path.parent/side/'pregrasp.npz')
  positions=trajectory['positions'];names=trajectory['joint_names'].tolist()
  endpoint=dict(measured);endpoint.update(zip(names,positions[-1]))
  for n,x in endpoint.items():
   if model.existJointName(n):q[model.joints[model.getJointId(n)].idx_q]=x
  for which in ('l','r'):
   n=f'{which}_Joint_finger2';q[model.joints[model.getJointId(n)].idx_q]=endpoint[f'{which}_Joint_finger1']
  jac=pin.computeFrameJacobian(model,data,q,model.getFrameId(f'{side}_grasp_tcp'),pin.LOCAL_WORLD_ALIGNED)
  cols=[model.joints[model.getJointId(f'{side}_joint{i}')].idx_v for i in range(1,8)]
  singular=np.linalg.svd(jac[:,cols],compute_uv=False)
  arm_names=[n for n in names if n.startswith(('l_joint','r_joint'))]
  delta=float(np.max(np.abs(positions[:,[names.index(n) for n in arm_names]]-np.array([measured[n] for n in arm_names]))))
  check={'sigma_min':float(singular[-1]),'condition':float(singular[0]/singular[-1]),'max_joint_delta_rad':delta}
  assert check['sigma_min']>=.005 and check['condition']<=100 and delta<=.8+1e-7,check
  row['independent_pinocchio_audit']=check
  print(side,'independent endpoint audit',check)
 path.write_text(json.dumps(report,indent=2)+'\n')
