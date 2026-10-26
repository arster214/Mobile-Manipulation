"""Independent Pinocchio FK/Jacobian audit of grasp-first right-arm result."""
import json
from pathlib import Path
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tests/artifacts/grasp_first'
report=json.loads((OUT/'result.json').read_text())
side=report['selected_arm']
assert side is not None
model=pin.buildModelFromUrdf(str(OUT/side/'mobile.urdf'));data=model.createData()
mobile=np.load(OUT/side/'pregrasp.npz');approach=np.load(OUT/side/'approach.npz')
values=dict(zip(report['start_state']['joint_names'],report['start_state']['positions']))
reference=values.copy()
values.update(zip(mobile['joint_names'],mobile['positions'][-1]))
values.update(zip(['base_x','base_y','base_yaw'],mobile['base_poses'][-1]))
frame=model.getFrameId(f'{side}_grasp_tcp')
columns=[model.joints[model.getJointId(f'{side}_joint{i}')].idx_v for i in range(1,8)]
actual=[];sigmas=[];conditions=[];deltas=[]
for positions in approach['positions']:
 values.update(zip(approach['joint_names'],positions))
 for arm in ('l','r'):values[f'{arm}_Joint_finger2']=values[f'{arm}_Joint_finger1']
 q=pin.neutral(model)
 for name,value in values.items():
  if model.existJointName(name):q[model.joints[model.getJointId(name)].idx_q]=value
 pin.framesForwardKinematics(model,data,q)
 pose=data.oMf[frame]
 actual.append(np.r_[pose.translation.copy(),Rotation.from_matrix(pose.rotation).as_quat(scalar_first=True)])
 jac=pin.computeFrameJacobian(model,data,q,frame,pin.LOCAL_WORLD_ALIGNED)[:,columns]
 singular=np.linalg.svd(jac,compute_uv=False);sigmas.append(singular[-1]);conditions.append(singular[0]/singular[-1])
 deltas.append(max(abs(values[f'{arm}_joint{i}']-reference[f'{arm}_joint{i}']) for arm in ('l','r') for i in range(1,8)))
a=np.array(actual);target=np.array(report['arms'][side]['grasp_pose']);pre=np.array(report['arms'][side]['target_pose'])
line=np.linspace(pre[:3],target[:3],len(a))
audit={'max_sample_position_error_m':float(np.linalg.norm(a[:,:3]-line,axis=1).max()),
       'final_position_error_m':float(np.linalg.norm(a[-1,:3]-target[:3])),
       'max_rotation_error_rad':float((Rotation.from_quat(target[3:],scalar_first=True).inv()*Rotation.from_quat(a[:,3:],scalar_first=True)).magnitude().max()),
       'min_singular_value':float(min(sigmas)),'max_condition_number':float(max(conditions)),
       'max_arm_joint_change_rad':float(max(deltas)),
       'approach_joint_names':approach['joint_names'].tolist(),'cup_collision_excluded':True,
       'pregrasp_duration_s':float(mobile['times'][-1]),'approach_duration_s':float(approach['times'][-1])}
assert audit['max_sample_position_error_m']<=.005
assert audit['max_rotation_error_rad']<=.05
assert audit['min_singular_value']>=.005 and audit['max_condition_number']<=100
assert audit['max_arm_joint_change_rad']<=.8+1e-7
assert audit['approach_joint_names']==[f'{side}_joint{i}' for i in range(1,8)]
report['independent_pinocchio_audit']=audit
report['collision_policy']={'connection':'Cup, table, walls and robot self collision checked','contact':'Cup excluded; other environment and self collision checked'}
report['planning_order']='grasp IK -> locked-platform retreat -> start-to-pregrasp connection -> reverse retreat'
(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(audit,indent=2))
