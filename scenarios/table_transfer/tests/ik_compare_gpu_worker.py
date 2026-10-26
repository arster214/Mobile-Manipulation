"""Persistent CUDA worker; JSON protocol is private local stdin/stdout."""
import json,sys,time
from pathlib import Path
import numpy as np,torch,yaml
from scipy.spatial.transform import Rotation
from curobo.inverse_kinematics import InverseKinematics,InverseKinematicsCfg
from curobo.types import JointState,Pose,GoalToolPose
P=Path(__file__).resolve().parents[1]/'ik_test/gpu_heading20'
r=json.loads((P/'result.json').read_text());cfg=yaml.safe_load((P/'whole_body.yml').read_text());world=yaml.safe_load((P/'world.yml').read_text())
t=time.perf_counter();ik=InverseKinematics(InverseKinematicsCfg.create(robot=cfg,scene_model=world,self_collision_check=True,num_seeds=32,seed_solver_num_seeds=32,max_batch_size=1,position_tolerance=.0001,orientation_tolerance=.001,optimizer_collision_activation_distance=.002,use_cuda_graph=True))
T=np.array(r['target_tcp_world']);quat=Rotation.from_matrix(T[:3,:3]).as_quat(scalar_first=True)
goal=GoalToolPose.from_poses({'r_grasp_tcp':Pose(position=torch.tensor(T[:3,3][None],device='cuda',dtype=torch.float32),quaternion=torch.tensor(quat[None],device='cuda',dtype=torch.float32))},num_goalset=1)
state=JointState.from_position(torch.tensor([r['rows'][0]['start_q']],device='cuda',dtype=torch.float32),joint_names=r['joint_names'])
seed=state.position[:,None,:].repeat(1,32,1)
ik.solve_pose(goal,current_state=state,seed_config=seed,return_seeds=32);torch.cuda.synchronize()
print('COMPARE_READY '+json.dumps({'setup_warmup_s':time.perf_counter()-t,'gpu':torch.cuda.get_device_name()}),flush=True)
for line in sys.stdin:
 try:
  req=json.loads(line)
  if req.get('stop'):break
  state=JointState.from_position(torch.tensor([req['start']],device='cuda',dtype=torch.float32),joint_names=r['joint_names'])
  seed=torch.tensor([req['seeds']],device='cuda',dtype=torch.float32)
  torch.cuda.synchronize();t=time.perf_counter();res=ik.solve_pose(goal,current_state=state,seed_config=seed,return_seeds=32);torch.cuda.synchronize()
  output={'solve_ms':(time.perf_counter()-t)*1000,'solutions':res.solution.detach().cpu().numpy().reshape(-1,11).tolist(),'solver_success':res.success.detach().cpu().numpy().reshape(-1).tolist()}
  print('COMPARE_RESULT '+json.dumps(output),flush=True)
 except Exception as e: print('COMPARE_ERROR '+repr(e),flush=True)
