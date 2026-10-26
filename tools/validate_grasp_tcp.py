from pathlib import Path
import sys,json,copy
sys.path.insert(0,'/home/fang/robot_ws/curobo_v2')
import curobo,torch,yaml,numpy as np
from curobo.kinematics import Kinematics,KinematicsCfg
from curobo.types import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import init_warp
from curobo._src.cost.cost_self_collision import SelfCollisionCost
from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
R=Path('/home/fang/robot_ws');data=json.loads((R/'postures/clockwise_z_candidates.json').read_text());cfg=yaml.safe_load((R/'postures/rm75_mobile_manipulation.yml').read_text());device=DeviceCfg(device=torch.device('cuda:0'));init_warp(quiet=True,device_cfg=device)
kin=Kinematics(KinematicsCfg.from_data_dict(copy.deepcopy(cfg),device_cfg=device));sc=kin.get_self_collision_config();cost=SelfCollisionCost(SelfCollisionCostCfg(weight=1.,device_cfg=device,self_collision_kin_config=sc,store_pair_distance=True))
combinations=[(l,r) for l in data['solutions']['l'] for r in data['solutions']['r']];q=np.array([l['q']+r['q'] for l,r in combinations]);qt=torch.tensor(q,device='cuda',dtype=torch.float32).unsqueeze(1)
cost.setup_batch_tensors(len(q),1);state=kin.compute_kinematics(JointState.from_position(qt,joint_names=data['active_joints']));distance=cost.forward(state.robot_spheres);torch.cuda.synchronize();dist=distance.detach().cpu().numpy().reshape(-1)
shape_ok=np.array([l['error']<.01 and r['error']<.01 and np.dot(l['tool_z'],[0,-1,0])>.999 and np.dot(r['tool_z'],[0,-1,0])>.999 and l['tool_x'][2]<-.999 and r['tool_x'][2]<-.999 and abs(l['tool_y'][2])<1e-5 and abs(r['tool_y'][2])<1e-5 for l,r in combinations]);valid=np.where((dist<=1e-7)&shape_ok)[0];rank=lambda i:combinations[i][0]['joint_change_norm']+combinations[i][1]['joint_change_norm'];idx=min(valid,key=rank) if len(valid) else int(dist.argmin())
assert len(valid), 'No collision-free aligned posture; keep current default'
report={'curobo_module':curobo.__file__,'gpu':torch.cuda.get_device_name(0),'config':data['config'],'candidates':len(q),'collision_free_candidates':len(valid),'selected_index':int(idx),'selected_self_collision_cost':float(dist[idx]),'joint_names':data['active_joints'],'joint_positions_rad':q[idx].tolist(),'left':combinations[idx][0],'right':combinations[idx][1],'sphere_count':int(state.robot_spheres.shape[-2]),'collision_pair_count':int(sc.collision_pairs.shape[0]),'sphere_padding_min_max_m':[float(sc.sphere_padding.min()),float(sc.sphere_padding.max())]}
sph=state.robot_spheres[idx,0].detach().cpu().numpy();pairs=sc.collision_pairs.detach().cpu().numpy();gaps=np.linalg.norm(sph[pairs[:,0],:3]-sph[pairs[:,1],:3],axis=1)-sph[pairs[:,0],3]-sph[pairs[:,1],3];report['minimum_checked_sphere_gap_m']=float(gaps.min())

report['tool_frames']=cfg['kinematics']['tool_frames']
assert report['selected_self_collision_cost']==0
(R/'postures/grasp_tcp_validation.json').write_text(json.dumps(report,indent=2))
print('GRASP_TCP_GPU_OK',report['tool_frames'],report['selected_self_collision_cost'])
