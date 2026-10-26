"""GPU 11-DoF pose IK with collision costs; not a base motion planner."""
from pathlib import Path
import json, copy, time, hashlib, xml.etree.ElementTree as ET
import numpy as np
import yaml
import torch
from scipy.spatial.transform import Rotation
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import JointState, Pose, GoalToolPose
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'ik_test/gpu_heading20';OUT.mkdir(exist_ok=True)
scene=json.loads((ROOT/'scene_config.json').read_text())
source=Path(scene['robot']['curobo_config']); raw=source.read_bytes(); cfg=yaml.safe_load(raw)
(OUT/'source_robot.yml').write_bytes(raw)
posture=json.loads(Path(scene['robot']['initial_posture']).read_text());(OUT/'initial_posture.json').write_text(json.dumps(posture,indent=2))
old=json.loads((ROOT/'ik_test/random20/result.json').read_text()); prev=json.loads((ROOT/'ik_test/result.json').read_text())
k=cfg['kinematics']; urdf=Path(k['urdf_path']);tree=ET.parse(urdf);robot=tree.getroot()
for m in robot.findall('.//mesh'):
    m.set('filename',str((urdf.parent/m.attrib['filename']).resolve()))
for name in ('planning_world','planning_x','planning_y','planning_yaw'):
    ET.SubElement(robot,'link',name=name)
for name,parent,child,axis,typ,lo,hi,vel in [('base_x','planning_world','planning_x','1 0 0','prismatic',-3.1,3.1,1.5),('base_y','planning_x','planning_y','0 1 0','prismatic',-3.1,3.1,1.5),('base_yaw','planning_y','planning_yaw','0 0 1','revolute',-3.14159265,3.14159265,1.)]:
    j=ET.SubElement(robot,'joint',name=name,type=typ);ET.SubElement(j,'parent',link=parent);ET.SubElement(j,'child',link=child);ET.SubElement(j,'axis',xyz=axis);ET.SubElement(j,'limit',lower=str(lo),upper=str(hi),velocity=str(vel),effort='1000')
j=ET.SubElement(robot,'joint',name='planning_mount',type='fixed');ET.SubElement(j,'parent',link='planning_yaw');ET.SubElement(j,'child',link=k['base_link']);ET.SubElement(j,'origin',xyz=f"0 0 {prev['base_world_transform'][2][3]}",rpy='0 0 0')
robot.find("joint[@name='platform_joint']/limit").set('upper','0.79')
tree.write(OUT/'whole_body.urdf')
values=dict(zip(posture['joint_names'],posture['joint_positions_rad']));values.update(k['lock_joints'])
oldcs=copy.deepcopy(k['cspace']); oldnames=oldcs['joint_names']; names=['base_x','base_y','base_yaw','platform_joint']+[f'r_joint{i}' for i in range(1,8)]
k['base_link']='planning_world';k['urdf_path']=str(OUT/'whole_body.urdf');k['tool_frames']=['r_grasp_tcp'];k['lock_joints'].pop('platform_joint')
for n in [f'l_joint{i}' for i in range(1,8)]:k['lock_joints'][n]=values[n]
cs=k['cspace'];cs['joint_names']=names
prefix={'default_joint_position':[-1.3,.5,3.0,.5],'null_space_weight':[.01,.01,.01,1.], 'cspace_distance_weight':[1.,1.,1.,1.], 'max_acceleration':[1.,1.,1.,.3], 'max_jerk':[5.,5.,5.,1.]}
for field,v in oldcs.items():
    if field!='joint_names' and isinstance(v,list) and len(v)==len(oldnames):
        assert field in prefix,field
        cs[field]=prefix[field]+[v[oldnames.index(n)] for n in names[4:]]
(OUT/'whole_body.yml').write_text(yaml.safe_dump(cfg,sort_keys=False))
world=yaml.safe_load((ROOT/'static_world.yml').read_text());world['cuboid'].pop('Floor')
# Bottle bounding box is conservative; support floor excluded at fixed mount height.
cup=prev['cup'];world['cuboid']['Bottle']={'dims':[2*cup['radius_m'],2*cup['radius_m'],cup['height_m']],'pose':cup['center_m']+[1,0,0,0]}
(OUT/'world.yml').write_text(yaml.safe_dump(world))
start=time.perf_counter()
ik=InverseKinematics(InverseKinematicsCfg.create(robot=cfg,scene_model=world,self_collision_check=True,num_seeds=32,max_batch_size=1,position_tolerance=.0001,orientation_tolerance=.001,optimizer_collision_activation_distance=.002,use_cuda_graph=True))
torch.cuda.synchronize(); setup=time.perf_counter()-start
assert list(ik.joint_names)==names,(ik.joint_names,names)
T=np.array(prev['selected']['target_tcp_world']);quat=Rotation.from_matrix(T[:3,:3]).as_quat(scalar_first=True)
goal=GoalToolPose.from_poses({'r_grasp_tcp':Pose(position=torch.tensor(T[:3,3][None],device='cuda',dtype=torch.float32),quaternion=torch.tensor(quat[None],device='cuda',dtype=torch.float32))},num_goalset=1)
rows=[]
rng=np.random.default_rng(20261026)
max_attempts=5
for row in old['results']:
    q=[row['base_x_m'],row['base_y_m'],np.deg2rad(row['base_yaw_deg']),values['platform_joint']]+[values[n] for n in names[4:]]
    state=JointState.from_position(torch.tensor([q],device='cuda',dtype=torch.float32),joint_names=names)
    if not rows:
        t=time.perf_counter();ik.solve_pose(goal,current_state=state,return_seeds=32);torch.cuda.synchronize();warmup=time.perf_counter()-t
    torch.cuda.synchronize();t=time.perf_counter()
    seed_config=None; attempt_log=[]; found=False
    for attempt in range(max_attempts):
        result=ik.solve_pose(goal,current_state=state,seed_config=seed_config,return_seeds=32)
        torch.cuda.synchronize()
        ends=result.solution.detach().cpu().numpy().reshape(-1,len(names))
        successes=result.success.detach().cpu().numpy().reshape(-1)
        bearing=np.arctan2(cup['center_m'][1]-ends[:,1],cup['center_m'][0]-ends[:,0])
        # URDF forward = local -Y => forward world bearing = yaw - pi/2.
        angles=np.arctan2(np.sin(bearing-(ends[:,2]-np.pi/2)),np.cos(bearing-(ends[:,2]-np.pi/2)))
        valid=successes & (np.abs(angles)<=np.deg2rad(30))
        attempt_log.append({'attempt':attempt+1,'ik_success_count':int(successes.sum()),'heading_valid_count':int(valid.sum())})
        if np.any(valid):
            ids=np.flatnonzero(valid)
            # Pick the nearest feasible terminal base pose among returned solutions.
            chosen=int(ids[np.argmin(np.linalg.norm(ends[ids,:2]-np.array(q[:2]),axis=1))]);found=True
            break
        chosen=int(np.argmin(np.abs(angles)+np.where(successes,0,10)))
        seeds=ends.copy()
        # Retry with changed, front-facing seeds on the permitted table side.
        seeds[:,:2]=np.array(cup['center_m'][:2])+np.column_stack([rng.uniform(-.45,.45,32),rng.uniform(-.9,-.5,32)])
        bearing_seed=np.arctan2(cup['center_m'][1]-seeds[:,1],cup['center_m'][0]-seeds[:,0])
        seed_yaw=bearing_seed+np.pi/2+rng.uniform(-.2,.2,32)
        seeds[:,2]=np.arctan2(np.sin(seed_yaw),np.cos(seed_yaw))
        seeds[:,3]=rng.uniform(.4,.65,32)
        seed_config=torch.tensor(seeds[None],device='cuda',dtype=torch.float32)
    elapsed=(time.perf_counter()-t)*1000
    end=ends[chosen].tolist()
    pe=result.position_error.detach().cpu().numpy().reshape(-1)
    re=result.rotation_error.detach().cpu().numpy().reshape(-1)
    r={'index':row['index'],'start_q':q,'end_q':end,'solver_success':bool(successes[chosen]),'heading_ok':bool(abs(angles[chosen])<=np.deg2rad(30)), 'heading_error_deg':float(np.rad2deg(angles[chosen])), 'attempts':attempt+1,'attempt_log':attempt_log,'wall_ms':elapsed,'position_error_m':float(pe[chosen]),'rotation_error_rad':float(re[chosen])}
    rows.append(r);print(json.dumps(r),flush=True)
report={'backend':'cuRobo V2 CUDA multi-seed IK','gpu':torch.cuda.get_device_name(),'source_config':str(source),'source_sha256':hashlib.sha256(raw).hexdigest(),'sphere_count':sum(len(v) for v in k['collision_spheres'].values()),'joint_names':names,'urdf':str(OUT/'whole_body.urdf'),'setup_s':setup,'warmup_s':warmup,'seeds_per_query':32,'base_angular_speed_status':'placeholder for pose IK only, not measured hardware limit','target_tcp_world':T.tolist(),'cup':cup,'self_collision_check':True,'environment_collision_check':True,'floor_support_excluded':True,'trajectory_validated':False,'rows':rows,'heading_limit_deg':30,'heading_rule':'bearing from final base XY to bottle, relative to base local -Y','heading_enforcement':'hard output acceptance + front-facing retry seeds, not an added CUDA optimizer cost','max_attempts':max_attempts}
(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print('DONE',sum(r['solver_success'] for r in rows), 'mean ms',np.mean([r['wall_ms'] for r in rows]),flush=True)
