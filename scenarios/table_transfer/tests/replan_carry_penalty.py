"""Replan saved carry endpoints with a soft carry-only reverse penalty."""
from pathlib import Path
import sys,json,time
import numpy as np,yaml,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning.interfaces import RobotState,EnvironmentState,TaskTarget,Status
from planning.arm_comparison import state_at
from planning.model import state_vector
from planning.curobo_backend import CuroboPlanner
from planning.carried_object import CarriedObject,matrix_pose
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'tests/artifacts/random_room_resident';OUT=ROOT/'tests/artifacts/carry_reverse_penalty'
scene=json.loads((ROOT/'scene_config.json').read_text());settings=json.loads((ROOT/'planning/config.json').read_text())
world=yaml.safe_load((ROOT/'static_world.yml').read_text());world['cuboid'].pop('Floor',None);world=EnvironmentState(world,0.,version=2)
rows=[]
for folder in sorted(OLD.glob('case_*')):
 if not (folder/'transport/carry.npz').exists():continue
 saved=json.loads((folder/'transport/result.json').read_text());grasp=json.loads((folder/'replay/result.json').read_text());side=grasp['selected_arm']
 r=saved['reference_state'];ref=RobotState(r['joint_names'],np.array(r['positions']),np.zeros(len(r['positions'])),0.,np.array(r['base_pose']))
 lift=np.load(folder/'transport/lift.npz');old=np.load(folder/'transport/carry.npz')
 names=['base_x','base_y','base_yaw']+lift['joint_names'].tolist()
 start=state_at(ref,names,np.r_[lift['base_poses'][-1],lift['positions'][-1]])
 planner=CuroboPlanner(Path(scene['robot']['curobo_config']),start,world,settings,OUT/folder.name/'model',arm=side,reference_state=ref)
 planner.motion_phase='carry'
 attachment=CarriedObject(np.array(saved['tcp_from_object']),np.array([2*scene['cup']['radius_m']]*2+[scene['cup']['height_m']]),(f'{side}_Link_finger1',f'{side}_Link_finger2',f'{side}_base_link_rmg24'))
 planner.extra_validation=lambda q:attachment.audit(planner,q,world)
 end=state_at(ref,['base_x','base_y','base_yaw']+old['joint_names'].tolist(),np.r_[old['base_poses'][-1],old['positions'][-1]])
 goal=state_vector(end,planner.names);target=TaskTarget(planner.tcp_pose(goal[None])[0],old['object_poses'][-1],0.,tool_frame=f'{side}_grasp_tcp',target_type='carry')
 planner.plan_mobile(target,goal_configs=[goal]);torch.cuda.synchronize();started=time.perf_counter()
 plan=planner.plan_mobile(target,goal_configs=[goal]);torch.cuda.synchronize();elapsed=time.perf_counter()-started
 row={'case':int(folder.name[-2:]),'success':plan.validation_status==Status.PASSED,'warm_planning_s':elapsed,'reason':plan.reason,'old_reverse_distance_m':float(np.sum(np.maximum(-old['base_controls'][:,0],0)*np.diff(old['times']))),'path_candidates':planner.search_stats['path_candidates']}
 if row['success']:
  configurations=np.column_stack([plan.base_poses,plan.positions]);tcp=planner.tcp_pose(configurations)
  object_poses=np.array([matrix_pose(t) for t in attachment.poses(tcp)])
  np.savez(OUT/folder.name/'carry.npz',joint_names=plan.joint_names,positions=plan.positions,velocities=plan.velocities,accelerations=plan.accelerations,times=plan.times,base_poses=plan.base_poses,base_controls=plan.base_controls,tcp_poses=tcp,object_poses=object_poses,motion_phase='carry')
  row.update(duration_s=float(plan.times[-1]),min_forward_speed=float(plan.base_controls[:,0].min()),reverse_distance_m=float(np.sum(np.maximum(-plan.base_controls[:,0],0)*np.diff(plan.times))),reverse_time_s=float(np.sum(np.diff(plan.times)[plan.base_controls[:,0]<0.])),cost_terms=plan.cost_terms)
 rows.append(row);(OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
 del planner
