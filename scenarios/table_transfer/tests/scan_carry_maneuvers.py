"""Sweep short start/end reverse adjustments and measure feasibility plateaus."""
from pathlib import Path
import sys,json,time
import numpy as np,yaml,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning.interfaces import RobotState,EnvironmentState,TaskTarget,Status
from planning.arm_comparison import state_at
from planning.model import state_vector
from planning.curobo_backend import CuroboPlanner
from planning.carried_object import CarriedObject,matrix_pose
from planning.trajectory import candidate
from planning.selection import path_cost
from itertools import product
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'tests/artifacts/random_room_resident';OUT=ROOT/'tests/artifacts/carry_maneuver_scan'
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
 limits={k:v[3:] for k,v in planner.limits.items()}
 def reverse_metrics(plan):
  dt=np.diff(plan.times);v=plan.base_controls[:,0];active=v<0
  starts=np.flatnonzero(active & ~np.r_[False,active[:-1]])
  ends=np.flatnonzero(active & ~np.r_[active[1:],False])+1
  return {'reverse_distance_m':float(np.sum(np.maximum(-v,0)*dt)),
          'reverse_time_s':float(np.sum(dt[active])),
          'max_reverse_burst_s':max([float(dt[a:b].sum()) for a,b in zip(starts,ends)],default=0.),
          'max_reverse_burst_m':max([float(np.sum(-v[a:b]*dt[a:b])) for a,b in zip(starts,ends)],default=0.),
          'reverse_bursts':len(starts)}
 warm=candidate(start,goal,limits,settings,world.version,1);warm.motion_phase='carry';planner.validate(warm)
 records=[];best=None;best_cost=float('inf');started=time.perf_counter()
 distances=[0.,.05,.1,.15,.2,.3,.4]
 variants=[(a,b,False) for a,b in product(distances,repeat=2)]+[(0.,0.,True)]
 for before,after,reverse in variants:
  for route,timing in product(['direct','x_then_y','y_then_x'],['together','early','late']):
   plan=candidate(start,goal,limits,settings,world.version,1,route,timing,reverse,backoff_m=(before,after));plan.motion_phase='carry'
   score,terms=path_cost(plan,side,settings)
   valid=planner.validate(plan)
   record={'before_m':before,'after_m':after,'full_reverse':reverse,'route':route,'upper_timing':timing,'success':valid,'cost':score,'time_penalty_s2':terms['carry_reverse_excess_time_squared_s2'],'duration_s':float(plan.times[-1]),**reverse_metrics(plan)}
   records.append(record)
   if valid and score<best_cost:best=plan;best_cost=score;best_record=record
  (OUT/folder.name/'candidates.json').write_text(json.dumps(records,indent=2)+'\n')
 torch.cuda.synchronize()
 row={'case':int(folder.name[-2:]),'candidates':len(records),'valid_candidates':sum(r['success'] for r in records),'scan_s':time.perf_counter()-started,'best':best_record if best is not None else None}
 if best is not None:
  q=np.column_stack([best.base_poses,best.positions]);tcp=planner.tcp_pose(q);object_poses=np.array([matrix_pose(t) for t in attachment.poses(tcp)])
  np.savez(OUT/folder.name/'carry.npz',joint_names=best.joint_names,positions=best.positions,velocities=best.velocities,accelerations=best.accelerations,times=best.times,base_poses=best.base_poses,base_controls=best.base_controls,tcp_poses=tcp,object_poses=object_poses,motion_phase='carry')
 rows.append(row);(OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
 del planner
