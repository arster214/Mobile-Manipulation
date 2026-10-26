"""Numerical lift/transport/place/withdraw from the saved successful grasp."""
from pathlib import Path
import sys,json,time
from dataclasses import replace
import numpy as np,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning.interfaces import RobotState,EnvironmentState,TaskTarget,Status,PlannedTrajectory
from planning.model import state_vector
from planning.arm_comparison import state_at
from planning.curobo_backend import CuroboPlanner
from planning.trajectory import candidate
from planning.carried_object import CarriedObject,pose_matrix,matrix_pose
from planning.selection import path_cost
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tests/artifacts/arm_placement';OLD=ROOT/'tests/artifacts/grasp_priority'


def main():
 started=time.perf_counter();timings={}
 OUT.mkdir(parents=True,exist_ok=True)
 old=json.loads((OLD/'result.json').read_text());scene=json.loads((ROOT/'scene_config.json').read_text());settings=json.loads((ROOT/'planning/config.json').read_text())
 source=Path(scene['robot']['curobo_config']);initial=old['start_state'];side=old['selected_arm']
 ref=RobotState(initial['joint_names'],np.array(initial['positions']),np.zeros(len(initial['positions'])),0.,np.array(initial['base_pose']))
 pre=np.load(OLD/side/'pregrasp.npz');insertion=np.load(OLD/side/'approach.npz')
 state=state_at(ref,['base_x','base_y','base_yaw']+pre['joint_names'].tolist(),np.r_[pre['base_poses'][-1],pre['positions'][-1]])
 state=state_at(state,insertion['joint_names'].tolist(),insertion['positions'][-1])
 start=state
 obstacles=yaml.safe_load((ROOT/'static_world.yml').read_text());obstacles['cuboid'].pop('Floor',None)
 world=EnvironmentState(obstacles,0.,version=2)
 report={'start_state':{'joint_names':state.joint_names,'positions':state.joint_positions.tolist(),'base_pose':state.base_pose.tolist()},
         'reference_state':initial,'object_start_pose':old['object_pose'],'object_goal_pose':[*scene['placement_center_m'],1,0,0,0],
         'stages':{},'complete':False,'scope':'numerical trajectories and sampled collisions, no grasp/release feedback or PhysX execution'}
 def make(snapshot,folder,attached=None,arm_only=False):
  planner=CuroboPlanner(source,snapshot,world,settings,OUT/folder,arm=side,arm_only=arm_only,reference_state=ref)
  if attached is not None:planner.extra_validation=lambda q:attached.audit(planner,q,world)
  return planner
 stage_started=time.perf_counter()
 first=make(state,'lift');grasp_tcp=first.tcp_pose(state_vector(state,first.names)[None])[0]
 attachment=CarriedObject.attach(grasp_tcp,np.array(old['object_pose']),scene['cup']['radius_m'],scene['cup']['height_m'],side)
 first.extra_validation=lambda q:attachment.audit(first,q,world)
 report['tcp_from_object']=attachment.tcp_from_object.tolist()
 def save(name,plan,planner,attached):
  score,terms=path_cost(plan,side,settings)
  report['stages'][name]={'validation':plan.validation_status.value,'reason':plan.reason,'duration_s':float(plan.times[-1]),
                         'cost':score,'cost_terms':terms,'object_metrics':plan.object_metrics,'constraint_metrics':plan.constraint_metrics,'motion_phase':plan.motion_phase}
  (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
  print(name,json.dumps(report['stages'][name]),flush=True)
  if plan.validation_status!=Status.PASSED:
   configs=plan.positions if planner.arm_only else np.column_stack([plan.base_poses,plan.positions]);qt=planner.tensor(configs[:,planner.to_internal])[None]
   kin=planner.checker.get_kinematics(qt)
   print('FAIL COSTS',name,'bounds',planner.checker.get_bound(qt).sum().item(),'self',planner.checker.get_self_collision(kin.robot_spheres).flatten().cpu().tolist(),flush=True)
   costs=planner.checker.get_collision_constraint(kin).cpu().numpy()[0]
   params=planner.checker.kinematics.config.kinematics_config
   links=[params.all_link_names[i] for i in params.link_sphere_idx_map.cpu().tolist()]
   print('WORLD LINKS',sorted(set(links[i] for i in np.where(costs.max(axis=0)>0)[0])),flush=True)
   return None
  configurations=plan.positions if planner.arm_only else np.column_stack([plan.base_poses,plan.positions])
  tcp=planner.tcp_pose(configurations)
  object_poses=np.array([matrix_pose(t) for t in attachment.poses(tcp)]) if attached else np.tile(report['object_goal_pose'],(len(tcp),1))
  saved_positions=np.tile(state_vector(plan.start_state,pre['joint_names'].tolist()),(len(plan.times),1))
  saved_velocities=np.zeros_like(saved_positions);saved_accelerations=np.zeros_like(saved_positions)
  for j,n in enumerate(plan.joint_names):
   idx=pre['joint_names'].tolist().index(n);saved_positions[:,idx]=plan.positions[:,j];saved_velocities[:,idx]=plan.velocities[:,j];saved_accelerations[:,idx]=plan.accelerations[:,j]
  np.savez(OUT/f'{name}.npz',joint_names=pre['joint_names'],positions=saved_positions,velocities=saved_velocities,
           accelerations=saved_accelerations,times=plan.times,base_poses=plan.base_poses,base_controls=plan.base_controls,
           object_poses=object_poses,tcp_poses=tcp,motion_phase=plan.motion_phase)
  return state_at(plan.start_state,planner.names,configurations[-1])
 def lift_plan(planner,goal):
  plan=candidate(planner.state,goal,{k:v[3:] for k,v in planner.limits.items()},settings,world.version,1)
  plan.ik_status=plan.planning_status=Status.PASSED;planner.validate(plan);return plan
 # Raise by the lift while preserving both arm configurations and bottle attitude.
 q=state_vector(state,first.names);q[3]-=scene['transfer']['lift_height_m']
 lifted=lift_plan(first,q);state=save('lift',lifted,first,True)
 if state is None:return
 timings['lift_planning_s']=time.perf_counter()-stage_started
 from planning.placement import plan_placement
 nominal=state_at(state,['base_x','base_y'],state.base_pose[:2]+np.array(scene['placement_center_m'][:2])-np.array(old['object_pose'][:2]))
 def placement_factory(snapshot,arm_only):
  return make(snapshot,'placement_arm' if arm_only else 'placement_search',attachment,arm_only=arm_only)
 stage_started=time.perf_counter()
 carry,place,placed_state,carry_planner,place_planner,search=plan_placement(state,nominal,report['object_goal_pose'],attachment,placement_factory,settings,scene['transfer']['lift_height_m'])
 timings['placement_and_carry_search_s']=time.perf_counter()-stage_started
 report['placement_search']=search
 state=save('carry',carry,carry_planner,True)
 if state is None:return
 state=save('place',place,place_planner,True)
 if state is None:return
 # The final bottle pose is measured from the solved placement, not snapped to its target.
 actual_object=attachment.poses(place_planner.tcp_pose(place.positions[-1:]))[0]
 report['object_goal_pose']=matrix_pose(actual_object).tolist()
 stage_started=time.perf_counter()
 # Release is a task boundary; solve a fresh arm-only withdrawal away from the tray.
 planner=make(state,'withdraw',arm_only=True)
 tcp=planner.tcp_pose(state_vector(state,planner.names)[None])[0]
 target_pose=tcp.copy();target_pose[1]-=scene['transfer']['withdraw_distance_m']
 target=TaskTarget(target_pose,np.array(report['object_goal_pose']),0.,tool_frame=f'{side}_grasp_tcp',target_type='withdraw')
 # Released-object collision against non-contact robot links (fingers are allowed to separate from contact).
 placed=pose_matrix(report['object_goal_pose'])
 def check_released(configurations):
  kin=planner.checker.get_kinematics(planner.tensor(configurations[:,planner.to_internal])[None])
  balls=kin.robot_spheres[0].cpu().numpy();params=planner.checker.kinematics.config.kinematics_config
  names=[params.all_link_names[i] for i in params.link_sphere_idx_map.cpu().tolist()]
  local=(balls[:,:,:3]-placed[:3,3])@placed[:3,:3]
  gaps=np.linalg.norm(np.maximum(abs(local)-attachment.dimensions/2,0),axis=-1)-balls[:,:,3]
  active=np.array([name not in attachment.contact_links for name in names])
  valid=bool(np.all((gaps>=-1e-6)|(~active[None])|(balls[:,:,3]<=0)))
  return {'passed':valid,'object_state':'released and fixed in tray','finger_contact_allowed':True}
 planner.extra_validation=check_released
 withdraw=planner.plan_approach(target,tcp)
 state=save('withdraw',withdraw,planner,False)
 timings['withdraw_planning_s']=time.perf_counter()-stage_started
 timings['wall_time_s']=time.perf_counter()-started
 report['timing']=timings
 report['execution_durations_s']={name:stage['duration_s'] for name,stage in report['stages'].items()}
 report['execution_durations_s'].update(pregrasp=float(pre['times'][-1]),approach=float(insertion['times'][-1]))
 report['total_execution_duration_s']=sum(report['execution_durations_s'].values())
 report['complete']=state is not None
 report['release_requires_feedback']=True
 (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
 print('COMPLETE',report['complete'],flush=True)

if __name__=='__main__':main()
