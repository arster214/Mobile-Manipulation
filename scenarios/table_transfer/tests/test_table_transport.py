"""Numerical lift/transport/place/withdraw from the saved successful grasp."""
from pathlib import Path
import sys,json
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
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tests/artifacts/transport';OLD=ROOT/'tests/artifacts/grasp_first'


def main():
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
 def make(snapshot,folder,attached=None):
  planner=CuroboPlanner(source,snapshot,world,settings,OUT/folder,arm=side,reference_state=ref)
  if attached is not None:planner.extra_validation=lambda q:attached.audit(planner,q,world)
  return planner
 first=make(state,'lift');grasp_tcp=first.tcp_pose(state_vector(state,first.names)[None])[0]
 attachment=CarriedObject.attach(grasp_tcp,np.array(old['object_pose']),scene['cup']['radius_m'],scene['cup']['height_m'],side)
 first.extra_validation=lambda q:attachment.audit(first,q,world)
 report['tcp_from_object']=attachment.tcp_from_object.tolist()
 def save(name,plan,planner,attached):
  score,terms=path_cost(plan,side,settings)
  report['stages'][name]={'validation':plan.validation_status.value,'reason':plan.reason,'duration_s':float(plan.times[-1]),
                         'cost':score,'cost_terms':terms,'object_metrics':plan.object_metrics,'constraint_metrics':plan.constraint_metrics}
  (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
  print(name,json.dumps(report['stages'][name]),flush=True)
  if plan.validation_status!=Status.PASSED:
   configs=np.column_stack([plan.base_poses,plan.positions]);qt=planner.tensor(configs[:,planner.to_internal])[None]
   kin=planner.checker.get_kinematics(qt)
   print('FAIL COSTS',name,'bounds',planner.checker.get_bound(qt).sum().item(),'self',planner.checker.get_self_collision(kin.robot_spheres).flatten().cpu().tolist(),flush=True)
   costs=planner.checker.get_collision_constraint(kin).cpu().numpy()[0]
   params=planner.checker.kinematics.config.kinematics_config
   links=[params.all_link_names[i] for i in params.link_sphere_idx_map.cpu().tolist()]
   print('WORLD LINKS',sorted(set(links[i] for i in np.where(costs.max(axis=0)>0)[0])),flush=True)
   return None
  configurations=np.column_stack([plan.base_poses,plan.positions])
  tcp=planner.tcp_pose(configurations)
  object_poses=np.array([matrix_pose(t) for t in attachment.poses(tcp)]) if attached else np.tile(report['object_goal_pose'],(len(tcp),1))
  np.savez(OUT/f'{name}.npz',joint_names=plan.joint_names,positions=plan.positions,velocities=plan.velocities,
           accelerations=plan.accelerations,times=plan.times,base_poses=plan.base_poses,base_controls=plan.base_controls,
           object_poses=object_poses,tcp_poses=tcp)
  return state_at(plan.start_state,planner.names,configurations[-1])
 def lift_plan(planner,goal):
  plan=candidate(planner.state,goal,{k:v[3:] for k,v in planner.limits.items()},settings,world.version,1)
  plan.ik_status=plan.planning_status=Status.PASSED;planner.validate(plan);return plan
 # Raise by the lift while preserving both arm configurations and bottle attitude.
 q=state_vector(state,first.names);q[3]-=scene['transfer']['lift_height_m']
 lifted=lift_plan(first,q);state=save('lift',lifted,first,True)
 if state is None:return
 planner=make(state,'carry',attachment);q=state_vector(state,planner.names)
 shift=np.array(scene['placement_center_m'][:2])-np.array(old['object_pose'][:2]);q[:2]+=shift
 tcp=planner.tcp_pose(q[None])[0]
 target=TaskTarget(tcp,np.array(report['object_goal_pose']),0.,tool_frame=f'{side}_grasp_tcp')
 carry=planner.plan_mobile(target,goal_configs=[q]);state=save('carry',carry,planner,True)
 if state is None:return
 planner=make(state,'place',attachment);q=state_vector(state,planner.names)
 current_object=attachment.poses(planner.tcp_pose(q[None]))[0]
 q[3]+=current_object[2,3]-scene['placement_center_m'][2]
 place=lift_plan(planner,q);state=save('place',place,planner,True)
 if state is None:return
 # Release is a task boundary. The object remains in the tray; withdraw the arm along the reverse of the known insertion.
 planner=make(state,'withdraw')
 q0=state_vector(state,planner.names)
 arm_positions=insertion['positions'][::-1];arm_times=insertion['times'][-1]-insertion['times'][::-1]
 times=np.linspace(0,arm_times[-1],int(np.ceil(arm_times[-1]/settings['interpolation_dt_s']))+1)
 positions=np.tile(q0[3:],(len(times),1))
 for j,name in enumerate(insertion['joint_names']):positions[:,pre['joint_names'].tolist().index(name)]=np.interp(times,arm_times,arm_positions[:,j])
 velocity=np.gradient(positions,times,axis=0);acceleration=np.gradient(velocity,times,axis=0)
 limits={k:v[3:] for k,v in planner.limits.items()}
 scale=max(1.,float(np.max(abs(velocity)/limits['velocity'])),float(np.sqrt(np.max(abs(acceleration)/limits['acceleration']))))
 times*=scale
 withdraw=PlannedTrajectory(pre['joint_names'].tolist(),positions,velocity/scale,acceleration/scale**2,times,state,world.version,1,
      ik_status=Status.PASSED,planning_status=Status.PASSED,base_poses=np.tile(q0[:3],(len(times),1)),base_controls=np.zeros((len(times)-1,2)))
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
 planner.validate(withdraw);state=save('withdraw',withdraw,planner,False)
 report['complete']=state is not None
 report['release_requires_feedback']=True
 (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
 print('COMPLETE',report['complete'],flush=True)

if __name__=='__main__':main()
