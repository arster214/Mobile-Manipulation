"""Validate local radius correction using the actual cuRobo collision checker."""
from pathlib import Path
import sys,json
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from planning.interfaces import RobotState,EnvironmentState
from planning.curobo_backend import CuroboPlanner
from planning.model import state_vector
OUT=ROOT/'tests/artifacts/column_grid'
data=json.loads((ROOT/'tests/artifacts/transport/result.json').read_text());s=data['start_state'];rs=data['reference_state']
def state(s):return RobotState(s['joint_names'],np.array(s['positions']),np.zeros(len(s['positions'])),0.,np.array(s['base_pose']))
start=state(s);ref=state(rs);world=yaml.safe_load((ROOT/'static_world.yml').read_text());world['cuboid'].pop('Floor',None)
settings=json.loads((ROOT/'planning/config.json').read_text());results={}
for label,source in [('before',Path('/home/fang/robot_ws/postures/rm75_mobile_manipulation.yml')),('after',OUT/'candidate.yml')]:
 planner=CuroboPlanner(source,start,EnvironmentState(world,0.,version=2),settings,OUT/label,arm='r',reference_state=ref)
 q=state_vector(start,planner.names);qs=np.tile(q,(1001,1));qs[:,3]=np.linspace(q[3],q[3]-.1,1001)
 kin=planner.checker.get_kinematics(planner.tensor(qs[:,planner.to_internal])[None]);cost=planner.checker.get_self_collision(kin.robot_spheres).flatten().cpu().numpy()
 full=np.tile(q,(791,1));full[:,3]=np.linspace(0,.79,791)
 fullkin=planner.checker.get_kinematics(planner.tensor(full[:,planner.to_internal])[None]);fullcost=planner.checker.get_self_collision(fullkin.robot_spheres).flatten().cpu().numpy()
 stages={}
 for stage in ('lift','carry','place','withdraw'):
  arr=np.load(ROOT/f'tests/artifacts/transport/{stage}.npz');names=['base_x','base_y','base_yaw']+arr['joint_names'].tolist();values=np.column_stack([arr['base_poses'],arr['positions']]);values=values[:,[names.index(n) for n in planner.names]]
  stages[stage]=planner.collision_free(np.vstack([values,(values[:-1]+values[1:])/2]))
 results[label]={'full_sweep_self_collision_samples':int((fullcost>0).sum()),'full_sweep_samples':791,'full_sweep_max_self_cost':float(fullcost.max()),'lift_10cm_self_collision_samples':int((cost>0).sum()),'lift_10cm_max_self_cost':float(cost.max()),'lift_10cm_robot_collision_free':planner.collision_free(qs),'saved_5cm_transport_robot_collision_free':stages}
 print(label,results[label],flush=True)
(OUT/'gpu_validation.json').write_text(json.dumps(results,indent=2))
