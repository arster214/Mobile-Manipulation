"""Recheck saved success with current collision kernels; no execution claims."""
from pathlib import Path
import json,sys
import numpy as np,yaml
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from planning.interfaces import RobotState,PlannedTrajectory,Status
from planning.curobo_backend import CuroboPlanner
from planning.vision_bridge import BoxPerception
from planning.perception import ObjectObservation
from planning.arm_comparison import state_at
from planning.model import state_vector
from planning.carried_object import CarriedObject
from planning.task_search import released_object_audit
folder=ROOT/'tests/artifacts/pnp/baseline';out=folder.parent/'numerical_audit';out.mkdir(exist_ok=True)
r=json.loads((folder/'result.json').read_text());scene=json.loads((folder/'scene.json').read_text());settings=json.loads((ROOT/'planning/config.json').read_text())
source=Path(scene['robot']['curobo_config']);k=yaml.safe_load(source.read_text())['kinematics']
initial=json.loads(Path(scene['robot']['initial_posture']).read_text());values=dict(zip(initial['joint_names'],initial['joint_positions_rad']));values.update(k['lock_joints'])
z=json.loads((ROOT/'scene_manifest.json').read_text())['robot_root_z_m'];b=r['initial_base'];state=RobotState(list(values),np.array(list(values.values())),np.zeros(len(values)),0.,np.r_[b[:2],z,Rotation.from_euler('z',b[2]).as_quat(scalar_first=True)])
reference=state;side=r['selected_arm'];world=yaml.safe_load((folder/'world.yml').read_text())
perception=BoxPerception(world,[.04]*3);env=perception.environment(ObjectObservation(np.array(r['object_pose']),0.,1));attachment=None;rows=[]
for name in ('pregrasp','approach','lift','carry','place','withdraw'):
 data=dict(np.load(folder/(name+'.npz')));names=data['joint_names'].tolist();arm_only=names==[f'{side}_joint{i}' for i in range(1,8)]
 if name=='approach':del env.obstacles['cuboid'][perception.object_name]
 planner=CuroboPlanner(source,state,env,settings,out/name,arm_only=arm_only,arm=side,reference_state=reference)
 planner.motion_phase='carry' if name=='carry' else 'approach'
 plan=PlannedTrajectory(names,data['positions'],data['velocities'],data['accelerations'],data['times'],state,1,1,
  ik_status=Status.PASSED,planning_status=Status.PASSED,base_poses=data['base_poses'],base_controls=data['base_controls'])
 if name=='lift':attachment=CarriedObject.attach_box(planner.tcp_pose(state_vector(state,planner.names)[None])[0],np.array(r['object_pose']),[.04]*3,side)
 if attachment is not None:
  planner.extra_validation=(lambda q:released_object_audit(planner,q,np.array(r['object_goal_pose']),attachment)) if name=='withdraw' else (lambda q:attachment.audit(planner,q,env))
 passed=planner.validate(plan);rows.append(dict(stage=name,passed=passed,reason=plan.reason,constraints=plan.constraint_metrics))
 q=plan.positions[-1] if arm_only else np.r_[plan.base_poses[-1],plan.positions[-1]]
 state=state_at(state,planner.names,q)
 print(name,passed,flush=True)
(out/'result.json').write_text(json.dumps(rows,indent=2));assert all(x['passed'] for x in rows)
