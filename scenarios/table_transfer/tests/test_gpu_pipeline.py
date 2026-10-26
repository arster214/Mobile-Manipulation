"""GPU integration check. Outputs are isolated under tests/artifacts/baseline."""
import json,sys
from pathlib import Path
from dataclasses import asdict
import numpy as np,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning.interfaces import RobotState,Status
from planning.perception import SimulationPerception,ObjectObservation
from planning.curobo_backend import CuroboPlanner
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tests/artifacts/baseline';OUT.mkdir(parents=True,exist_ok=True)
scene=json.loads((ROOT/'scene_config.json').read_text());settings=json.loads((ROOT/'planning/config.json').read_text());policy=json.loads((ROOT/'grasp_policy.json').read_text())
source=Path(scene['robot']['curobo_config']);k=yaml.safe_load(source.read_text())['kinematics'];initial=json.loads(Path(scene['robot']['initial_posture']).read_text());manifest=json.loads((ROOT/'scene_manifest.json').read_text())
values=dict(zip(initial['joint_names'],initial['joint_positions_rad']));values.update(k['lock_joints'])
state=RobotState(list(values),np.array(list(values.values())),np.zeros(len(values)),0.,np.array([*scene['robot']['spawn_xy_m'],manifest['robot_root_z_m'],0,0,0,1]))
observation=ObjectObservation(np.array([-1.05,1.32,.81,1,0,0,0]),0.,1)
perception=SimulationPerception(yaml.safe_load((ROOT/'static_world.yml').read_text()),scene['cup']['radius_m'],scene['cup']['height_m'])
environment=perception.environment(observation);target=perception.targets(observation,scene['tables'][0],policy)[0]
import os
if os.getenv('DISABLE_CUDA_GRAPH'): settings['use_cuda_graph']=False
planner=CuroboPlanner(source,state,environment,settings,OUT)
from planning.model import state_vector
initial_valid=planner.collision_free(state_vector(state,planner.names)[None])
print('Initial collision free:',initial_valid,flush=True)
assert initial_valid, 'Saved starting posture must be collision-free'
plan=planner.plan_mobile(target)
report={'ik_status':plan.ik_status.value,'planning_status':plan.planning_status.value,'validation_status':plan.validation_status.value,'execution_status':plan.execution_status.value,'reason':plan.reason,'planning_time_s':plan.planning_time,'samples':len(plan.times),'duration_s':float(plan.times[-1]),'validation_scope':'command and midpoint collision samples, joint velocity/acceleration, differential-drive integration; not continuous collision proof or PhysX execution'}
(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if plan.validation_status==Status.PASSED:
 np.savez(OUT/'trajectory.npz',joint_names=plan.joint_names,positions=plan.positions,velocities=plan.velocities,accelerations=plan.accelerations,times=plan.times,base_poses=plan.base_poses,base_controls=plan.base_controls)
assert plan.validation_status==Status.PASSED,plan.reason
# Synthetic endpoint feedback is only a numerical integration fixture, not PhysX execution.
from dataclasses import replace
from scipy.spatial.transform import Rotation
endpoint=replace(state,joint_positions=state.joint_positions.copy(),joint_velocities=np.zeros_like(state.joint_positions),base_pose=state.base_pose.copy())
for name,value in zip(plan.joint_names,plan.positions[-1]):endpoint.joint_positions[endpoint.joint_names.index(name)]=value
endpoint.base_pose[:2]=plan.base_poses[-1,:2]
endpoint.base_pose[3:]=Rotation.from_euler('z',plan.base_poses[-1,2]).as_quat(scalar_first=True)
arm=CuroboPlanner(source,endpoint,environment,settings,OUT,arm_only=True)
q=state_vector(endpoint,arm.names)
pose=arm.checker.get_kinematics(arm.tensor(q[arm.to_internal][None,None])).tool_poses
tcp=np.r_[pose.position.cpu().numpy().reshape(-1,3)[0],pose.quaternion.cpu().numpy().reshape(-1,4)[0]]
report['endpoint_position_error_m']=float(np.linalg.norm(tcp[:3]-target.target_pose[:3]))
report['endpoint_orientation_error_rad']=float((Rotation.from_quat(tcp[3:],scalar_first=True).inv()*Rotation.from_quat(target.target_pose[3:],scalar_first=True)).magnitude())
assert report['endpoint_position_error_m'] <= settings['position_tolerance_m']
assert report['endpoint_orientation_error_rad'] <= settings['orientation_tolerance_rad']
try:
 approach=arm.plan_approach(replace(target,target_type='approach',target_pose=np.array(target.task_constraints['grasp_pose'])),tcp)
 report['approach']={'validation_status':approach.validation_status.value,'reason':approach.reason,'active_joints':approach.joint_names,'samples':len(approach.times)}
except RuntimeError as error:
 np.savez(OUT/'approach_partial.npz',joint_names=arm.names,positions=error.partial_positions,
          failed_target_pose=error.failed_target_pose,failed_fraction=error.failed_fraction,
          object_pose=observation.pose,pregrasp_pose=tcp,grasp_pose=target.task_constraints['grasp_pose'])
 report['approach']={'validation_status':'failed','reason':str(error),'active_joints':arm.names}
 # Diagnostic only: omit the target object to distinguish approach geometry from reachability.
 from copy import deepcopy
 diagnostic_world=deepcopy(environment.obstacles)
 diagnostic_world['cuboid'].pop('Cup')
 diagnostic=CuroboPlanner(source,endpoint,replace(environment,obstacles=diagnostic_world),settings,OUT/'diagnostic',arm_only=True)
 try:
  probe=diagnostic.plan_approach(replace(target,target_type='approach',target_pose=np.array(target.task_constraints['grasp_pose'])),tcp)
  report['approach']['without_target_object_diagnostic']=probe.validation_status.value
 except RuntimeError as probe_error:
  report['approach']['without_target_object_diagnostic']=str(probe_error)

(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
