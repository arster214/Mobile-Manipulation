"""Same-start, same-world left/right constrained path comparison on CUDA."""
import json
import logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger('planning.arm_comparison').setLevel(logging.INFO)
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planning.interfaces import RobotState, Status
from planning.perception import SimulationPerception, ObjectObservation
from planning.curobo_backend import CuroboPlanner
from planning.model import state_vector
from planning.selection import path_cost, heading_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'tests/artifacts/grasp_priority'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scene = json.loads((ROOT/'scene_config.json').read_text())
    settings = json.loads((ROOT/'planning/config.json').read_text())
    policy = json.loads((ROOT/'grasp_policy.json').read_text())
    source = Path(scene['robot']['curobo_config'])
    model = yaml.safe_load(source.read_text())['kinematics']
    initial = json.loads(Path(scene['robot']['initial_posture']).read_text())
    manifest = json.loads((ROOT/'scene_manifest.json').read_text())
    values = dict(zip(initial['joint_names'], initial['joint_positions_rad']))
    values.update(model['lock_joints'])
    state = RobotState(list(values),np.array(list(values.values())),np.zeros(len(values)),0.,
                      np.array([*scene['robot']['spawn_xy_m'],manifest['robot_root_z_m'],0,0,0,1]))
    observation = ObjectObservation(np.array([-1.05,1.32,.81,1,0,0,0]),0.,1)
    perception = SimulationPerception(yaml.safe_load((ROOT/'static_world.yml').read_text()),
                                     scene['cup']['radius_m'],scene['cup']['height_m'])
    environment = perception.environment(observation)
    report = {'constraints':settings, 'jacobian':'selected arm geometric TCP Jacobian, 6x7, linear m/rad and angular rad/rad; world-aligned',
              'scope':'sampled numerical planning, no PhysX or hardware execution','arms':{},'selected_arm':None}
    from planning.arm_comparison import compare_arms
    targets = {side:perception.targets(observation,scene['tables'][0],policy,arm=side) for side in ('l','r')}
    def factory(snapshot,world,arm_only,arm,reference_state):
        return CuroboPlanner(source,snapshot,world,settings,OUT/arm,arm_only=arm_only,
                             arm=arm,reference_state=reference_state)
    candidates, preferred, selected = compare_arms(state,environment,targets,factory,settings)
    report.update(preferred_pregrasp_arm=preferred, selected_arm=selected)
    report['start_state']={'joint_names':state.joint_names,'positions':state.joint_positions.tolist(),
                           'base_pose':state.base_pose.tolist()}
    report['object_pose']=observation.pose.tolist()
    for side,item in candidates.items():
        plan=item.pregrasp
        row={'pregrasp_status':plan.validation_status.value,'reason':plan.reason,
             'planning_time_s':plan.planning_time,'search_stats':item.search_stats,
             'tool_frame':item.target.tool_frame,'target_pose':item.target.target_pose.tolist(),
             'grasp_pose':item.target.task_constraints['grasp_pose'],'eligible':item.complete,
             'cost':None,'pregrasp_cost':None,'approach_reason':item.approach_reason}
        if plan.validation_status==Status.PASSED:
            score,terms=path_cost(plan,side,settings)
            row.update(pregrasp_cost=score,cost_terms=terms,constraint_metrics=plan.constraint_metrics,
                       heading_deg=float(np.degrees(heading_error(plan.base_poses[-1],observation.pose))))
            np.savez(OUT/side/'pregrasp.npz',joint_names=plan.joint_names,positions=plan.positions,
                     velocities=plan.velocities,accelerations=plan.accelerations,times=plan.times,
                     base_poses=plan.base_poses,base_controls=plan.base_controls)
        row['approach_status']=item.approach.validation_status.value if item.approach is not None else 'failed'
        if item.complete:
            approach_cost,terms=path_cost(item.approach,side,settings)
            row.update(cost=score+approach_cost,approach_cost_terms=terms,
                       approach_constraint_metrics=item.approach.constraint_metrics)
            np.savez(OUT/side/'approach.npz',joint_names=item.approach.joint_names,
                     positions=item.approach.positions,times=item.approach.times)
        report['arms'][side]=row
        print(side,json.dumps(row),flush=True)
    if selected is None:
        report['selection_reason']='Pregrasp preference is provisional: no complete locked-base approach passed'
    (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PREFERRED',preferred,'SELECTED',selected,flush=True)

if __name__=='__main__':
    main()
