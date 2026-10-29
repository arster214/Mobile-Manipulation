"""One measured snapshot -> validated trajectories, with shared planner policy."""
from pathlib import Path
import json,sys,logging,traceback
import numpy as np
from planning.interfaces import RobotState,TaskTarget,Status
from planning.perception import ObjectObservation
from planning.vision_bridge import BoxPerception
from planning.curobo_backend import CuroboPlanner
from planning.task_search import plan_pick_place,released_object_audit
from planning.placement import plan_placement
from planning.carried_object import CarriedObject,matrix_pose,pose_matrix
from planning.model import state_vector
from planning.arm_comparison import state_at,contact_environment


def run(folder):
    request=json.loads((folder/'request.json').read_text())
    settings=request['settings']; geometry=request['geometry']; goal=request['goal']
    saved=request['state']
    state=RobotState(saved['joint_names'],np.array(saved['joint_positions']),
                     np.zeros(len(saved['joint_names'])),saved['timestamp'],np.array(saved['base_pose']))
    source=Path(request['robot_source'])
    perception=BoxPerception(request['world'],goal['dimensions'])
    obs=ObjectObservation(np.array(request['object_pose']),state.timestamp,1)
    environment=perception.environment(obs)
    def factory(snapshot,world,arm_only,arm,reference):
        return CuroboPlanner(source,snapshot,world,settings,folder/'runtime'/arm,
                             arm_only=arm_only,arm=arm,reference_state=reference)
    target_pose=np.array(geometry['object_goal'])
    if request['mode']=='pick':
        policy=request['policy'];policy['pregrasp_backoff_m']=geometry['approach_m']
        targets={arm:perception.targets(obs,request['source_table'],policy,arm) for arm in ('l','r')}
        solution=plan_pick_place(state,environment,targets,matrix_pose(target_pose),factory,
                                 settings,geometry['lift_m'],geometry['withdraw_m'], allowed_arm=goal['arm'])
        arm=solution.grasp.arm
        pre,approach=solution.grasp.pregrasp,solution.grasp.approach
        contact=contact_environment(environment,solution.grasp.target)
        stages={'pregrasp':(pre,factory(pre.start_state,environment,False,arm,state)),
                'approach':(approach,factory(approach.start_state,contact,True,arm,state)),**solution.stages}
    else:
        arm=request['arm']
        attachment=CarriedObject(np.array(geometry['tcp_from_object']),np.array(goal['dimensions']),
                                 (arm+'_Link_finger1',arm+'_Link_finger2',arm+'_base_link_rmg24'))
        del environment.obstacles['cuboid'][perception.object_name]
        def make(snapshot,arm_only):
            p=factory(snapshot,environment,arm_only,arm,state)
            p.extra_validation=lambda q:attachment.audit(p,q,environment)
            return p
        nominal=state_at(state,['base_x','base_y'],state.base_pose[:2]+target_pose[:2,3]-obs.pose[:2])
        release={}
        def withdraw_check(placed,planner,lowering):
            actual=matrix_pose(attachment.poses(planner.tcp_pose(lowering.positions[-1:]))[0])
            p=factory(placed,environment,True,arm,state)
            tcp=p.tcp_pose(state_vector(placed,p.names)[None])[0]
            target=tcp.copy();target[:3]-=pose_matrix(tcp)[:3,2]*geometry['withdraw_m']
            p.extra_validation=lambda q:released_object_audit(p,q,actual,attachment)
            plan=p.plan_approach(TaskTarget(target,actual,state.timestamp,tool_frame=arm+'_grasp_tcp',target_type='withdraw'),tcp)
            if plan.validation_status!=Status.PASSED:raise RuntimeError(plan.reason)
            release['stage']=plan,p
        carry,place,_,cp,pp,_=plan_placement(state,nominal,matrix_pose(target_pose),attachment,make,settings,
                                           geometry['lift_m'],withdraw_check)
        stages={'carry':(carry,cp),'place':(place,pp),'withdraw':release['stage']}
    for name,(plan,planner) in stages.items():
        q=plan.positions if planner.arm_only else np.column_stack([plan.base_poses,plan.positions])
        np.savez(folder/(name+'.npz'),times=plan.times,joint_names=plan.joint_names,positions=plan.positions,
                 velocities=plan.velocities,accelerations=plan.accelerations,base_poses=plan.base_poses,
                 base_controls=plan.base_controls,tcp_poses=planner.tcp_pose(q))
    return {'complete':True,'selected_arm':arm,'object_pose':obs.pose.tolist(),
            'object_goal_pose':matrix_pose(target_pose).tolist(),'stages':list(stages)}


if __name__=='__main__':
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('planning.arm_comparison').setLevel(logging.INFO)
    folder=Path(sys.argv[1])
    try:result=run(folder)
    except RuntimeError as error:
        message=str(error)
        code='planning_budget_exhausted' if message.startswith(('No complete','No grasp')) else 'planner_error'
        traceback.print_exc()
        result={'complete':False,'reason':code,'message':message}
    except Exception as error:
        traceback.print_exc()
        result={'complete':False,'reason':'planner_error','message':repr(error)}
    (folder/'result.json').write_text(json.dumps(result,indent=2))
