"""Ablation of failed approach IK; diagnostic solutions are never executable."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'tests/artifacts/arm_comparison/diagnosis'


def cpu(request_path):
    import pinocchio as pin
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    request=json.loads(Path(request_path).read_text())
    model=pin.buildModelFromUrdf(request['urdf']);data=model.createData()
    full=pin.neutral(model)
    values=request['locks'].copy();values.update(zip(request['names'],request['seed']))
    for arm in ('l','r'):values[f'{arm}_Joint_finger2']=values[f'{arm}_Joint_finger1']
    for name,value in values.items():
        if model.existJointName(name):full[model.joints[model.getJointId(name)].idx_q]=value
    indices=[model.joints[model.getJointId(n)].idx_q for n in request['names']]
    frame=model.getFrameId(request['tool'])
    results={}
    for label,pose in request['targets'].items():
        desired=pin.SE3(Rotation.from_quat(pose[3:],scalar_first=True).as_matrix(),np.array(pose[:3]))
        def residual(q):
            full[indices]=q
            pin.framesForwardKinematics(model,data,full)
            current=data.oMf[frame]
            return np.r_[current.translation-desired.translation,pin.log3(desired.rotation.T@current.rotation)]
        for bound_kind in ('restricted','physical'):
            lo,hi=np.array(request['bounds'][bound_kind])
            rng=np.random.default_rng(7)
            seeds=[np.clip(request['seed'],lo+1e-8,hi-1e-8)]+[rng.uniform(lo,hi) for _ in range(23)]
            solutions=[];best=None
            for seed in seeds:
                fit=least_squares(residual,seed,bounds=(lo,hi),max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-10)
                err=residual(fit.x);pe=float(np.linalg.norm(err[:3]));re=float(np.linalg.norm(err[3:]))
                if best is None or pe+re<best['position_error_m']+best['rotation_error_rad']:
                    best={'q':fit.x.tolist(),'position_error_m':pe,'rotation_error_rad':re}
                if pe<=.005 and re<=.05:
                    solutions.append({'q':fit.x.tolist(),'position_error_m':pe,'rotation_error_rad':re})
            results[label+'_'+bound_kind]={'solutions':solutions,'best':best}
    Path(request_path).with_suffix('.solutions.json').write_text(json.dumps(results))


def gpu():
    import subprocess,xml.etree.ElementTree as ET
    import yaml
    from dataclasses import replace
    from copy import deepcopy
    from scipy.spatial.transform import Rotation
    from planning.interfaces import RobotState,TaskTarget,EnvironmentState
    from planning.curobo_backend import CuroboPlanner
    from planning.model import state_vector
    OUT.mkdir(parents=True,exist_ok=True)
    report=json.loads((OUT.parent/'result.json').read_text());scene=json.loads((ROOT/'scene_config.json').read_text())
    source=Path(scene['robot']['curobo_config']);source_cfg=yaml.safe_load(source.read_text())['kinematics']
    source_urdf=ET.parse(source_cfg['urdf_path']).getroot()
    settings=json.loads((ROOT/'planning/config.json').read_text())
    initial=report['start_state']
    ref=RobotState(initial['joint_names'],np.array(initial['positions']),np.zeros(len(initial['positions'])),0.,np.array(initial['base_pose']))
    world=yaml.safe_load((ROOT/'static_world.yml').read_text());world['cuboid'].pop('Floor',None)
    world['cuboid']['Cup']={'pose':report['object_pose'],'dims':[2*scene['cup']['radius_m']]*2+[scene['cup']['height_m']]}
    all_results={}
    for side in ('l','r'):
        trajectory=np.load(OUT.parent/side/'pregrasp.npz')
        state=replace(ref,joint_positions=ref.joint_positions.copy(),base_pose=ref.base_pose.copy())
        for name,value in zip(trajectory['joint_names'],trajectory['positions'][-1]):state.joint_positions[state.joint_names.index(name)]=value
        state.base_pose[:2]=trajectory['base_poses'][-1,:2]
        state.base_pose[3:]=Rotation.from_euler('z',trajectory['base_poses'][-1,2]).as_quat(scalar_first=True)
        planner=CuroboPlanner(source,state,EnvironmentState(world,0.),settings,OUT/side,True,side,ref)
        q0=state_vector(state,planner.names);tcp=planner.tcp_pose(q0[None])[0]
        target=TaskTarget(np.array(report['arms'][side]['grasp_pose']),np.array(report['object_pose']),0.,tool_frame=f'{side}_grasp_tcp')
        seed=q0;trace=[];failed=None
        for fraction in np.linspace(0,1,settings['approach_samples'])[1:]:
            pose=tcp.copy();pose[:3]+=fraction*(target.target_pose[:3]-tcp[:3])
            raw=planner.ik_candidates(replace(target,target_pose=pose),seed)
            metrics=[planner.posture_metrics(q[None]) for q in raw]
            valid=[q for q,m in zip(raw,metrics) if m['passed']]
            trace.append({'fraction':float(fraction),'raw_ik_success':len(raw),'posture_pass':len(valid),'metrics':metrics})
            if not valid:failed=pose.copy();break
            seed=min(valid,key=lambda q:np.linalg.norm(q-seed))
        physical=np.array([[float(source_urdf.find(f"joint[@name='{name}']/limit").get(key)) for name in planner.names] for key in ('lower','upper')])
        request={'urdf':planner.robot['kinematics']['urdf_path'],'locks':planner.robot['kinematics']['lock_joints'],
                 'names':planner.names,'tool':target.tool_frame,'seed':seed.tolist(),
                 'bounds':{'restricted':planner.bounds.tolist(),'physical':physical.tolist()},
                 'targets':{'failed':failed.tolist(),'final':np.r_[target.target_pose[:3],tcp[3:]].tolist()}}
        file=OUT/f'{side}.json';file.write_text(json.dumps(request))
        if '--reuse' not in sys.argv:
            subprocess.run(['/home/fang/robot_ws/pin_pink/python.sh',str(Path(__file__)), '--cpu',str(file)],check=True)
        solutions=json.loads(file.with_suffix('.solutions.json').read_text())
        no_cup=deepcopy(world);no_cup['cuboid'].pop('Cup')
        checker=CuroboPlanner(source,state,EnvironmentState(no_cup,0.),settings,OUT/side/'no_cup',True,side,ref)
        summary={}
        for label,group in solutions.items():
            rows=[]
            for item in group['solutions']:
                q=np.array(item['q'])[None];kin=planner.checker.get_kinematics(planner.tensor(q[:,planner.to_internal])[None])
                kin_no=checker.checker.get_kinematics(checker.tensor(q[:,checker.to_internal])[None])
                costs=planner.checker.get_collision_constraint(kin).flatten().cpu().numpy()
                params=planner.checker.kinematics.config.kinematics_config
                links=[params.all_link_names[i] for i in params.link_sphere_idx_map.cpu().tolist()]
                overlaps=sorted(set(links[i] for i in np.where(costs>0)[0]))
                rows.append({**item,'overlapping_world_links':overlaps,'posture':planner.posture_metrics(q),
                    'self_cost':float(planner.checker.get_self_collision(kin.robot_spheres).sum()),
                    'world_cost':float(planner.checker.get_collision_constraint(kin).sum()),
                    'world_without_cup_cost':float(checker.checker.get_collision_constraint(kin_no).sum())})
            summary[label]={'geometric_solutions':len(rows),'best_geometric_error':group['best'],
                'posture_pass':sum(r['posture']['passed'] for r in rows),
                'self_clear':sum(r['self_cost']==0 for r in rows),
                'world_clear':sum(r['world_cost']==0 for r in rows),
                'world_without_cup_clear':sum(r['world_without_cup_cost']==0 for r in rows),
                'all_pass':sum(r['posture']['passed'] and r['self_cost']==0 and r['world_cost']==0 for r in rows),
                'rows':rows}
        all_results[side]={'trace':trace,'ablation':summary}
        (OUT/'result.json').write_text(json.dumps(all_results,indent=2))
        print(side,json.dumps({'failed_step':trace[-1], 'summary':{k:{n:v for n,v in x.items() if n not in ('rows','best_geometric_error')} for k,x in summary.items()}}),flush=True)

if __name__=='__main__':
    cpu(sys.argv[2]) if '--cpu' in sys.argv else gpu()
