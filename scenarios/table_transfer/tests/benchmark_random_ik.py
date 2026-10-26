"""20 reproducible collision-filtered starts; fixed-base right arm + lift IK."""
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
import yaml
import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'ik_test/random20'
OUT.mkdir(exist_ok=True)
scene = json.loads((ROOT/'scene_config.json').read_text())
previous = json.loads((ROOT/'ik_test/result.json').read_text())
posture = json.loads(Path(scene['robot']['initial_posture']).read_text())
k = yaml.safe_load(Path(scene['robot']['curobo_config']).read_text())['kinematics']
world = yaml.safe_load((ROOT/'static_world.yml').read_text())['cuboid']
full = pin.buildModelFromUrdf(k['urdf_path'])
values = dict(zip(posture['joint_names'],posture['joint_positions_rad']))
values.update(k['lock_joints'])
for side in ('l','r'): values[f'{side}_Joint_finger2'] = values[f'{side}_Joint_finger1']
qfull = pin.neutral(full)
for name,value in values.items(): qfull[full.joints[full.getJointId(name)].idx_q] = value
active = ['platform_joint'] + [f'r_joint{i}' for i in range(1,8)]
model = pin.buildReducedModel(full,[i for i in range(1,full.njoints) if full.names[i] not in active],qfull)
q0 = np.array([values[name] for name in list(model.names)[1:]])
j = model.joints[model.getJointId('platform_joint')]
model.lowerPositionLimit[j.idx_q] = 0.; model.upperPositionLimit[j.idx_q] = .79
conf = pink.Configuration(model,model.createData(),q0)
frame = 'r_grasp_tcp'
# Sphere model is read-only. Include all links, including locked left arm and chassis.
local_spheres = []; radii=[]; links=[]
for link,spheres in k['collision_spheres'].items():
    T = conf.get_transform_frame_to_world(link)
    for s in spheres:
        if s['radius'] <= 0: continue
        local_spheres.append(T.act(np.array(s['center']))); radii.append(s['radius']); links.append(link)
local_spheres=np.array(local_spheres); radii=np.array(radii)
ignore=k['self_collision_ignore']
pairs=np.array([(i,j) for i in range(len(links)) for j in range(i+1,len(links)) if links[i]!=links[j] and links[j] not in ignore.get(links[i],[]) and links[i] not in ignore.get(links[j],[])])
self_gaps=np.linalg.norm(local_spheres[pairs[:,0]]-local_spheres[pairs[:,1]],axis=1)-radii[pairs[:,0]]-radii[pairs[:,1]]
assert self_gaps.min() >= -1e-6, ('initial posture self collision',self_gaps.min())
base_z=previous['base_world_transform'][2][3]
bottle=np.array(previous['cup']['center_m']); bottle_radius=previous['cup']['radius_m']; bottle_halfheight=previous['cup']['height_m']/2
boxes=[]
for name,b in world.items():
    if name=='Floor': continue # ground support handled by fixed scene base height, not sphere penetration
    assert np.allclose(b['pose'][3:],[1,0,0,0]), 'Only axis aligned boxes in current scene'
    boxes.append((np.array(b['pose'][:3]),np.array(b['dims'])/2))

def environment_gap(centers):
    gaps=[]
    for center,half in boxes:
        d=np.abs(centers-center)-half
        sdf=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(np.max(d,axis=1),0)
        gaps.append(np.min(sdf-radii))
    delta=centers-bottle
    d=np.stack([np.linalg.norm(delta[:,:2],axis=1)-bottle_radius,np.abs(delta[:,2])-bottle_halfheight],axis=1)
    sdf=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(np.max(d,axis=1),0)
    gaps.append(np.min(sdf-radii))
    return float(min(gaps))

rng=np.random.default_rng(20261025)
starts=[]; attempts=0; sample_start=time.perf_counter()
while len(starts)<20 and attempts<100000:
    attempts+=1
    x,y=rng.uniform(-3.5,3.5,2); yaw=rng.uniform(-math.pi,math.pi)
    T=pin.SE3(pin.rpy.rpyToMatrix(0.,0.,yaw),np.array([x,y,base_z]))
    c=local_spheres@T.rotation.T+T.translation
    # Keep whole sphere envelope inside room, even when components extend above walls.
    inside=bool(np.all(np.abs(c[:,:2])+radii[:,None] <= 3.5))
    gap=environment_gap(c)
    if inside and gap>=.01:
        starts.append((T,gap))
assert len(starts)==20
sampling_time=time.perf_counter()-sample_start
R=np.array(previous['selected']['target_tcp_world'])[:3,:3]
target_world=pin.SE3(R,np.array(previous['selected']['target_tcp_world'])[:3,3])
# One untimed warm-up initializes the QP path; benchmark all starts afterwards.
task=FrameTask(frame,position_cost=1.,orientation_cost=1.,lm_damping=1e-3)
task.set_target(conf.get_transform_frame_to_world(frame))
pink.solve_ik(conf,[task],.02,solver='quadprog')
results=[]
for index,(T,gap) in enumerate(starts):
    config=pink.Configuration(model,model.createData(),q0.copy())
    target=T.inverse()*target_world
    task=FrameTask(frame,position_cost=1.,orientation_cost=1.,lm_damping=1e-3); task.set_target(target)
    rest=PostureTask(cost=1e-6); rest.set_target(q0)
    # Same iteration cap and tolerances as the previous local test, no reachability prefilter.
    started=time.perf_counter(); error=None; success=False
    for step in range(1200):
        try:
            v=pink.solve_ik(config,[task,rest],.02,solver='quadprog')
            config.integrate_inplace(v,.02)
        except Exception as exc:
            error=repr(exc); break
        actual=config.get_transform_frame_to_world(frame)
        pe=float(np.linalg.norm(actual.translation-target.translation))
        re=float(np.linalg.norm(pin.log3(target.rotation.T@actual.rotation)))
        if pe<1e-4 and re<1e-3:
            success=True; break
    elapsed=(time.perf_counter()-started)*1000
    actual=config.get_transform_frame_to_world(frame)
    pe=float(np.linalg.norm(actual.translation-target.translation)); re=float(np.linalg.norm(pin.log3(target.rotation.T@actual.rotation)))
    limit_ok=bool(np.all(config.q>=model.lowerPositionLimit-1e-7) and np.all(config.q<=model.upperPositionLimit+1e-7))
    r={'index':index+1,'base_x_m':float(T.translation[0]),'base_y_m':float(T.translation[1]),'base_yaw_deg':float(math.degrees(math.atan2(T.rotation[1,0],T.rotation[0,0]))),'initial_environment_gap_m':gap,'initial_self_gap_m':float(self_gaps.min()),'success':success and limit_ok,'time_ms':elapsed,'iterations':step+1,'position_error_m':pe,'rotation_error_rad':re,'joint_limits_ok':limit_ok,'exception':error,'joint_positions':config.q.tolist()}
    results.append(r); print(json.dumps(r),flush=True)

def statistics(rows):
    if not rows: return {'count':0}
    t=np.array([r['time_ms'] for r in rows])
    return {'count':len(rows),'mean_ms':float(t.mean()),'median_ms':float(np.median(t)),'p95_ms':float(np.percentile(t,95)),'min_ms':float(t.min()),'max_ms':float(t.max())}
report={'seed':20261025,'mode':'right_arm_plus_lift_8dof_base_fixed_at_each_random_pose','bottle_center_world_m':bottle.tolist(),'target_tcp_world':target_world.homogeneous.tolist(),'initial_posture':scene['robot']['initial_posture'],'robot_config':scene['robot']['curobo_config'],'sphere_count':len(radii),'sampling_attempts':attempts,'sampling_time_ms':sampling_time*1000,'sampling_domain':'uniform x,y in 7m x 7m room and yaw in [-pi,pi], rejected if whole sphere envelope leaves room or comes within 1cm of wall/table/bottle','ground_policy':'fixed original scene spawn height; support contact excluded from sphere obstacle tests','initial_collision_check':'sphere self collision using config ignore pairs; sphere-box and sphere-cylinder world checks','final_collision_checked':False,'execution_ready':False,'timer_scope':'QP iterations, integration and convergence checks only; excludes loading, sampling, collision filtering, target setup, visualization','max_iterations':1200,'position_tolerance_m':1e-4,'rotation_tolerance_rad':1e-3,'all':statistics(results),'success':statistics([r for r in results if r['success']]),'failure':statistics([r for r in results if not r['success']]),'joint_names':list(model.names)[1:],'results':results}
(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'results.csv').open('w') as f:
    fields=[k for k in results[0] if k not in ('joint_positions','exception')]
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(results)
print('SUMMARY',json.dumps({k:report[k] for k in ('all','success','failure','sampling_attempts','sampling_time_ms')}))
