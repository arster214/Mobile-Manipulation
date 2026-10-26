import json,yaml,numpy as np,pinocchio as pin
from pathlib import Path
root=Path(__file__).resolve().parents[1]
scene=json.loads((root/'scene_config.json').read_text());k=yaml.safe_load(Path(scene['robot']['curobo_config']).read_text())['kinematics'];initial=json.loads(Path(scene['robot']['initial_posture']).read_text())
m=pin.buildModelFromUrdf(k['urdf_path']);d=m.createData();q=pin.neutral(m)
v=dict(k['lock_joints']);v.update(zip(initial['joint_names'],initial['joint_positions_rad']))
for side in ('l','r'):v[f'{side}_Joint_finger2']=v[f'{side}_Joint_finger1']
for n,x in v.items():q[m.joints[m.getJointId(n)].idx_q]=x
pin.framesForwardKinematics(m,d,q)
a=[(n,d.oMf[m.getFrameId(n)].act(np.array(s['center'])),s['radius']) for n,ss in k['collision_spheres'].items() for s in ss if s['radius']>0]
ignore=k['self_collision_ignore'];bad=[]
for i,(n,c,r) in enumerate(a):
 for nn,cc,rr in a[i+1:]:
  if n==nn or nn in ignore.get(n,[]) or n in ignore.get(nn,[]):continue
  gap=float(np.linalg.norm(c-cc)-r-rr)
  if gap < -1e-6:bad.append((gap,n,nn))
print(sorted(bad)[:20])
