from pathlib import Path
from functools import partial
import json,numpy as np,yaml,yourdfpy
from scipy.optimize import least_squares
R=Path('/home/fang/robot_ws');config=yaml.safe_load((R/'collision_models/hybrid/rm75_platform_rebuilt.yml').read_text());k=config['kinematics'];p=Path(k['urdf_path']);robot=yourdfpy.URDF.load(p,build_scene_graph=True,load_meshes=False,filename_handler=partial(yourdfpy.filename_handler_magic,dir=p.parent));base=dict(zip(k['cspace']['joint_names'],k['cspace']['default_joint_position']));base.update(k['lock_joints']);robot.update_cfg(base);rng=np.random.default_rng(551)
solutions={}
for side,sign in [('l',1),('r',-1)]:
 names=[f'{side}_joint{i}' for i in range(1,8)];joints={j.name:j for j in robot.robot.joints};lo=np.array([joints[n].limit.lower for n in names]);hi=np.array([joints[n].limit.upper for n in names]);shoulder=robot.get_transform(f'{side}_link1')[:3,3].copy();sol=[]
 for spread in [0,.025,.05]:
  elbow=shoulder+np.array([sign*spread,-.015,-np.sqrt(.256**2-spread**2-.015**2)])
  wrist=elbow+np.array([0,-.210,0])
  def residual(q):
   state=dict(base);state.update(dict(zip(names,q)));robot.update_cfg(state)
   e=robot.get_transform(f'{side}_link3')[:3,3];w=robot.get_transform(f'{side}_link5')[:3,3];tool=robot.get_transform(f'{side}_tool0')
   return np.r_[(e-elbow)*10,(w-wrist)*10,(tool[:3,2]-[0,0,1])*.7,q*.002]
  for attempt in range(10):
   initial=np.zeros(7) if attempt==0 else rng.uniform(lo,hi)*.7
   fit=least_squares(residual,initial,bounds=(lo+1e-5,hi-1e-5),max_nfev=180,ftol=1e-8,xtol=1e-8,gtol=1e-8)
   err=float(np.linalg.norm(residual(fit.x)[:6]));tool=robot.get_transform(f'{side}_tool0');e=robot.get_transform(f'{side}_link3')[:3,3];w=robot.get_transform(f'{side}_link5')[:3,3]
   sol.append({'q':fit.x.tolist(),'spread':spread,'error':err,'elbow':e.tolist(),'wrist':w.tolist(),'tool_position':tool[:3,3].tolist(),'tool_z':tool[:3,2].tolist(),'shoulder':shoulder.tolist()})
 solutions[side]=sorted(sol,key=lambda s:s['error'])[:20];print(side,solutions[side][0],flush=True)
(R/'postures/rest_candidates.json').write_text(json.dumps({'config':str(R/'collision_models/hybrid/rm75_platform_rebuilt.yml'),'active_joints':k['cspace']['joint_names'],'lock_joints':k['lock_joints'],'solutions':solutions},indent=2))
