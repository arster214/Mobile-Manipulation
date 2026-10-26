"""Keep the initial arm shape; align tool Z forward and tool X horizontally."""
from pathlib import Path
from functools import partial
import json, shutil, numpy as np, yaml, yourdfpy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
R=Path('/home/fang/robot_ws')
old=json.loads((R/'postures/mobile_manipulation_initial.json').read_text())
cfg=yaml.safe_load(Path(old['curobo_config']).read_text());k=cfg['kinematics'];p=Path(k['urdf_path'])
robot=yourdfpy.URDF.load(p,build_scene_graph=True,load_meshes=False,filename_handler=partial(yourdfpy.filename_handler_magic,dir=p.parent))
base=dict(zip(old['joint_names'],old['joint_positions_rad']));base.update(k['lock_joints']);robot.update_cfg(base)
solutions={}
for side,label in [('l','left'),('r','right')]:
 names=[f'{side}_joint{i}' for i in range(1,8)];q0=np.array([base[n] for n in names]);joints=robot.joint_map
 lo=np.array([joints[n].limit.lower for n in names]);hi=np.array([joints[n].limit.upper for n in names])
 robot.update_cfg(base)
 e0=robot.get_transform(f'{side}_link3')[:3,3].copy();w0=robot.get_transform(f'{side}_link5')[:3,3].copy();t0=robot.get_transform(f'{side}_tool0').copy()
 sol=[]
 for sign in [-1]:
  z=np.array([0.,-1,0]);x=np.array([0.,0,sign]);target=np.column_stack([x,np.cross(z,x),z])
  def residual(q):
   state=dict(base);state.update(dict(zip(names,q)));robot.update_cfg(state)
   e=robot.get_transform(f'{side}_link3')[:3,3];w=robot.get_transform(f'{side}_link5')[:3,3];t=robot.get_transform(f'{side}_tool0')
   return np.r_[10*(e-e0),10*(w-w0),Rotation.from_matrix(target.T@t[:3,:3]).as_rotvec()*5,(q-q0)*1e-5]
  fit=least_squares(residual,q0,bounds=(lo+1e-6,hi-1e-6),max_nfev=300,gtol=1e-11,ftol=1e-11,xtol=1e-11)
  err=float(np.linalg.norm(residual(fit.x)[:6]));t=robot.get_transform(f'{side}_tool0');d=dict(old[label]);d.update(q=fit.x.tolist(),error=err,elbow=robot.get_transform(f'{side}_link3')[:3,3].tolist(),wrist=robot.get_transform(f'{side}_link5')[:3,3].tolist(),tool_position=t[:3,3].tolist(),tool_z=t[:3,2].tolist(),tool_x=t[:3,0].tolist(),tool_y=t[:3,1].tolist(),joint_change_norm=float(np.linalg.norm(fit.x-q0)))
  sol.append(d)
 solutions[side]=sol
(R/'postures/clockwise_z_candidates.json').write_text(json.dumps({'config':old['config'],'active_joints':old['joint_names'],'lock_joints':k['lock_joints'],'solutions':solutions},indent=2))
