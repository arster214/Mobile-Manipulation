"""Refine bounded centers using enclosing spheres; preserve three balls per link."""
from pathlib import Path
import copy,json,hashlib
import numpy as np,trimesh,yaml
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from build_sphere_comparison import load_mesh,metrics
ROOT=Path('/home/fang/robot_ws/collision_models/hybrid')
source=max(ROOT.glob('rm75_hybrid_edited_*.yml'),key=lambda p:p.stat().st_mtime)
config=yaml.safe_load(source.read_text()); k=config['kinematics'];report={'source':str(source),'method':'alternating minimum enclosing spheres; center shifts bounded to 20 mm per axis; fitted on mesh vertices and 30000 area samples; final radius margin 1 mm','links':{}}
for side in ['l','r']:
 for i in range(1,8):
  name=f'{side}_link{i}';old=copy.deepcopy(k['collision_spheres'][name]);assert len(old)==3
  mesh=load_mesh(Path(k['urdf_path']),name); np.random.seed(811);samples,_=trimesh.sample.sample_surface(mesh,30000)
  pts=np.unique(np.vstack([mesh.vertices,samples]),axis=0)*1000
  c=np.array([s['center'] for s in old])*1000;anchor=c.copy();r=np.array([s['radius'] for s in old])*1000
  lo,hi=mesh.bounds*1000
  excess=max(0,float(np.maximum(lo-(c-r[:,None]),c+r[:,None]-hi).max()))
  for step in range(12):
   labels=(np.linalg.norm(pts[:,None]-c[None],axis=2)-r).argmin(1)
   previous=np.r_[c.ravel(),r]
   for j in range(3):
    points=pts[labels==j]
    if len(points)<4:continue
    try:points=points[ConvexHull(points).vertices]
    except Exception:pass
    def constraint(x):return np.r_[x[3]**2-np.sum((points-x[:3])**2,axis=1), x[:3]-(x[3]+1.1)-(lo-excess), (hi+excess)-x[:3]-(x[3]+1.1)]
    x0=np.r_[c[j],np.linalg.norm(points-c[j],axis=1).max()]
    result=minimize(lambda x:x[3],x0,method='SLSQP',bounds=[(a-20,a+20) for a in anchor[j]]+[(1,150)],constraints={'type':'ineq','fun':constraint},options={'maxiter':120,'ftol':1e-7})
    if result.success:
     c[j]=result.x[:3];r[j]=np.linalg.norm(points-c[j],axis=1).max()
   if np.max(np.abs(np.r_[c.ravel(),r]-previous))<.02:break
  # Full fit-point check, then 1 mm independent numerical/geometric allowance.
  r=np.ceil((r+1)*10)/10
  balls=[{'center':(cc/1000).tolist(),'radius':float(rr/1000)} for cc,rr in zip(c,r)]
  np.random.seed(917);check,_=trimesh.sample.sample_surface(mesh,60000)
  stats=metrics(mesh,balls,check);before=metrics(mesh,old,check)
  assert stats['sampled_max_gap_mm']<1e-5 and stats['vertex_max_gap_mm']<1e-5
  if stats['aabb_excess_mm'] > before['aabb_excess_mm'] + .1:
   balls=old;stats=before;r=np.array([s['radius']*1000 for s in old]);c=anchor.copy()
  k['collision_spheres'][name]=balls
  report['links'][name]={'before':before,'after':stats,'old_radii_mm':[s['radius']*1000 for s in old],'new_radii_mm':r.tolist(),'center_shift_mm':np.linalg.norm(c-anchor,axis=1).tolist()}
  print(name, np.round([s['radius']*1000 for s in old],1),'->',r,'AABB',round(before['aabb_excess_mm'],1),'->',round(stats['aabb_excess_mm'],1),flush=True)
(ROOT/'rm75_hybrid_three_refined.yml').write_text(yaml.safe_dump(config,sort_keys=False))
(ROOT/'three_refined_metrics.json').write_text(json.dumps(report,indent=2))
