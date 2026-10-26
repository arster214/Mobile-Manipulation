"""Fit shoulder mount collision meshes, then audit against the column sweep."""
from pathlib import Path
from functools import partial
import json,xml.etree.ElementTree as ET
import numpy as np,yaml,trimesh
from scipy.optimize import minimize
from scipy.cluster.vq import kmeans2
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tests/artifacts/lift_spheres'
cfg=yaml.safe_load(Path('/home/fang/robot_ws/postures/rm75_mobile_manipulation.yml').read_text());k=cfg['kinematics'];urdf=Path(k['urdf_path']);tree=ET.parse(urdf)
poses=json.loads((OUT/'transforms.json').read_text());metrics={}
def mesh_for(name):
 parts=[]
 for col in tree.find(f"link[@name='{name}']").findall('collision'):
  m=col.find('geometry/mesh');mesh=trimesh.load(urdf.parent/m.get('filename'),force='mesh');mesh.apply_scale(np.fromstring(m.get('scale','1 1 1'),sep=' '));o=col.find('origin')
  if o is not None:
   t=trimesh.transformations.euler_matrix(*np.fromstring(o.get('rpy','0 0 0'),sep=' '));t[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ');mesh.apply_transform(t)
  parts.append(mesh)
 return trimesh.util.concatenate(parts)
for name in ('l_base_link1','r_base_link1'):
 mesh=mesh_for(name);np.random.seed(441);pts=np.vstack([mesh.vertices,trimesh.sample.sample_surface(mesh,50000)[0]])
 old=k['collision_spheres'][name];oc=np.array([b['center'] for b in old]);orr=np.array([b['radius'] for b in old]);oldgap=(np.linalg.norm(pts[:,None]-oc,axis=2)-orr).min(1)
 best=None
 for count in (3,4,5,6,8):
  centers,labels=kmeans2(pts,count,minit='++',seed=42);balls=[]
  for j in range(count):
   p=pts[labels==j];c=centers[j];radius=np.linalg.norm(p-c,axis=1).max()
   # A convex cluster's hull suffices for an enclosing sphere.
   from scipy.spatial import ConvexHull
   hull=p[ConvexHull(p).vertices]
   res=minimize(lambda x:x[3],np.r_[c,radius],method='SLSQP',constraints=[{'type':'ineq','fun':lambda x:x[3]**2-np.sum((hull-x[:3])**2,axis=1)}],bounds=[(None,None)]*3+[(0,None)],options={'ftol':1e-12,'maxiter':200})
   c=res.x[:3];radius=float(np.linalg.norm(p-c,axis=1).max()+.0001);balls.append({'center':c.tolist(),'radius':radius})
  minimum=1
  for t in poses.values():
   relative=np.linalg.inv(np.array(t[name]))@np.array(t['body_base_link']);bs=k['collision_spheres']['body_base_link'];bc=np.array([b['center'] for b in bs])@relative[:3,:3].T+relative[:3,3];br=np.array([b['radius'] for b in bs])
   gap=(np.linalg.norm(np.array([b['center'] for b in balls])[:,None]-bc[None],axis=2)-np.array([b['radius'] for b in balls])[:,None]-br).min();minimum=min(minimum,gap)
  print(name,count,minimum,flush=True)
  if best is None or minimum>best[0]:best=(float(minimum),balls)
  if minimum>=0:break
 k['collision_spheres'][name]=best[1];metrics[name]={'old_coverage_pct':float((oldgap<=0).mean()*100),'new_coverage_pct':100.,'count':len(best[1]),'min_column_gap_mm':best[0]*1000}
(OUT/'shoulder_fit.yml').write_text(yaml.safe_dump(cfg,sort_keys=False));(OUT/'shoulder_metrics.json').write_text(json.dumps(metrics,indent=2));print(metrics,flush=True)
