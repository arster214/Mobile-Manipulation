"""Replace rear platform overlarge spheres, retaining sampled mesh coverage."""
from pathlib import Path
import json,copy
import numpy as np,trimesh,yaml
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tests/artifacts/platform_rear';OUT.mkdir(parents=True,exist_ok=True)
source=Path('/home/fang/robot_ws/postures/rm75_mobile_manipulation.yml');cfg=yaml.safe_load(source.read_text());k=cfg['kinematics'];name='platform_base_link';old=k['collision_spheres'][name]
mesh=trimesh.load(Path(k['urdf_path']).parent/'../meshes/platform_base_link.STL',force='mesh')
np.random.seed(674);points=np.vstack([mesh.vertices,trimesh.sample.sample_surface(mesh,180000)[0]])
def gap(p,balls):
 c=np.array([b['center'] for b in balls]);r=np.array([b['radius'] for b in balls])
 return np.concatenate([(np.linalg.norm(a[:,None]-c,axis=2)-r).min(1) for a in np.array_split(p,60)])
removed=[i for i,b in enumerate(old) if b['center'][1]>=-.001 and b['radius']>.045]
balls=[copy.deepcopy(b) for i,b in enumerate(old) if i not in removed]
before=gap(points,old);after=gap(points,balls)
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
patch=points[(before<=0)&(after>-.0005)]
pending=[patch];replacement=[]
while pending:
 p=pending.pop();c=(p.min(0)+p.max(0))/2;r=float(np.linalg.norm(p-c,axis=1).max()+.001)
 if r>.045:
  axis=np.ptp(p,axis=0).argmax();cut=(p[:,axis].min()+p[:,axis].max())/2
  pending.extend([p[p[:,axis]<=cut],p[p[:,axis]>cut]])
 else:replacement.append({'center':c.tolist(),'radius':r})
balls+=replacement
np.random.seed(675);validation=trimesh.sample.sample_surface(mesh,250000)[0];b=gap(validation,old);a=gap(validation,balls)
k['collision_spheres'][name]=balls
(OUT/'candidate.yml').write_text(yaml.safe_dump(cfg,sort_keys=False))
metrics={'removed_indices':removed,'old_platform_count':len(old),'new_platform_count':len(balls),'replacement_radius_range_mm':[min(b['radius'] for b in replacement)*1000,max(b['radius'] for b in replacement)*1000],'coverage_before_pct':float((b<=0).mean()*100),'coverage_after_pct':float((a<=0).mean()*100),'newly_uncovered_max_mm':float(max(0,a[b<=0].max())*1000),'rear_extent_before_m':max(x['center'][1]+x['radius'] for x in old),'rear_extent_after_m':max(x['center'][1]+x['radius'] for x in balls)}
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2));print(metrics)
