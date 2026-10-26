from pathlib import Path
import json,copy,numpy as np,trimesh,yaml
from build_sphere_comparison import load_mesh
R=Path('/home/fang/robot_ws/collision_models/hybrid');cfg=yaml.safe_load((R/'rm75_patch_rebuilt_40.yml').read_text());report={}
for name in ['platform_base_link','body_base_link']:
 balls=cfg['kinematics']['collision_spheres'][name];mesh=load_mesh(Path(cfg['kinematics']['urdf_path']),name);np.random.seed(12811);pts=np.vstack([trimesh.sample.sample_surface(mesh,400000)[0],mesh.vertices]);c=np.array([s['center'] for s in balls]);r=np.array([s['radius'] for s in balls]);hit=[];strong=[]
 for j in range(0,len(pts),3000):
  d=np.linalg.norm(pts[j:j+3000,None]-c,axis=2);hit.append(d<=r);strong.append(d<=r-.0005)
 hit=np.concatenate(hit);strong=np.concatenate(strong);count=strong.sum(1);alive=np.ones(len(balls),bool)
 for i in np.argsort(r):
  if not np.any(hit[:,i] & ((count-strong[:,i])==0)):
   count-=strong[:,i];alive[i]=False
 new=[s for i,s in enumerate(balls) if alive[i]];cfg['kinematics']['collision_spheres'][name]=new
 np.random.seed(13811);val=trimesh.sample.sample_surface(mesh,500000)[0]
 def gap(bs):
  cc=np.array([s['center'] for s in bs]);rr=np.array([s['radius'] for s in bs]);return np.concatenate([(np.linalg.norm(val[j:j+3000,None]-cc,axis=2)-rr).min(1) for j in range(0,len(val),3000)])
 a=gap(balls);b=gap(new);loss=max(0,float(b[a<=0].max())*1000)
 assert loss<.1,(name,loss)
 report[name]={'before':len(balls),'after':len(new),'removed_indices':np.where(~alive)[0].tolist(),'validation_coverage_pct':float(np.mean(b<=0)*100),'new_gap_mm':loss};print(name,report[name],flush=True)
(R/'rm75_patch_compact.yml').write_text(yaml.safe_dump(cfg,sort_keys=False));(R/'patch_compact_metrics.json').write_text(json.dumps(report,indent=2))
base=yaml.safe_load((R/'rm75_individual_refined.yml').read_text());base['kinematics']['collision_spheres']['platform_base_link']=cfg['kinematics']['collision_spheres']['platform_base_link'];(R/'rm75_platform_rebuilt.yml').write_text(yaml.safe_dump(base,sort_keys=False))
