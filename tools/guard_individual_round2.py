from pathlib import Path
import json,yaml,numpy as np,trimesh
from build_sphere_comparison import load_mesh
R=Path('/home/fang/robot_ws/collision_models/hybrid');d=json.loads((R/'individual_refined_metrics.json').read_text());old=yaml.safe_load(Path(d['source']).read_text());new=yaml.safe_load((R/'rm75_individual_refined.yml').read_text())
for n in ['body_base_link','platform_base_link']:
 v=d['links'][n];a=old['kinematics']['collision_spheres'][n];changed={x['index']:x for x in v['changes']};it=iter(new['kinematics']['collision_spheres'][n]);b=[None if i in changed and changed[i]['action']=='remove' else next(it) for i in range(len(a))]
 mesh=load_mesh(Path(new['kinematics']['urdf_path']),n);np.random.seed(8811);p=np.vstack([trimesh.sample.sample_surface(mesh,300000)[0],mesh.vertices]);ac=np.array([x['center'] for x in a]);ar=np.array([x['radius'] for x in a]);da=np.concatenate([np.linalg.norm(p[j:j+5000,None]-ac,axis=2) for j in range(0,len(p),5000)]);ga=(da-ar).min(1)
 def bg():
  c=np.array([s['center'] for s in b if s]);r=np.array([s['radius'] for s in b if s]);return np.concatenate([(np.linalg.norm(p[j:j+5000,None]-c,axis=2)-r).min(1) for j in range(0,len(p),5000)])
 reverted=[]
 for _ in range(len(changed)):
  g=bg();bad=(ga<=0)&(g>.0004)
  for i in changed:
   patch=da[:,i]<=ar[i]+.003
   if np.mean(ga[patch]<=0)-np.mean(g[patch]<=0)>.0015:bad|=patch&(ga<=0)&(g>0)
  if not bad.any():break
  options=[i for i in changed if i not in reverted];j=max(options,key=lambda i:int(np.sum(bad&(da[:,i]<=ar[i]))));b[j]=a[j];reverted.append(j)
 new['kinematics']['collision_spheres'][n]=[s for s in b if s]
 v['reverted_after_holdout']=v.get('reverted_after_holdout',[])+reverted;v['count_after']=len([s for s in b if s]);v['changes']=[x for x in v['changes'] if x['index'] not in reverted]
 v['mean_radius_after_mm']=float(np.mean([s['radius'] for s in b if s])*1000)
 print(n,'restored indices',reverted,'remaining changes',len(v['changes']),flush=True)
(R/'rm75_individual_refined.yml').write_text(yaml.safe_dump(new,sort_keys=False));(R/'individual_refined_metrics.json').write_text(json.dumps(d,indent=2))
