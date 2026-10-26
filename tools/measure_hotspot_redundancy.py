from pathlib import Path
import numpy as np,trimesh,yaml,json
from build_sphere_comparison import load_mesh
R=Path('/home/fang/robot_ws');k=yaml.safe_load((R/'postures/rm75_mobile_manipulation.yml').read_text())['kinematics'];out={}
for n,ids in {'body_base_link':[85,86,95,96],'platform_base_link':[1,2,3,4,44,26,27,28,29,63],'l_base_link1':[0,2],'r_base_link1':[0,2]}.items():
 m=load_mesh(Path(k['urdf_path']),n);np.random.seed(20261028);p=trimesh.sample.sample_surface(m,120000)[0];b=k['collision_spheres'][n];c=np.array([x['center'] for x in b]);r=np.array([x['radius'] for x in b]);counts=np.zeros(len(b));unique=np.zeros(len(b))
 for x in np.array_split(p,40):
  inside=np.linalg.norm(x[:,None]-c,axis=2)<=r;counts+=inside.sum(0);unique+=(inside&(inside.sum(1)==1)[:,None]).sum(0)
 out[n]=[{'index':i,'radius_mm':r[i]*1000,'link_surface_fraction_pct':counts[i]/len(p)*100,'unique_link_surface_fraction_pct':unique[i]/len(p)*100,'covered_surface_redundancy_pct':float((1-unique[i]/counts[i])*100) if counts[i] else 100} for i in ids]
 print(n,out[n],flush=True)
(R/'collision_models/hybrid/failed20_audit/hotspot_redundancy.json').write_text(json.dumps(out,indent=2))
