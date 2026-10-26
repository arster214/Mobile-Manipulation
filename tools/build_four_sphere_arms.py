"""Four centers per arm link: endpoints and two equally spaced path positions."""
from pathlib import Path
import copy,json,hashlib
import numpy as np
import trimesh,yaml
from build_sphere_comparison import load_mesh,metrics
ROOT=Path('/home/fang/robot_ws/collision_models/hybrid')
source=max(ROOT.glob('rm75_hybrid_edited_*.yml'),key=lambda p:p.stat().st_mtime)
config=yaml.safe_load(source.read_text()); before=copy.deepcopy(config)
paths={
 'link1':[[0,0,-.045],[0,0,0],[0,.075,0]],
 'link2':[[0,.006,0],[0,.006,.120]],
 'link3':[[0,0,-.045],[0,0,0],[0,.080,0]],
 'link4':[[0,.006,0],[0,.006,.100]],
 'link5':[[0,0,-.030],[0,0,0],[0,.060,0]],
 'link6':[[0,.005,0],[0,.005,.090]],
 'link7':[[-.050,0,.025],[.015,0,.025]],
}
k=config['kinematics'];report={'source':str(source),'layout':'four centers at path arclength fractions 0, 1/3, 2/3, 1; end centers lie inside end regions, not at raw mesh extrema','links':{}}
for side in ['l','r']:
 for suffix,path in paths.items():
  name=f'{side}_{suffix}';p=np.array(path,float)
  if name=='r_link7':p[:,2]+=.01
  lengths=np.linalg.norm(np.diff(p,axis=0),axis=1);cum=np.r_[0,np.cumsum(lengths)]
  centers=np.array([[np.interp(t,cum,p[:,a]) for a in range(3)] for t in np.linspace(0,cum[-1],4)])
  mesh=load_mesh(Path(k['urdf_path']),name);np.random.seed(173);sample,_=trimesh.sample.sample_surface(mesh,30000)
  pts=np.concatenate([mesh.vertices,sample]);d=np.linalg.norm(pts[:,None]-centers[None],axis=2);assignment=d.argmin(1)
  radii=np.array([np.ceil((d[assignment==i,i].max()+.001)/.001)*.001 for i in range(4)])
  balls=[{'center':c.tolist(),'radius':float(r)} for c,r in zip(centers,radii)]
  k['collision_spheres'][name]=balls
  np.random.seed(271);check,_=trimesh.sample.sample_surface(mesh,30000)
  stats=metrics(mesh,balls,check)
  report['links'][name]={'before':metrics(mesh,before['kinematics']['collision_spheres'][name],check),'after':stats,'centers_m':centers.tolist(),'radii_m':radii.tolist()}
  print(name,stats['count'],'coverage',stats['sampled_surface_coverage_pct'],'radii mm',radii*1000,flush=True)
for n,b in k['collision_spheres'].items():
 if n not in report['links']:assert b==before['kinematics']['collision_spheres'][n]
assert all(len(k['collision_spheres'][n])==4 for n in report['links'])
report['total_spheres']=sum(map(len,k['collision_spheres'].values()))
(ROOT/'rm75_hybrid_four_per_arm_link.yml').write_text(yaml.safe_dump(config,sort_keys=False))
(ROOT/'four_per_link_evaluation.json').write_text(json.dumps(report,indent=2))
print('TOTAL',report['total_spheres'])
