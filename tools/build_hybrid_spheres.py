"""Create an editable hybrid draft; never change the active robot configuration."""
from pathlib import Path
import copy, hashlib, json
import numpy as np
import trimesh
import yaml
from build_sphere_comparison import load_mesh, regular_spheres, metrics, clip_polygon
ROOT=Path('/home/fang/robot_ws')
OUT=ROOT/'collision_models/hybrid'
SOURCE=ROOT/'realman_rm75/config/rm75_dual.yml'

def section(mesh, low, high):
    triangles=[]
    for tri in mesh.triangles:
        if tri[:,2].max()<low or tri[:,2].min()>high: continue
        p=clip_polygon(clip_polygon(tri,2,low,True),2,high,False)
        for j in range(1,len(p)-1): triangles.append([p[0],p[j],p[j+1]])
    t=np.asarray(triangles)
    return trimesh.Trimesh(t.reshape(-1,3),np.arange(t.size//3).reshape(-1,3),process=False)

# Hand-authored center paths in each link's local coordinates (metres).
# Radius initialization fits nearest-center surface samples; these remain editable drafts.
CENTERS={
 'base_link1': [[0,0,-.025],[0,0,.015],[0,0,.055]],
 'link1': [[0,0,-.045],[0,0,0],[0,.035,0],[0,.075,0]],
 'link2': [[0,.006,z] for z in [0,.030,.060,.090,.120]],
 'link3': [[0,0,-.045],[0,0,0],[0,.035,0],[0,.080,0]],
 'link4': [[0,.006,z] for z in [0,.030,.060,.090]],
 'link5': [[0,0,-.030],[0,0,0],[0,.028,0],[0,.060,0]],
 'link6': [[0,.005,z] for z in [0,.030,.060,.090]],
 'link7': [[-.050,0,.025],[-.015,-.030,.025],[-.015,.030,.025],[.015,0,.025]],
}

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 original=SOURCE.read_bytes();config=yaml.safe_load(original);k=config['kinematics']
 report={'status':'editable_draft_not_planning_validated','source_sha256':hashlib.sha256(original).hexdigest(), 'links':{}}
 for name, old in list(k['collision_spheres'].items()):
  mesh=load_mesh(Path(k['urdf_path']),name)
  np.random.seed(73);samples,_=trimesh.sample.sample_surface(mesh,12000)
  method='unchanged_baseline';spheres=copy.deepcopy(old)
  if name in ('base_link_underpan','platform_base_link'):
   grid=(4,6,3) if name=='base_link_underpan' else (4,4,3)
   spheres=regular_spheres(mesh,grid);method='occupied_regular_cells'
  elif name=='body_base_link':
   spheres=[]
   for lo,hi,grid in [(-.008,.15,(4,6,2)),(.15,.32,(3,3,2)),(.32,.57,(3,3,3)),(.57,1.30,(2,1,10)),(1.30,1.481,(3,2,2))]:
    part=section(mesh,lo,hi)
    if len(part.faces): spheres.extend(regular_spheres(part,grid))
   method='height_partitioned_regular_cells'
  elif name[:2] in ('l_','r_') and name[2:] in CENTERS:
   centers=np.asarray(CENTERS[name[2:]],float)
   if name=='r_link7': centers[:,2]+=.01
   if name=='r_base_link1': centers[:,1]*=-1
   points=np.concatenate([samples,mesh.vertices]);dist=np.linalg.norm(points[:,None]-centers[None],axis=2)
   assignment=dist.argmin(1)
   radii=[np.ceil((dist[assignment==i,i].max()+.001)/.0025)*.0025 if np.any(assignment==i) else .01 for i in range(len(centers))]
   spheres=[{'center':c.tolist(),'radius':float(r)} for c,r in zip(centers,radii)]
   method='hand_authored_centers_with_sample_fitted_initial_radii'
  k['collision_spheres'][name]=spheres
  report['links'][name]={'method':method,'baseline':metrics(mesh,old,samples),'candidate':metrics(mesh,spheres,samples)}
  print(name, len(old),'->',len(spheres), 'coverage',round(report['links'][name]['candidate']['sampled_surface_coverage_pct'],2),flush=True)
 (OUT/'rm75_hybrid_draft.yml').write_text(yaml.safe_dump(config,sort_keys=False))
 (OUT/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 (OUT/'arm_center_templates.json').write_text(json.dumps(CENTERS,indent=2))
 assert SOURCE.read_bytes()==original
 print('TOTAL',sum(len(s) for s in k['collision_spheres'].values()),flush=True)
if __name__=='__main__':main()
