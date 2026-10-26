"""Rebuild the straight lift column with 3 X rows by 2 Y columns per layer."""
from pathlib import Path
import copy,json,hashlib
import numpy as np
import trimesh,yaml
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tests/artifacts/column_grid';OUT.mkdir(parents=True,exist_ok=True)
SOURCE=Path('/home/fang/robot_ws/postures/rm75_mobile_manipulation.yml')

def gaps(points,balls):
 c=np.array([b['center'] for b in balls]);r=np.array([b['radius'] for b in balls])
 return np.concatenate([(np.linalg.norm(p[:,None]-c[None],axis=2)-r).min(1) for p in np.array_split(points,100)])

def main():
 cfg=yaml.safe_load(SOURCE.read_text());k=cfg['kinematics'];old=k['collision_spheres']['body_base_link']
 mesh=trimesh.load(Path(k['urdf_path']).parent/'../meshes/body_base_link.STL',force='mesh')
 low,high=.57,1.30
 part=mesh.slice_plane([0,0,low],[0,0,1]).slice_plane([0,0,high],[0,0,-1])
 lo,hi=part.bounds.copy();lo[2]=low;hi[2]=high
 np.random.seed(150);fit=np.vstack([trimesh.sample.sample_surface(part,150000)[0],part.vertices])
 np.random.seed(851);validation=trimesh.sample.sample_surface(mesh,250000)[0]
 column=(validation[:,2]>=low)&(validation[:,2]<=high)
 kept=[copy.deepcopy(b) for b in old if not low<=b['center'][2]<=high]
 poses=json.loads((ROOT/'tests/artifacts/lift_spheres/transforms.json').read_text())
 obstacles=[];radii=[]
 for transforms in poses.values():
  bt=np.array(transforms['body_base_link'])
  for name in ('l_base_link1','r_base_link1'):
   t=np.linalg.inv(bt)@np.array(transforms[name]);ss=k['collision_spheres'][name]
   obstacles.extend(np.array([b['center'] for b in ss])@t[:3,:3].T+t[:3,3]);radii.extend([b['radius'] for b in ss])
 obstacles=np.array(obstacles);radii=np.array(radii);trials=[]
 for layers in (16,):
  dims=np.array([3,2,layers]);step=(hi-lo)/dims;assignment=np.minimum(((fit-lo)/step).astype(int),dims-1).clip(0)
  balls=[]
  for z in range(layers):
   for y in range(2):
    for x in range(3):
     index=np.array([x,y,z]);center=lo+(index+.5)*step;p=fit[np.all(assignment==index,axis=1)]
     radius=float(np.linalg.norm(p-center,axis=1).max()+.0005) if len(p) else float(np.linalg.norm(step)/2)
     balls.append({'center':center.tolist(),'radius':radius})
  cc=np.array([b['center'] for b in balls]);rr=np.array([b['radius'] for b in balls])
  minimum=float((np.linalg.norm(cc[:,None]-obstacles[None],axis=2)-rr[:,None]-radii).min())
  trial={'layers':layers,'grid_spheres':len(balls),'min_column_shoulder_gap_mm':minimum*1000,'radius_range_mm':[float(rr.min()*1000),float(rr.max()*1000)]}
  trials.append(trial);print(trial,flush=True)
  candidate=copy.deepcopy(cfg);candidate['kinematics']['collision_spheres']['body_base_link']=kept+balls
  (OUT/f'grid_{layers}.yml').write_text(yaml.safe_dump(candidate,sort_keys=False))
  if minimum>=0:break
 new=kept+balls;before=gaps(validation,old);after=gaps(validation,new)
 result={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'layout':'3 rows X by 2 columns Y per Z layer','column_z_range_m':[low,high],'bounds_m':[lo.tolist(),hi.tolist()],'trials':trials,'selected_layers':layers,'old_body_count':len(old),'new_body_count':len(new),'total_spheres':sum(len(b) for b in k['collision_spheres'].values())-len(old)+len(new),'whole_body_link_surface_coverage_before_pct':float((before<=0).mean()*100),'whole_body_link_surface_coverage_after_pct':float((after<=0).mean()*100),'column_surface_coverage_before_pct':float((before[column]<=0).mean()*100),'column_surface_coverage_after_pct':float((after[column]<=0).mean()*100),'newly_uncovered_max_mm':float(max(0,after[before<=0].max())*1000),'independent_surface_samples':len(validation)}
 (OUT/'candidate.yml').write_text(yaml.safe_dump(candidate,sort_keys=False));(OUT/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
