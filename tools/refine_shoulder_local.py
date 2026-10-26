"""Local shoulder sphere refit, preserving previously covered sampled surface points."""
from pathlib import Path
from functools import partial
import copy,json
import numpy as np,trimesh,yaml,yourdfpy
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from build_sphere_comparison import load_mesh
ROOT=Path('/home/fang/robot_ws/collision_models/hybrid')
source=max(ROOT.glob('rm75_hybrid_edited_*.yml'),key=lambda p:p.stat().st_mtime)
config=yaml.safe_load(source.read_text());k=config['kinematics'];baseline=copy.deepcopy(config)
urdfpath=Path(k['urdf_path']);robot=yourdfpy.URDF.load(urdfpath,build_scene_graph=True,load_meshes=False,filename_handler=partial(yourdfpy.filename_handler_magic,dir=urdfpath.parent))
q=dict(zip(k['cspace']['joint_names'],k['cspace']['default_joint_position']));q.update(k['lock_joints']);robot.update_cfg(q)
report={'source':str(source),'policy':'Only body/platform spheres intersecting the four baseline arm pairs at default pose are refit. Keep all previously covered training samples/vertices covered. Fit uniquely needed points, split if needed. No ignore changes.','links':{}}
def gaps(pts,spheres):
 if not spheres:return np.full(len(pts),np.inf)
 c=np.array([s['center'] for s in spheres]);r=np.array([s['radius'] for s in spheres]);return np.concatenate([(np.linalg.norm(pts[i:i+5000,None]-c,axis=2)-r).min(1) for i in range(0,len(pts),5000)])
def fit(points,obstacles):
 # mm units: enclosing ball and clearance from opposing spheres at default pose.
 pts=points*1000; obs=np.array([x['center'] for x in obstacles])*1000; rad=np.array([x['radius'] for x in obstacles])*1000
 if len(pts)>4:
  try:pts=pts[ConvexHull(pts).vertices]
  except Exception:pass
 center=pts.mean(0);r=max(.5,np.linalg.norm(pts-center,axis=1).max()+.3)
 def cons(x):return np.r_[x[3]**2-((pts-x[:3])**2).sum(1),np.linalg.norm(obs-x[:3],axis=1)-rad-x[3]-1.5]
 res=minimize(lambda x:x[3],np.r_[center,r],method='SLSQP',bounds=[(a-35,b+35) for a,b in zip(pts.min(0),pts.max(0))]+[(.3,200)],constraints={'type':'ineq','fun':cons},options={'maxiter':160,'ftol':1e-7})
 if res.success and cons(res.x).min()>-1e-4:
  return [{'center':(res.x[:3]/1000).tolist(),'radius':float((res.x[3]+1.)/1000)}]
 return None
for name,othernames in [('body_base_link',['l_base_link1','r_base_link1']),('platform_base_link',['l_link2','r_link2'])]:
 old=copy.deepcopy(k['collision_spheres'][name]);T=robot.get_transform(name);obstacles=[]
 for other in othernames:
  rel=np.linalg.inv(T)@robot.get_transform(other)
  for s in k['collision_spheres'][other]:obstacles.append({'center':(rel[:3,:3]@s['center']+rel[:3,3]).tolist(),'radius':s['radius']})
 def collides(s):return any(np.linalg.norm(np.array(s['center'])-o['center'])<s['radius']+o['radius']-1e-8 for o in obstacles)
 indices=[i for i,s in enumerate(old) if collides(s)];print(name,'targets',indices,flush=True)
 mesh=load_mesh(urdfpath,name);np.random.seed(369);train=np.vstack([trimesh.sample.sample_surface(mesh,600000)[0],mesh.vertices]);bg=gaps(train,old);protected=train[bg<=0]
 current=copy.deepcopy(old);changes=[]
 for idx in indices:
  ball=current[idx];others=[s for j,s in enumerate(current) if j!=idx and s is not None]
  # Only points not already protected by other balls need replacement coverage.
  needed=protected[gaps(protected,others)>0]
  if not len(needed):current[idx]=None;changes.append({'index':idx,'action':'remove_redundant','new_count':0});continue
  replacement=fit(needed,obstacles)
  if replacement is None:
   # Spatial subdivision of the needed patch, up to sixteen smaller balls.
   groups=[needed]
   for _ in range(7):
    trial=[fit(g,obstacles) for g in groups]
    if all(x is not None for x in trial):break
    j=max((j for j,x in enumerate(trial) if x is None),key=lambda j:np.ptp(groups[j],axis=0).max())
    g=groups.pop(j);axis=np.ptp(g,axis=0).argmax();cut=np.median(g[:,axis]);mask=g[:,axis]<=cut
    if not mask.any() or mask.all():groups.insert(j,g);break
    groups.extend([g[mask],g[~mask]])
   trial=[fit(g,obstacles) for g in groups]
   if all(x is not None for x in trial):replacement=sum(trial,[])
  if replacement is None:
   changes.append({'index':idx,'action':'retained_infeasible','needed_points':len(needed)});continue
  current[idx]=replacement[0];current.extend(replacement[1:]);changes.append({'index':idx,'action':'refit','old':ball,'new':replacement,'needed_points':len(needed)})
  print(' ',idx,changes[-1]['action'],'replacement',len(replacement),flush=True)
 current=[s for s in current if s is not None]
 np.random.seed(469);test=trimesh.sample.sample_surface(mesh,240000)[0];before=gaps(test,old);after=gaps(test,current)
 # Independently check the changed local patch, not just whole-link coverage.
 centers=np.array([old[i]['center'] for i in indices]);radii=np.array([old[i]['radius'] for i in indices])
 local=(np.linalg.norm(test[:,None]-centers,axis=2)<=radii+.003).any(1)
 record={'target_indices':indices,'changes':changes,'count_before':len(old),'count_after':len(current),'baseline_coverage_pct':float(100*np.mean(before<=0)),'new_coverage_pct':float(100*np.mean(after<=0)),'local_sample_count':int(local.sum()),'local_before_coverage_pct':float(100*np.mean(before[local]<=0)),'local_after_coverage_pct':float(100*np.mean(after[local]<=0)),'local_max_gap_mm':float(max(0,after[local].max())*1000),'newly_uncovered_max_gap_mm':float(max(0,after[before<=0].max())*1000),'remaining_default_sphere_overlaps':sum(collides(s) for s in current)}
 assert record['local_after_coverage_pct']>=record['local_before_coverage_pct']-.1,record
 assert record['newly_uncovered_max_gap_mm']<.5,record
 report['links'][name]=record;k['collision_spheres'][name]=current;print('RESULT',name,json.dumps({a:b for a,b in record.items() if a!='changes'}),flush=True)
for n,b in k['collision_spheres'].items():
 if n not in report['links']:assert b==baseline['kinematics']['collision_spheres'][n]
assert k['self_collision_ignore']==baseline['kinematics']['self_collision_ignore']
(ROOT/'rm75_shoulder_local.yml').write_text(yaml.safe_dump(config,sort_keys=False));(ROOT/'shoulder_local_metrics.json').write_text(json.dumps(report,indent=2))
