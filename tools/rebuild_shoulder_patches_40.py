"""Replace collision-prone body balls with surface patch balls, triangle coverage checked."""
from pathlib import Path
from functools import partial
import copy,json,collections
import numpy as np,trimesh,yaml,yourdfpy
from build_sphere_comparison import load_mesh,clip_polygon
ROOT=Path('/home/fang/robot_ws/collision_models/hybrid')
source=max(ROOT.glob('rm75_hybrid_edited_*.yml'),key=lambda p:p.stat().st_mtime)
config=yaml.safe_load(source.read_text());k=config['kinematics'];audit=json.loads((ROOT/'individual_refined_collision_audit.json').read_text())
p=Path(k['urdf_path']);robot=yourdfpy.URDF.load(p,build_scene_graph=True,load_meshes=False,filename_handler=partial(yourdfpy.filename_handler_magic,dir=p.parent))
report={'source':str(source),'method':'Select body/platform balls from regression false-positive witnesses only; replace their intersecting mesh triangles with capped-radius surface patches. Every selected triangle is either inside one retained sphere or subdivided and covered by a new sphere. No test holdout poses used for selection.','links':{}}
def gap(pts,balls):
 c=np.array([s['center'] for s in balls]);r=np.array([s['radius'] for s in balls]);return np.concatenate([(np.linalg.norm(pts[i:i+5000,None]-c,axis=2)-r).min(1) for i in range(0,len(pts),5000)])
def tri_covered(t,balls):
 if not len(balls):return False
 c=np.array([s['center'] for s in balls]);r=np.array([s['radius'] for s in balls]);return bool(np.any(np.linalg.norm(t[:,None]-c,axis=2).max(0)<=r-1e-7))
for name,limit,cap in [('platform_base_link',8,.040),('body_base_link',6,.040)]:
 old=k['collision_spheres'][name];oc=np.array([s['center'] for s in old]);rr=np.array([s['radius'] for s in old]);scores=collections.Counter();witnesses=[]
 for e in audit['examples']:
  if name not in e['pair'] or e['outcome']!='fp' or e['ignored_curobo'] or e['pose'].startswith('holdout_'):continue
  other=next(n for n in e['pair'] if n!=name)
  if not other.startswith(('l_','r_')):continue
  robot.update_cfg(e['q']);t=np.linalg.inv(robot.get_transform(name))@robot.get_transform(other)
  ob=k['collision_spheres'][other];cent=np.array([s['center'] for s in ob])@t[:3,:3].T+t[:3,3];rad=np.array([s['radius'] for s in ob]);hits=np.where((np.linalg.norm(oc[:,None]-cent,axis=2)-rr[:,None]-rad).min(1)<0)[0]
  for i in hits:scores[int(i)]+=1
  witnesses.append({'pose':e['pose'],'pair':e['pair'],'sphere_indices':hits.tolist()})
 selected=[i for i,_ in scores.most_common(limit)];assert selected
 kept=[s for i,s in enumerate(old) if i not in selected];mesh=load_mesh(p,name);tri=mesh.triangles;patches=[]
 for i in selected:
  lo=oc[i]-rr[i]-.0005;hi=oc[i]+rr[i]+.0005
  candidates=tri[np.all(tri.max(1)>=lo,axis=1)&np.all(tri.min(1)<=hi,axis=1)]
  for t in candidates:
   poly=t
   for axis in range(3):
    poly=clip_polygon(poly,axis,lo[axis],True);poly=clip_polygon(poly,axis,hi[axis],False)
   for j in range(1,len(poly)-1):patches.append(np.array([poly[0],poly[j],poly[j+1]]))
 patch=np.asarray(patches);queue=list(patch);tiles=[]
 while queue:
  t=queue.pop()
  if tri_covered(t,kept):continue
  edges=np.array([np.linalg.norm(t[0]-t[1]),np.linalg.norm(t[1]-t[2]),np.linalg.norm(t[2]-t[0])])
  if edges.max()<=cap*.7:tiles.append(t);continue
  a,b=[(0,1),(1,2),(2,0)][edges.argmax()];c=3-a-b;mid=(t[a]+t[b])/2
  queue.extend([np.array([t[a],mid,t[c]]),np.array([mid,t[b],t[c]])])
 tiles=np.asarray(tiles);added=[];remaining=np.ones(len(tiles),bool)
 areas=trimesh.triangles.area(tiles);centroids=tiles.mean(1)
 while remaining.any():
  # Greedy patch cover, using a deterministic set of largest-area seed triangles.
  ids=np.flatnonzero(remaining);seeds=ids[np.argsort(areas[ids])[-min(32,len(ids)):]];best=None
  for i in seeds:
   c=centroids[i];d=np.linalg.norm(tiles[ids]-c,axis=2).max(1);covered=d<=cap-.001;score=areas[ids[covered]].sum()
   if best is None or score>best[0]:best=(score,c,ids[covered],d[covered])
  _,c,chosen,ds=best;assert len(chosen)
  radius=min(cap,float(np.ceil((ds.max()+.0005)/.001)*.001));added.append({'center':c.tolist(),'radius':radius});remaining[chosen]=False
 balls=kept+added
 np.random.seed(11811);samples=trimesh.sample.sample_surface(mesh,400000)[0];before=gap(samples,old);after=gap(samples,balls)
 local=(np.linalg.norm(samples[:,None]-oc[selected],axis=2)<=rr[selected]+.003).any(1)
 v={'selected_indices':selected,'scores':dict(scores),'witnesses':witnesses,'count_before':len(old),'retained':len(kept),'added':len(added),'count_after':len(balls),'new_radius_cap_mm':cap*1000,'new_radius_min_mm':min(s['radius'] for s in added)*1000,'original_patch_triangles':len(patch),'subdivided_tiles':len(tiles),'coverage_before_pct':float(np.mean(before<=0)*100),'coverage_after_pct':float(np.mean(after<=0)*100),'local_before_pct':float(np.mean(before[local]<=0)*100),'local_after_pct':float(np.mean(after[local]<=0)*100),'newly_uncovered_max_mm':float(max(0,after[before<=0].max())*1000)}
 assert v['newly_uncovered_max_mm']<1e-5,v
 report['links'][name]=v;k['collision_spheres'][name]=balls
 print(name,json.dumps({a:b for a,b in v.items() if a not in ['witnesses','scores']}),flush=True)
(ROOT/'rm75_patch_rebuilt_40.yml').write_text(yaml.safe_dump(config,sort_keys=False));(ROOT/'patch_rebuilt_40_metrics.json').write_text(json.dumps(report,indent=2))
print('TOTAL',sum(map(len,k['collision_spheres'].values())),flush=True)
