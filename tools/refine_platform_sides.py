"""Replace four redundant torso balls with mirrored, clearance-constrained small balls."""
exec(open('/home/fang/robot_ws/tools/audit_failed20_torso.py').read().split('rows=[];hits=')[0])
import copy,hashlib
O=R/'collision_models/hybrid/platform_sides_refined';O.mkdir(exist_ok=True)
source=Path(data['robot_config']);cfg=yaml.safe_load(source.read_text());old=copy.deepcopy(k['collision_spheres']);name='platform_base_link';removed=[1,2,3,4,44,26,27,28,29,63];kept=[b for i,b in enumerate(old[name]) if i not in removed]
poses=[]
for row in data['results']:
 q=dict(base);v=row['joint_positions'];q['platform_joint']=v[4];q.update(dict(zip([f'r_joint{i}' for i in range(1,8)],v[5:])));poses.append(q)
poses.append(dict(base));rng=np.random.default_rng(20261029)
for _ in range(128):
 q=dict(base)
 for n in k['cspace']['joint_names']:
  j=r.joint_map[n];q[n]=float(rng.uniform(j.limit.lower,j.limit.upper))
 q['platform_joint']=float(rng.uniform(0,1));poses.append(q)
Ts=[];obc=[];obr=[]
for ix,q in enumerate(poses):
 r.update_cfg(q);T={n:r.get_transform(n).copy() for n in names};Ts.append(T)
 if ix<21:
  for n in names:
   if not n.startswith(('l_','r_')) or tuple(sorted((name,n))) in ignored:continue
   rel=np.linalg.inv(T[name])@T[n];obc.extend(np.array([b['center'] for b in old[n]])@rel[:3,:3].T+rel[:3,3]);obr.extend([b['radius'] for b in old[n]])
obc=np.array(obc);obr=np.array(obr);m=meshes[name];np.random.seed(20261029);samples,face=trimesh.sample.sample_surface(m,120000);normal=m.face_normals[face]
def gap(p,balls):
 c=np.array([b['center'] for b in balls]);rr=np.array([b['radius'] for b in balls]);return np.concatenate([(np.linalg.norm(x[:,None]-c,axis=2)-rr).min(1) for x in np.array_split(p,40)])
before=gap(samples,old[name]);after=gap(samples,kept);local=(np.linalg.norm(samples[:,None]-np.array([old[name][i]['center'] for i in removed]),axis=2)-np.array([old[name][i]['radius'] for i in removed])).min(1)<=0
added=[]
for step in range(48):
 if np.mean(after[local]<=0)>=.955:break
 bad=np.flatnonzero((after>.003)&(before<=0));
 if not len(bad):break
 seeds=bad[np.linspace(0,len(bad)-1,min(160,len(bad))).astype(int)];best=None
 for idx in seeds:
  c=samples[idx]-normal[idx]*.003;cm=c*[-1,1,1];rad=min(.025,float((np.linalg.norm(obc-c,axis=1)-obr).min())-.001,float((np.linalg.norm(obc-cm,axis=1)-obr).min())-.001)
  if rad<.008:continue
  dist=np.minimum(np.linalg.norm(samples[bad]-c,axis=1),np.linalg.norm(samples[bad]-cm,axis=1));score=int(np.sum(dist<=rad+.003))
  if best is None or score>best[0]:best=(score,c,cm,rad)
 if best is None or best[0]==0:break
 _,c,cm,rad=best;added.extend([{'center':x.tolist(),'radius':rad} for x in [c,cm]]);after=np.minimum(after,np.minimum(np.linalg.norm(samples-c,axis=1),np.linalg.norm(samples-cm,axis=1))-rad)
new=copy.deepcopy(old);new[name]=kept+added;cfg['kinematics']['collision_spheres']=new
# Independent surface validation.
np.random.seed(20261030);val=trimesh.sample.sample_surface(m,200000)[0];vb=gap(val,old[name]);va=gap(val,new[name]);vc=np.array([old[name][i]['center'] for i in removed]);vr=np.array([old[name][i]['radius'] for i in removed]);vl=(np.linalg.norm(val[:,None]-vc,axis=2)-vr).min(1)<=0
pairlist=[p for p in pairs if name in p];checks=[];new_misses=[]
for ix,T in enumerate(Ts):
 def calc(bs,a,b):
  ca=np.array([x['center'] for x in bs[a]])@T[a][:3,:3].T+T[a][:3,3];cb=np.array([x['center'] for x in bs[b]])@T[b][:3,:3].T+T[b][:3,3];return float((np.linalg.norm(ca[:,None]-cb[None,:],axis=2)-np.array([x['radius'] for x in bs[a]])[:,None]-np.array([x['radius'] for x in bs[b]])[None,:]).min())
 current=[]
 for a,b in pairlist:
  g0=calc(old,a,b);g1=calc(new,a,b)
  if g0<=0 or g1<=0:
   for n in [a,b]:objects[n].setTransform(fcl.Transform(T[n][:3,:3],T[n][:3,3]))
   hit=bool(fcl.collide(objects[a],objects[b],fcl.CollisionRequest(),fcl.CollisionResult()))
   current.append({'pair':[a,b],'before_gap_mm':g0*1000,'after_gap_mm':g1*1000,'mesh_hit':hit})
   if hit and g0<=0 and g1>0:new_misses.append({'pose':ix,'pair':[a,b]})
 checks.append(current)
report={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'removed_indices':removed,'removed_spheres':[old[name][i] for i in removed],'added_mirrored_spheres':added,'platform_count_before':len(old[name]),'platform_count_after':len(new[name]),'total_before':sum(map(len,old.values())),'total_after':sum(map(len,new.values())),'surface_coverage_before_pct':float(np.mean(vb<=0)*100),'surface_coverage_after_pct':float(np.mean(va<=0)*100),'local_surface_coverage_after_pct':float(np.mean(va[vl]<=0)*100),'max_new_gap_mm':float(max(0,va[vb<=0].max())*1000),'failed20_platform_overlap_before':sum(any(c['before_gap_mm']<0 for c in row) for row in checks[:20]),'failed20_platform_overlap_after':sum(any(c['after_gap_mm']<0 for c in row) for row in checks[:20]),'initial_platform_pairs':checks[20],'holdout128_overlaps_before':sum(c['before_gap_mm']<0 for row in checks[21:] for c in row),'holdout128_overlaps_after':sum(c['after_gap_mm']<0 for row in checks[21:] for c in row),'new_mesh_misses':new_misses,'checks':checks,'scope':'Only platform_base_link modified; torso refinement retained and arms unchanged. Surface area sampling, FCL open-mesh limitations, no world/path validation.'}
(O/'candidate.yml').write_text(yaml.safe_dump(cfg,sort_keys=False));(O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({a:b for a,b in report.items() if a not in ['checks','removed_spheres','added_mirrored_spheres']},indent=2),flush=True)
