"""Symmetric incremental edits; fixed count, unchanged collision exclusions."""
exec(open('/home/fang/robot_ws/tools/audit_failed20_torso.py').read().split('rows=[];hits=')[0])
from scipy.optimize import linear_sum_assignment
import copy
from build_sphere_comparison import load_mesh
O=R/'collision_models/hybrid/symmetric_adjustment';O.mkdir(exist_ok=True)
config=yaml.safe_load(Path(data['robot_config']).read_text());orig=copy.deepcopy(spheres)
hot={'platform_base_link':[1,2,3,4,44],'body_base_link':[86,95,96]};groups={};correspondence={}
for name,ids in hot.items():
 c=np.array([b['center'] for b in orig[name]]);negative=np.where(c[:,0]<-1e-4)[0];positive=np.where(c[:,0]>1e-4)[0];ref=c[negative]*[-1,1,1];cost=np.linalg.norm(ref[:,None]-c[positive],axis=2);ii,jj=linear_sum_assignment(cost);mapping={int(negative[i]):int(positive[j]) for i,j in zip(ii,jj)};mapping.update({v:k for k,v in list(mapping.items())});groups[name]=sorted(set(ids+[mapping[i] for i in ids]));correspondence[name]=[{'index':i,'mirror_index':mapping[i],'center_mismatch_mm':float(np.linalg.norm(c[i]*[-1,1,1]-c[mapping[i]])*1000)} for i in ids]
poses=[]
for row in data['results']:
 q=dict(base);v=row['joint_positions'];q['platform_joint']=v[4];q.update(dict(zip([f'r_joint{i}' for i in range(1,8)],v[5:])));poses.append(q)
# Holdout: independent right/left arm configurations with changing lift.
rng=np.random.default_rng(20261027);holdout=[]
for _ in range(128):
 q=dict(base)
 for n in k['cspace']['joint_names']:
  j=r.joint_map[n];q[n]=float(rng.uniform(j.limit.lower,j.limit.upper))
 q['platform_joint']=float(rng.uniform(0,1));holdout.append(q)
Ts=[]
for q in poses+[base]+holdout:
 r.update_cfg(q);Ts.append({n:r.get_transform(n).copy() for n in names})
def sphere_gaps(bs,T):
 world={n:(np.array([s['center'] for s in bs[n]])@T[n][:3,:3].T+T[n][:3,3],np.array([s['radius'] for s in bs[n]])) for n in names}
 return np.array([(np.linalg.norm(world[a][0][:,None]-world[b][0][None,:],axis=2)-world[a][1][:,None]-world[b][1][None,:]).min() for a,b in pairs])
oldg=np.array([sphere_gaps(orig,T) for T in Ts]);samples={};oldd={};changed=list(hot)+['l_base_link1','r_base_link1']
def gaps_points(p,balls):
 c=np.array([b['center'] for b in balls]);rr=np.array([b['radius'] for b in balls]);return np.concatenate([(np.linalg.norm(x[:,None]-c,axis=2)-rr).min(1) for x in np.array_split(p,20)])
for n in changed:
 np.random.seed(20261027);samples[n]=trimesh.sample.sample_surface(meshes[n],100000)[0];oldd[n]=gaps_points(samples[n],orig[n])
results=[]
for strength in [.5,1.,1.5,2.,2.5]:
 bs=copy.deepcopy(orig)
 for n,ids in groups.items():
  for i in ids:
   b=bs[n][i];c=np.array(b['center'])
   if n=='platform_base_link':c[0]-=np.sign(c[0])*.008*strength;b['radius']-=.004*strength
   else:c[1]+=.002*strength;b['radius']-=.0015*strength
   b['center']=c.tolist()
 for side in ['l','r']:bs[side+'_base_link1'][2]['radius']-=.002*strength
 g=np.array([sphere_gaps(bs,T) for T in Ts]);metrics={}
 for n in changed:
  after=gaps_points(samples[n],bs[n]);before=oldd[n];ids=groups.get(n,[2]);c=np.array([orig[n][i]['center'] for i in ids]);rr=np.array([orig[n][i]['radius'] for i in ids]);local=(np.linalg.norm(samples[n][:,None]-c,axis=2)-rr).min(1)<=0
  metrics[n]={'coverage_before_pct':float(np.mean(before<=0)*100),'coverage_after_pct':float(np.mean(after<=0)*100),'local_coverage_after_pct':float(np.mean(after[local]<=0)*100),'newly_uncovered_max_mm':float(max(0,after[before<=0].max())*1000),'surface_gap_p99_mm':float(max(0,np.quantile(after,.99))*1000)}
 # FCL audit only decisions changed from collision to clear, including holdout.
 lost=np.argwhere((oldg<=0)&(g>0));new_misses=[]
 for pi,pj in lost:
  a,b=pairs[pj];T=Ts[pi]
  for n in [a,b]:objects[n].setTransform(fcl.Transform(T[n][:3,:3],T[n][:3,3]))
  if fcl.collide(objects[a],objects[b],fcl.CollisionRequest(),fcl.CollisionResult()):new_misses.append({'pose':int(pi),'pair':[a,b]})
 cc=copy.deepcopy(config);cc['kinematics']['collision_spheres']=bs;path=O/f'candidate_{strength:g}.yml';path.write_text(yaml.safe_dump(cc,sort_keys=False))
 row={'strength':strength,'config':str(path),'failed20_now_clear':int(np.sum(g[:20].min(1)>0)),'failed20_worst_gap_mm':float(g[:20].min()*1000),'initial_gap_mm':float(g[20].min()*1000),'holdout_pair_overlaps_before':int(np.sum(oldg[21:]<=0)),'holdout_pair_overlaps_after':int(np.sum(g[21:]<=0)),'new_mesh_misses_on_changed_decisions':new_misses,'coverage':metrics};results.append(row);print(json.dumps(row),flush=True)
(O/'report.json').write_text(json.dumps({'method':'Equal mirror-paired increments; original sphere layout is not exactly symmetric. No ball count or exclusion edits. Platform move toward X=0 by 8*s mm and shrink 4*s mm. Body move +Y 2*s mm and shrink 1.5*s mm. Both shoulder base sphere 2 shrink 2*s mm.','mirror_mapping':correspondence,'modified_indices':groups,'holdout_seed':20261027,'holdout_count':128,'sphere_count':sum(map(len,orig.values())),'results':results,'scope':'CPU geometric sphere checks and FCL changed-decision regression; no trajectory/world validation. Area samples cannot certify full coverage.'},indent=2))
