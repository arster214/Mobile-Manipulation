"""Reproducible pairwise FCL triangle-surface reference versus collision spheres."""
from pathlib import Path
from functools import partial
import copy,itertools,json,hashlib,xml.etree.ElementTree as ET
import numpy as np,trimesh,yaml,yourdfpy,fcl
ROOT=Path('/home/fang/robot_ws/collision_models/hybrid'); SRDF=Path('/home/fang/isaacsim_rm75/moveit2/rm75_dual_moveit_config/config/overseas_75_b_v_description.srdf')
refined=yaml.safe_load((ROOT/'rm75_patch_rebuilt_40.yml').read_text());k=refined['kinematics'];source=ROOT/'rm75_individual_refined.yml'
versions={'before_rebuild':yaml.safe_load(source.read_text())['kinematics']['collision_spheres'],'patch30':yaml.safe_load((ROOT/'rm75_patch_rebuilt.yml').read_text())['kinematics']['collision_spheres'],'compact_both':yaml.safe_load((ROOT/'rm75_patch_compact.yml').read_text())['kinematics']['collision_spheres'],'platform_only':yaml.safe_load((ROOT/'rm75_platform_rebuilt.yml').read_text())['kinematics']['collision_spheres']}
urdfpath=Path(k['urdf_path']);tree=ET.parse(urdfpath);names=list(k['collision_spheres']);pairs=list(itertools.combinations(names,2))
def key(a,b):return tuple(sorted((a,b)))
ignored={key(a,b) for a,bs in k['self_collision_ignore'].items() for b in bs if a in names and b in names}
srdf={key(x.get('link1'),x.get('link2')):x.get('reason') for x in ET.parse(SRDF).findall('disable_collisions') if x.get('link1') in names and x.get('link2') in names}
adjacent={key(j.find('parent').get('link'),j.find('child').get('link')) for j in tree.findall('joint')}
meshes={};objects={};corners={};visual_matches={}
def canonical(x):
 return None if x is None else (x.tag,tuple(sorted(x.attrib.items())),tuple(canonical(c) for c in x))
for link in tree.findall('link'):
 name=link.get('name')
 if name not in names:continue
 parts=[]
 for col in link.findall('collision'):
  el=col.find('geometry/mesh');assert el is not None
  m=trimesh.load(urdfpath.parent/el.get('filename'),force='mesh');m.apply_scale(np.fromstring(el.get('scale','1 1 1'),sep=' '))
  origin=col.find('origin');t=np.eye(4)
  if origin is not None:
   t=trimesh.transformations.euler_matrix(*np.fromstring(origin.get('rpy','0 0 0'),sep=' '));t[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
  m.apply_transform(t);parts.append(m)
 m=trimesh.util.concatenate(parts);meshes[name]=m
 visual_matches[name]=all(canonical(v.find('geometry'))==canonical(c.find('geometry')) and (v.find('origin').attrib if v.find('origin') is not None else {})==(c.find('origin').attrib if c.find('origin') is not None else {}) for v,c in zip(link.findall('visual'),link.findall('collision')))
 model=fcl.BVHModel();model.beginModel(len(m.vertices),len(m.faces));model.addSubModel(np.asarray(m.vertices,dtype=float),np.asarray(m.faces,dtype=np.int32));model.endModel();objects[name]=fcl.CollisionObject(model)
 corners[name]=np.array(list(itertools.product(*zip(*m.bounds))))
robot=yourdfpy.URDF.load(urdfpath,build_scene_graph=True,load_meshes=False,filename_handler=partial(yourdfpy.filename_handler_magic,dir=urdfpath.parent))
active=k['cspace']['joint_names'];start=dict(zip(active,k['cspace']['default_joint_position']));start.update(k.get('lock_joints') or {})
limits={j.name:(j.limit.lower,j.limit.upper) for j in robot.robot.joints if j.name in active}
poses=[('default',dict(start))]
for z in [0,.25,.75,1]: q=dict(start);q['platform_joint']=z;poses.append((f'lift_{z}',q))
for name in active:
 lo,hi=limits[name]
 for f in [0,.25,.5,.75,1]:q=dict(start);q[name]=float(lo+(hi-lo)*f);poses.append((f'sweep_{name}_{f}',q))
rng=np.random.default_rng(20261025)
for i in range(128):
 q=dict(start);q.update({n:float(rng.uniform(*limits[n])) for n in active});poses.append((f'random_{i}',q))
rng_holdout=np.random.default_rng(20261026)
for i in range(128):
 q=dict(start);q.update({n:float(rng_holdout.uniform(*limits[n])) for n in active});q['platform_joint']=float(rng_holdout.uniform(0,1));poses.append((f'holdout_{i}',q))
counts_by_group={g:{v:{s:0 for s in ['tp','fp','fn','tn']} for v in versions} for g in ['regression203','holdout128']}
counts={v:{'tp':0,'fp':0,'fn':0,'tn':0} for v in versions};pairdata={};examples=[];cache={};mesh_calls=0
for a,b in pairs:
 p=key(a,b);pairdata[p]={'pair':p,'ignored_curobo':p in ignored,'srdf_reason':srdf.get(p),'adjacent_urdf':p in adjacent,'mesh_hits':0,'versions':{v:{'tp':0,'fp':0,'fn':0,'tn':0} for v in versions}}
request=fcl.CollisionRequest(num_max_contacts=1,enable_contact=False)
for ix,(label,q) in enumerate(poses):
 robot.update_cfg(q);transforms={n:robot.get_transform(n) for n in names};bounds={};world={v:{} for v in versions}
 for n in names:
  t=transforms[n];objects[n].setTransform(fcl.Transform(t[:3,:3],t[:3,3]));pts=corners[n]@t[:3,:3].T+t[:3,3];bounds[n]=(pts.min(0),pts.max(0))
  for v,spheres in versions.items():
   c=np.array([s['center'] for s in spheres[n]]);r=np.array([s['radius'] for s in spheres[n]]);world[v][n]=(c@t[:3,:3].T+t[:3,3],r)
 for a,b in pairs:
  p=key(a,b);pd=pairdata[p];lo,hi=bounds[a];lo2,hi2=bounds[b];hit=False
  if np.all(hi>=lo2) and np.all(hi2>=lo):
   relative=np.linalg.inv(transforms[a])@transforms[b];ck=(p,tuple(np.round(relative,8).ravel()))
   if ck not in cache:
    cache[ck]=bool(fcl.collide(objects[a],objects[b],request,fcl.CollisionResult()));mesh_calls+=1
   hit=cache[ck]
  pd['mesh_hits']+=int(hit)
  for v in versions:
   ca,ra=world[v][a];cb,rb=world[v][b];gap=float((np.linalg.norm(ca[:,None]-cb[None],axis=2)-ra[:,None]-rb[None]).min());ball_hit=gap<=0
   outcome='tp' if hit and ball_hit else 'fn' if hit else 'fp' if ball_hit else 'tn'
   pd['versions'][v][outcome]+=1
   if p not in ignored:
    counts[v][outcome]+=1
    counts_by_group['holdout128' if label.startswith('holdout_') else 'regression203'][v][outcome]+=1
   if v=='platform_only' and ((outcome in ('fp','fn') and p not in ignored) or (hit and srdf.get(p)=='Never')):
    if len([e for e in examples if e['pair']==list(p) and e['outcome']==outcome and e['ignored_curobo']==(p in ignored)])<2:
     examples.append({'pose':label,'q':q,'pair':list(p),'outcome':outcome,'sphere_gap_mm':gap*1000,'ignored_curobo':p in ignored,'srdf_reason':srdf.get(p)})
 if ix%20==0:print('pose',ix,'/',len(poses),'FCL calls',mesh_calls,flush=True)
report={'counts_by_group':counts_by_group,'holdout_recipe':'128 independent random poses seed 20261026 with randomized lift 0..1 m','pose_count':len(poses),'pose_recipe':'default + 4 lift positions + 70 single-joint samples + 128 joint-limit uniform random poses at lift=0.5; seed 20261025','pair_count':len(pairs),'checked_pairs_after_curobo_ignore':len(pairs)-len(ignored),'confusion_after_curobo_ignore':counts,'srdf_path':str(SRDF),'srdf_sha256':hashlib.sha256(SRDF.read_bytes()).hexdigest(),'urdf_path':str(urdfpath),'visual_collision_geometry_match':visual_matches,'mesh_watertight':{n:bool(m.is_watertight) for n,m in meshes.items()},'curobo_ignore_count':len(ignored),'srdf_ignore_count':len(srdf),'curobo_only_ignored':sorted(ignored-set(srdf)),'srdf_only_ignored':sorted(set(srdf)-ignored),'adjacent_not_ignored':sorted(p for p in adjacent-set(ignored) if all(n in names for n in p)),'fcl_calls':mesh_calls,'pairs':list(pairdata.values()),'examples':examples,'limitations':'FCL triangle intersection is a surface reference, not certified solid containment for open meshes. Finite pose sampling does not establish all-configuration equivalence. Raw sphere overlap uses zero configured buffers. No CUDA cuRobo kernel execution or motion planning certification.'}
(ROOT/'patch_comparison_audit.json').write_text(json.dumps(report,indent=2))
print('RESULT',json.dumps(counts),flush=True);print('IGNORE',len(ignored),len(srdf),'differences',len(ignored-set(srdf)),len(set(srdf)-ignored),flush=True)
