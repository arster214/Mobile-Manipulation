"""Local lift-sphere audit and refinement; artifacts stay under tests/artifacts."""
from pathlib import Path
from functools import partial
import json,copy,xml.etree.ElementTree as ET
import numpy as np,yaml,trimesh,yourdfpy,fcl
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tests/artifacts/lift_spheres';OUT.mkdir(parents=True,exist_ok=True)
SOURCE=Path('/home/fang/robot_ws/postures/rm75_mobile_manipulation.yml')
config=yaml.safe_load(SOURCE.read_text());k=config['kinematics'];urdf=Path(k['urdf_path']);tree=ET.parse(urdf)
state=json.loads((ROOT/'tests/artifacts/transport/result.json').read_text())['start_state']
initial=dict(zip(state['joint_names'],state['positions']));lift=initial['platform_joint']
robot=yourdfpy.URDF.load(urdf,build_scene_graph=True,load_meshes=False,filename_handler=partial(yourdfpy.filename_handler_magic,dir=urdf.parent))

def load_mesh(name):
 parts=[]
 for collision in tree.find(f"link[@name='{name}']").findall('collision'):
  item=collision.find('geometry/mesh');mesh=trimesh.load(urdf.parent/item.get('filename'),force='mesh')
  mesh.apply_scale(np.fromstring(item.get('scale','1 1 1'),sep=' '));origin=collision.find('origin')
  transform=np.eye(4)
  if origin is not None:
   transform=trimesh.transformations.euler_matrix(*np.fromstring(origin.get('rpy','0 0 0'),sep=' '));transform[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
  mesh.apply_transform(transform);parts.append(mesh)
 return trimesh.util.concatenate(parts)

links=[];indices=[];centers=[];radii=[]
for name,balls in k['collision_spheres'].items():
 for i,ball in enumerate(balls):
  if ball['radius']<=0:continue
  links.append(name);indices.append(i);centers.append(ball['center']);radii.append(ball['radius'])
centers=np.array(centers);radii=np.array(radii);ignore=k['self_collision_ignore']
pairs=np.array([(i,j) for i in range(len(links)) for j in range(i+1,len(links)) if links[i]!=links[j] and links[j] not in ignore.get(links[i],[]) and links[i] not in ignore.get(links[j],[])])
records={};transforms={};mesh_cache={};objects={}
for value in sorted(set(np.r_[np.linspace(max(0,lift-.1),lift,41),np.linspace(0,.79,80)].tolist())):
 values=initial.copy();values['platform_joint']=value;robot.update_cfg(values)
 poses={name:robot.get_transform(name) for name in set(links)};transforms[str(value)]={name:t.tolist() for name,t in poses.items()}
 world=np.array([poses[name][:3,:3]@c+poses[name][:3,3] for name,c in zip(links,centers)])
 gaps=np.linalg.norm(world[pairs[:,0]]-world[pairs[:,1]],axis=1)-radii[pairs[:,0]]-radii[pairs[:,1]]
 for ix in np.where(gaps < -1e-6)[0]:
  a,b=pairs[ix];na,nb=links[a],links[b];key=f'{na}:{indices[a]} / {nb}:{indices[b]}'
  if key not in records or gaps[ix]<records[key]['gap_m']:
   for name in (na,nb):
    if name not in objects:
     mesh=load_mesh(name);mesh_cache[name]=mesh;geometry=fcl.BVHModel();geometry.beginModel(len(mesh.vertices),len(mesh.faces));geometry.addSubModel(np.asarray(mesh.vertices,dtype=float),np.asarray(mesh.faces,dtype=np.int32));geometry.endModel();objects[name]=fcl.CollisionObject(geometry)
    t=poses[name];objects[name].setTransform(fcl.Transform(t[:3,:3],t[:3,3]))
   hit=bool(fcl.collide(objects[na],objects[nb],fcl.CollisionRequest(),fcl.CollisionResult()))
   records[key]={'a':na,'a_index':indices[a],'b':nb,'b_index':indices[b],'gap_m':float(gaps[ix]),'lift_q':value,'mesh_surface_collision':hit,
                 'within_10cm_lift':bool(lift-.1-1e-8<=value<=lift+1e-8)}
report={'initial_lift_q':lift,'sphere_count':len(radii),'records':sorted(records.values(),key=lambda x:x['gap_m']),
        'mesh_reference_limitation':'triangle-surface intersection; not certified solid containment or real hardware geometry'}
(OUT/'audit_before.json').write_text(json.dumps(report,indent=2));(OUT/'transforms.json').write_text(json.dumps(transforms))
print(json.dumps(report,indent=2),flush=True)
# First measure the smallest radius-only correction for the actual 10 cm lift.
poses_selected=[poses for value,poses in transforms.items() if lift-.1-1e-8<=float(value)<=lift+1e-8]
body='body_base_link';original=copy.deepcopy(k['collision_spheres'][body]);caps=np.array([b['radius'] for b in original])
for poses in poses_selected:
 t=np.array(poses[body]);bc=np.array([b['center'] for b in original])@t[:3,:3].T+t[:3,3]
 for other in ('l_base_link1','r_base_link1'):
  ot=np.array(poses[other]);ss=k['collision_spheres'][other];oc=np.array([b['center'] for b in ss])@ot[:3,:3].T+ot[:3,3];rr=np.array([b['radius'] for b in ss])
  limits=(np.linalg.norm(bc[:,None]-oc[None],axis=2)-rr[None]-.0001).min(1)
  caps=np.minimum(caps,limits)
changed=np.where(caps<np.array([b['radius'] for b in original])-1e-8)[0]
mesh=load_mesh(body);np.random.seed(4201);samples=trimesh.sample.sample_surface(mesh,200000)[0]
c=np.array([b['center'] for b in original]);r=np.array([b['radius'] for b in original])
def gaps(points,cc,rr):
 return np.concatenate([(np.linalg.norm(batch[:,None]-cc[None],axis=2)-rr).min(1) for batch in np.array_split(points,50)])
before=gaps(samples,c,r);after=gaps(samples,c,caps)
metrics={'changed_indices':changed.tolist(),'changes':[{'index':int(i),'old_radius_mm':r[i]*1000,'new_radius_mm':caps[i]*1000} for i in changed],
 'coverage_before_pct':float(np.mean(before<=0)*100),'coverage_after_pct':float(np.mean(after<=0)*100),
 'newly_uncovered_max_mm':float(max(0,after[before<=0].max())*1000)}
print('RADIUS_ONLY',json.dumps(metrics),flush=True)
(OUT/'radius_only_metrics.json').write_text(json.dumps(metrics,indent=2))
np.savez_compressed(OUT/'body_samples.npz',points=samples,before=before)
proposal=copy.deepcopy(config)
for i in changed:proposal['kinematics']['collision_spheres'][body][i]['radius']=float(caps[i])
(OUT/'radius_only.yml').write_text(yaml.safe_dump(proposal,sort_keys=False))
