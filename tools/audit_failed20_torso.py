from pathlib import Path
from functools import partial
import json,itertools,collections,xml.etree.ElementTree as ET
import numpy as np,trimesh,yaml,yourdfpy,fcl
R=Path('/home/fang/robot_ws');out=R/'collision_models/hybrid/failed20_audit';out.mkdir(exist_ok=True)
data=json.loads(Path('/home/fang/.codex/attachments/cdafce39-2e4b-4352-b574-c35f49a662e6/已粘贴的文本.txt').read_text());k=yaml.safe_load(Path(data['robot_config']).read_text())['kinematics'];p=Path(k['urdf_path']);tree=ET.parse(p);names=list(k['collision_spheres']);spheres=k['collision_spheres'];ignored={tuple(sorted((a,b))) for a,bs in k['self_collision_ignore'].items() for b in bs};pairs=[(a,b) for a,b in itertools.combinations(names,2) if tuple(sorted((a,b))) not in ignored]
r=yourdfpy.URDF.load(p,load_meshes=False,filename_handler=partial(yourdfpy.filename_handler_magic,dir=p.parent));base=dict(zip(k['cspace']['joint_names'],k['cspace']['default_joint_position']));base.update(k['lock_joints']);objects={};meshes={}
for link in tree.findall('link'):
 n=link.get('name')
 if n not in names:continue
 parts=[]
 for c in link.findall('collision'):
  e=c.find('geometry/mesh');m=trimesh.load(p.parent/e.get('filename'),force='mesh');m.apply_scale(np.fromstring(e.get('scale','1 1 1'),sep=' '));o=c.find('origin');t=np.eye(4)
  if o is not None:t=trimesh.transformations.euler_matrix(*np.fromstring(o.get('rpy','0 0 0'),sep=' '));t[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
  m.apply_transform(t);parts.append(m)
 m=trimesh.util.concatenate(parts);meshes[n]=m;model=fcl.BVHModel();model.beginModel(len(m.vertices),len(m.faces));model.addSubModel(m.vertices,m.faces.astype(np.int32));model.endModel();objects[n]=fcl.CollisionObject(model)
rows=[];hits=collections.Counter();meshcount=collections.Counter()
for row in data['results']:
 q=dict(base);v=row['joint_positions'];assert len(v)==12;q['platform_joint']=v[4];q.update(dict(zip([f'r_joint{i}' for i in range(1,8)],v[5:])))
 r.update_cfg(q);T={n:r.get_transform(n) for n in names};world={}
 for n in names:
  t=T[n];objects[n].setTransform(fcl.Transform(t[:3,:3],t[:3,3]));world[n]=(np.array([s['center'] for s in spheres[n]])@t[:3,:3].T+t[:3,3],np.array([s['radius'] for s in spheres[n]]))
 checks=[]
 for a,b in pairs:
  ca,ra=world[a];cb,rb=world[b];gap=np.linalg.norm(ca[:,None]-cb[None,:],axis=2)-ra[:,None]-rb[None,:];ii,jj=np.unravel_index(gap.argmin(),gap.shape)
  if gap[ii,jj]>0 and not (a in ['platform_base_link','body_base_link'] and b in ['r_link2','r_link3']):continue
  res=fcl.CollisionResult();hit=bool(fcl.collide(objects[a],objects[b],fcl.CollisionRequest(num_max_contacts=20,enable_contact=True),res));dist=None if hit else float(fcl.distance(objects[a],objects[b],fcl.DistanceRequest(enable_nearest_points=True),fcl.DistanceResult()))
  checks.append({'pair':[a,b],'sphere_gap_mm':float(gap[ii,jj]*1000),'mesh_intersection':hit,'mesh_distance_mm':None if dist is None else dist*1000,'deepest_sphere_indices':[int(ii),int(jj)],'overlapping_sphere_pairs':[[int(i),int(j),float(gap[i,j]*1000)] for i,j in zip(*np.where(gap<0))]})
  if gap[ii,jj]<0:
   for i,j in zip(*np.where(gap<0)):hits[(a,int(i),b,int(j))]+=1
  if hit:meshcount[(a,b)]+=1
 assert abs(min(x['sphere_gap_mm'] for x in checks)/1000-row['final_self_gap_m'])<1e-6
 rows.append({'index':row['index'],'checks':checks});print(row['index'],[(x['pair'],round(x['sphere_gap_mm'],2),x['mesh_intersection'],x['mesh_distance_mm']) for x in checks if x['sphere_gap_mm']<0],flush=True)
report={'source':str(data['robot_config']),'rows':rows,'mesh_intersection_counts':{str(k):v for k,v in meshcount.items()},'overlap_frequency':[{'a':a,'i':i,'b':b,'j':j,'count':n,'sphere_a':spheres[a][i],'sphere_b':spheres[b][j]} for (a,i,b,j),n in hits.most_common()],'mesh_watertight':{n:m.is_watertight for n,m in meshes.items()},'scope':'20 terminal configurations only; FCL triangle surfaces, not certified solid containment; no path/world validation'}
(out/'report.json').write_text(json.dumps(report,indent=2))
