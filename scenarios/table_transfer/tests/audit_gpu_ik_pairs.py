"""Independent Pinocchio FK and sphere collision audit of GPU IK endpoints."""
from pathlib import Path
import json, math, csv
import numpy as np
import pinocchio as pin
import yaml
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'ik_test/gpu_collision20'
r=json.loads((OUT/'result.json').read_text()); k=yaml.safe_load((OUT/'whole_body.yml').read_text())['kinematics']
model=pin.buildModelFromUrdf(r['urdf']);data=model.createData();names=r['joint_names']
links=[];radii=[];spheres=[]
for link,ss in k['collision_spheres'].items():
 for s in ss:
  if s['radius']>0: links.append(link);radii.append(s['radius']);spheres.append(s['center'])
radii=np.array(radii);spheres=np.array(spheres)
ignore=k['self_collision_ignore']; pairs=np.array([(i,j) for i in range(len(links)) for j in range(i+1,len(links)) if links[i]!=links[j] and links[j] not in ignore.get(links[i],[]) and links[i] not in ignore.get(links[j],[])])
world=yaml.safe_load((OUT/'world.yml').read_text())['cuboid']

def evaluate(q):
 full=pin.neutral(model); values=dict(k['lock_joints']);values.update(zip(names,q))
 for side in ('l','r'):values[f'{side}_Joint_finger2']=values[f'{side}_Joint_finger1']
 for n,v in values.items():full[model.joints[model.getJointId(n)].idx_q]=v
 pin.framesForwardKinematics(model,data,full)
 frames={f.name:data.oMf[i].homogeneous.tolist() for i,f in enumerate(model.frames) if f.type==pin.FrameType.BODY}
 centers=np.array([data.oMf[model.getFrameId(link)].act(center) for link,center in zip(links,spheres)])
 gaps=np.linalg.norm(centers[pairs[:,0]]-centers[pairs[:,1]],axis=1)-radii[pairs[:,0]]-radii[pairs[:,1]]
 bad=np.where(gaps< -1e-6)[0]; j=int(np.argmin(gaps)); pair=pairs[j]
 world_gaps=[]
 for name,box in world.items():
  d=np.abs(centers-np.array(box['pose'][:3]))-np.array(box['dims'])/2
  sdf=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(np.max(d,axis=1),0)
  world_gaps.append((float(np.min(sdf-radii)),name))
 worst=min(world_gaps)
 within=bool(np.all(np.abs(centers[:,:2])+radii[:,None]<=3.5))
 limits=bool(np.all(full>=model.lowerPositionLimit-1e-6) and np.all(full<=model.upperPositionLimit+1e-6))
 return {'self_collision_free':len(bad)==0,'min_self_gap_m':float(gaps[j]),'closest_self_links':[links[pair[0]],links[pair[1]]],'colliding_sphere_pairs':len(bad),'min_world_gap_m':worst[0],'closest_obstacle':worst[1],'world_collision_free':worst[0]>=-1e-6 and within,'joint_limits_ok':limits},frames

poses=[];target=np.array(r['target_tcp_world'])
for row in r['rows']:
 start,sframes=evaluate(row['start_q']);end,eframes=evaluate(row['end_q'])
 actual=np.array(eframes['r_grasp_tcp']);pe=float(np.linalg.norm(actual[:3,3]-target[:3,3]));re=float(np.linalg.norm(pin.log3(target[:3,:3].T@actual[:3,:3])))
 row.update({'start_audit':start,'end_audit':end,'independent_position_error_m':pe,'independent_rotation_error_rad':re,'accepted':row['solver_success'] and end['self_collision_free'] and end['world_collision_free'] and end['joint_limits_ok'] and pe<=.0001 and re<=.001})
 poses.append({'index':row['index'],'start':sframes,'end':eframes})
 print(row['index'],row['solver_success'],row['accepted'],start['self_collision_free'],start['world_collision_free'],end['min_self_gap_m'],end['min_world_gap_m'],pe)
r['accepted_count']=sum(x['accepted'] for x in r['rows']);r['all_starts_valid_under_new_model']=all(x['start_audit']['self_collision_free'] and x['start_audit']['world_collision_free'] for x in r['rows'])
r['independent_audit']='Pinocchio FK + sphere-sphere / sphere-AABB distances, original ignore pairs, numerical tolerance 1 micrometer; no swept path check'
(OUT/'result.json').write_text(json.dumps(r,indent=2)+'\n');(OUT/'pair_frames.json').write_text(json.dumps(poses)+'\n')
with (OUT/'summary.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['pair','solver_success','accepted','wall_ms','start_x','start_y','start_yaw','end_x','end_y','end_yaw','end_lift','self_gap_m','world_gap_m'])
 for x in r['rows']:writer.writerow([x['index'],x['solver_success'],x['accepted'],x['wall_ms'],*x['start_q'][:3],*x['end_q'][:4],x['end_audit']['min_self_gap_m'],x['end_audit']['min_world_gap_m']])
