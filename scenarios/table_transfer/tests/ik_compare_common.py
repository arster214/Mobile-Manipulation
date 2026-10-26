"""Shared geometry, seed bank and acceptance rules for the CPU/GPU comparison."""
from pathlib import Path
import json
import numpy as np
import pinocchio as pin
import yaml
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'ik_test/gpu_heading20'
OUT=ROOT/'ik_test/comparison'

class Problem:
 def __init__(self):
  self.reference=json.loads((SOURCE/'result.json').read_text())
  self.cfg=yaml.safe_load((SOURCE/'whole_body.yml').read_text())
  self.k=self.cfg['kinematics'];self.names=self.reference['joint_names']
  full=pin.buildModelFromUrdf(self.k['urdf_path']);q=pin.neutral(full)
  values=dict(self.k['lock_joints'])
  for side in ('l','r'):values[f'{side}_Joint_finger2']=values[f'{side}_Joint_finger1']
  for n,v in values.items():q[full.joints[full.getJointId(n)].idx_q]=v
  self.model=pin.buildReducedModel(full,[i for i in range(1,full.njoints) if full.names[i] not in self.names],q)
  assert list(self.model.names)[1:]==self.names
  self.data=self.model.createData();self.target=np.array(self.reference['target_tcp_world']);self.cup=np.array(self.reference['cup']['center_m'])
  self.links=[];self.local=[];self.radii=[];self.frame=[]
  for link,ss in self.k['collision_spheres'].items():
   for s in ss:
    if s['radius']>0:self.links.append(link);self.local.append(s['center']);self.radii.append(s['radius']);self.frame.append(self.model.getFrameId(link))
  self.local=np.array(self.local);self.radii=np.array(self.radii);self.frame=np.array(self.frame)
  self.groups=[(fid,np.flatnonzero(self.frame==fid)) for fid in np.unique(self.frame)]
  ig=self.k['self_collision_ignore']
  self.pairs=np.array([(i,j) for i in range(len(self.links)) for j in range(i+1,len(self.links)) if self.links[i]!=self.links[j] and self.links[j] not in ig.get(self.links[i],[]) and self.links[i] not in ig.get(self.links[j],[])])
  world=yaml.safe_load((SOURCE/'world.yml').read_text())['cuboid']
  self.box_c=np.array([b['pose'][:3] for b in world.values()]);self.box_h=np.array([b['dims'] for b in world.values()])/2
 def geometry(self,q,jac=False):
  pin.computeJointJacobians(self.model,self.data,q);pin.updateFramePlacements(self.model,self.data)
  centers=np.empty_like(self.local);J=np.zeros((len(self.local),3,self.model.nv)) if jac else None
  for fid,ids in self.groups:
   T=self.data.oMf[int(fid)];offsets=self.local[ids]@T.rotation.T;centers[ids]=offsets+T.translation
   if jac:
    jf=pin.getFrameJacobian(self.model,self.data,int(fid),pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    for i,offset in zip(ids,offsets):J[i]=jf[:3]-pin.skew(offset)@jf[3:]
  a,b=self.pairs.T;delta=centers[a]-centers[b];dist=np.linalg.norm(delta,axis=1);sg=dist-self.radii[a]-self.radii[b]
  d=np.abs(centers[:,None,:]-self.box_c)-self.box_h;positive=np.maximum(d,0);norm=np.linalg.norm(positive,axis=2)
  sdf=norm+np.minimum(np.max(d,axis=2),0);wg=sdf-self.radii[:,None]
  return centers,J,sg,wg,delta,dist,d,positive,norm
 def check(self,q):
  c,_,sg,wg,*_=self.geometry(q)
  T=self.data.oMf[self.model.getFrameId('r_grasp_tcp')]
  pe=float(np.linalg.norm(T.translation-self.target[:3,3]));re=float(np.linalg.norm(pin.log3(self.target[:3,:3].T@T.rotation)))
  bearing=np.arctan2(self.cup[1]-q[1],self.cup[0]-q[0]);angle=float(np.arctan2(np.sin(bearing-q[2]+np.pi/2),np.cos(bearing-q[2]+np.pi/2)))
  limits=bool(np.all(q>=self.model.lowerPositionLimit-1e-6) and np.all(q<=self.model.upperPositionLimit+1e-6))
  inside=bool(np.all(np.abs(c[:,:2])+self.radii[:,None]<=3.5))
  good=pe<=1e-4 and re<=1e-3 and sg.min()>=-1e-6 and wg.min()>=-1e-6 and abs(angle)<=np.pi/6 and limits and inside
  return {'accepted':bool(good),'position_error_m':pe,'rotation_error_rad':re,'heading_deg':float(np.rad2deg(angle)),'self_gap_m':float(sg.min()),'world_gap_m':float(wg.min()),'limits_ok':limits,'inside_room':inside}
 def banks(self):
  rng=np.random.default_rng(20261026);banks=[]
  for row in self.reference['rows']:
   start=np.array(row['start_q']);bank=np.tile(start,(5,32,1))
   for attempt in range(5):
    q=bank[attempt];q[:,:2]=self.cup[:2]+np.column_stack([rng.uniform(-.5,.5,32),rng.uniform(-.95,-.5,32)])
    theta=np.arctan2(self.cup[1]-q[:,1],self.cup[0]-q[:,0])+np.pi/2+rng.uniform(-.2,.2,32)
    q[:,2]=np.arctan2(np.sin(theta),np.cos(theta));q[:,3]=rng.uniform(.4,.65,32)
    q[:,4:]+=rng.normal(0,.3,(32,7))
    q[:]=np.clip(q,self.model.lowerPositionLimit+1e-5,self.model.upperPositionLimit-1e-5)
   bank[0,0]=start
   banks.append(bank)
  return np.array(banks)
