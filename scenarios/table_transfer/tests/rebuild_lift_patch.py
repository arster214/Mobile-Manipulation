"""Refit column side spheres against the shoulder sweep, preserving sampled coverage."""
from pathlib import Path
import json, copy
import numpy as np
import yaml
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'tests/artifacts/lift_spheres'
config=yaml.safe_load(Path('/home/fang/robot_ws/postures/rm75_mobile_manipulation.yml').read_text())
k=config['kinematics']; balls=k['collision_spheres']['body_base_link']
c=np.array([b['center'] for b in balls]); r=np.array([b['radius'] for b in balls]); oldc=c.copy(); oldr=r.copy()
poses=json.loads((OUT/'transforms.json').read_text()); obstacle_c=[];obstacle_r=[]
lift=json.loads((OUT/'audit_before.json').read_text())['initial_lift_q']
for value,transforms in poses.items():
 if not lift-.1-1e-8 <= float(value) <= lift+1e-8: continue
 bt=np.array(transforms['body_base_link'])
 for name in ('l_base_link1','r_base_link1'):
  t=np.linalg.inv(bt)@np.array(transforms[name]); ss=k['collision_spheres'][name]
  obstacle_c.extend(np.array([b['center'] for b in ss])@t[:3,:3].T+t[:3,3]);obstacle_r.extend([b['radius'] for b in ss])
oc=np.array(obstacle_c);orr=np.array(obstacle_r)
data=np.load(OUT/'body_samples.npz');points=data['points'];before=data['before']
def gap(cc,rr):
 return np.concatenate([(np.linalg.norm(p[:,None]-cc[None],axis=2)-rr).min(1) for p in np.array_split(points,50)])
changed=[]
for i in range(len(r)):
 allowed=(np.linalg.norm(c[i]-oc,axis=1)-orr-.0001).min()
 if r[i]<=allowed:continue
 # Move the center inward while fitting the surface patch uniquely covered by this ball.
 dist=np.linalg.norm(points-c[i],axis=1)
 local=np.where(dist<=r[i])[0]
 others=np.delete(np.arange(len(r)),i)
 othergap=np.concatenate([(np.linalg.norm(p[:,None]-c[others][None],axis=2)-r[others]).min(1) for p in np.array_split(points[local],10)])
 patch=points[local[othergap>0]]
 if len(patch)>1500:patch=patch[np.linspace(0,len(patch)-1,1500,dtype=int)]
 def constraint(x):
  free=np.linalg.norm(x[:3]-oc,axis=1)-orr-x[3]-.0001
  cover=x[3]-np.linalg.norm(patch-x[:3],axis=1) if len(patch) else np.array([])
  return np.r_[free,cover]
 result=minimize(lambda x:np.sum((x[:3]-oldc[i])**2)+(x[3]-oldr[i])**2,np.r_[c[i],max(.001,allowed)],method='SLSQP',bounds=[(v-.05,v+.05) for v in c[i]]+[(.001,float(oldr[i]))],constraints=[{'type':'ineq','fun':constraint}], options={'maxiter':150,'ftol':1e-12})
 if result.success and constraint(result.x).min()>-1e-7:c[i]=result.x[:3];r[i]=result.x[3]
 else:r[i]=max(.001,allowed)
 changed.append(i)
 print(i,result.success,'r',r[i],flush=True)
after=gap(c,r)
# Patch only sampled surface previously covered. Small local balls must clear the full shoulder sweep.
added=[]
for iteration in range(12):
 missing=np.where((before<=0)&(after>1e-5))[0]
 if not len(missing):break
 p=points[missing[np.argmax(after[missing])]]
 radius=min(.025,float((np.linalg.norm(p-oc,axis=1)-orr-.0001).min()))
 if radius<=.0001:break
 c=np.vstack([c,p]);r=np.r_[r,radius];added.append({'center':p.tolist(),'radius':radius})
 after=np.minimum(after,np.linalg.norm(points-p,axis=1)-radius)
proposal=copy.deepcopy(config)
proposal['kinematics']['collision_spheres']['body_base_link']=[{'center':p.tolist(),'radius':float(v)} for p,v in zip(c,r)]
metrics={'changed_indices':changed,'added_count':len(added),'sphere_count':464+len(added),'coverage_before_pct':float(np.mean(before<=0)*100),'coverage_after_pct':float(np.mean(after<=0)*100),'newly_uncovered_max_mm':float(max(0,after[before<=0].max())*1000),'min_shoulder_gap_mm':float(min((np.linalg.norm(p-oc,axis=1)-orr-v).min() for p,v in zip(c,r))*1000),'changes':[{'index':i,'old_center':oldc[i].tolist(),'new_center':c[i].tolist(),'old_radius':float(oldr[i]),'new_radius':float(r[i])} for i in changed]}
(OUT/'refined.yml').write_text(yaml.safe_dump(proposal,sort_keys=False));(OUT/'refined_metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics),flush=True)
