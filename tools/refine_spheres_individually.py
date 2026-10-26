"""Per-ball redundancy pruning and subset-constrained refits with independent validation."""
from pathlib import Path
import copy,json,re
import numpy as np,trimesh,yaml
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from build_sphere_comparison import load_mesh
ROOT=Path('/home/fang/robot_ws/collision_models/hybrid')
source=max(ROOT.glob('rm75_hybrid_edited_*.yml'),key=lambda p:p.stat().st_mtime)
config=yaml.safe_load(source.read_text());original=copy.deepcopy(config);k=config['kinematics']
report={'source':str(source),'training_points_per_link':180000,'validation_points_per_link':220000,'policy':'Fit each sphere inside its old sphere, preserving sampled original coverage; prune redundant spheres. Per-link validation loss <=0.10 percentage point, newly exposed depth <=0.5mm. Independent local validation loss <=0.20 percentage point per changed sphere patch. No ignore changes.','links':{}}
def dist(pts,c):return np.concatenate([np.linalg.norm(pts[j:j+5000,None]-c,axis=2) for j in range(0,len(pts),5000)])
for name,old in k['collision_spheres'].items():
 if name not in ['body_base_link','platform_base_link'] and not re.fullmatch('[lr]_(link[1-7]|base_link1)',name):continue
 old=copy.deepcopy(old);mesh=load_mesh(Path(k['urdf_path']),name)
 np.random.seed(3811);pts=np.vstack([trimesh.sample.sample_surface(mesh,180000)[0],mesh.vertices])
 np.random.seed(4811);val=np.vstack([trimesh.sample.sample_surface(mesh,220000)[0],mesh.vertices])
 c=np.array([s['center'] for s in old]);r=np.array([s['radius'] for s in old]);d=dist(pts,c);dv=dist(val,c);origval=(dv-r).min(1)
 cover=d<=r;strong=d<=r-.0005;vcover=dv<=r;vcnt=vcover.sum(1);cnt=cover.sum(1);scnt=strong.sum(1)
 active=np.ones(len(old),bool);newc=c.copy();newr=r.copy();changes=[]
 # Largest overlap/redundancy first, deterministic order.
 order=sorted(range(len(old)),key=lambda i:int(np.sum(cover[:,i]&(cnt==1))))
 for i in order:
  # Covered points for which other balls offer less than 0.5mm protection.
  needed=pts[cover[:,i] & ((scnt-strong[:,i])==0)]
  proposal=None
  if not np.any(cover[:,i] & ((cnt-cover[:,i])==0)):proposal=(newc[i],0.)
  elif len(needed)>=1:
   pp=needed*1000
   if len(pp)>4:
    try:pp=pp[ConvexHull(pp).vertices]
    except Exception:pass
   anchor=c[i]*1000;oldrad=r[i]*1000
   def cons(x):return np.r_[x[3]-np.linalg.norm(pp-x[:3],axis=1),oldrad-x[3]-np.linalg.norm(x[:3]-anchor)]
   result=minimize(lambda x:x[3],np.r_[anchor,oldrad],method='SLSQP',bounds=[(x-25,x+25) for x in anchor]+[(1,oldrad)],constraints={'type':'ineq','fun':cons},options={'maxiter':100,'ftol':1e-7})
   if result.success and cons(result.x).min()>-1e-5:
    cc=result.x[:3]/1000;rr=result.x[3]/1000
    # Add numerical margin only within the original sphere.
    rr=min(rr+.0003,r[i]-np.linalg.norm(cc-c[i]))
    if rr<r[i]-.0003:proposal=(cc,rr)
  if proposal is None and re.fullmatch('[lr]_link[1-7]',name):
   otherids=np.where(active & (np.arange(len(old))!=i))[0]
   og=(dv[:,otherids]-newr[otherids]).min(1)
   patch=dv[:,i]<=r[i]+.003
   for factor in [.98,.99,.995,.998]:
    g=np.minimum(og,dv[:,i]-r[i]*factor)
    loss=100*(np.mean(origval[:220000]<=0)-np.mean(g[:220000]<=0))
    ploss=100*(np.mean(origval[patch]<=0)-np.mean(g[patch]<=0))
    if loss<=.07 and ploss<=.15 and max(0,float(g[origval<=0].max()))<=.00035:
     proposal=(c[i],r[i]*factor);break
  if proposal is None:continue
  cc,rr=proposal
  candidate_dist=np.linalg.norm(val-cc,axis=1) if rr else np.full(len(val),np.inf)
  hit=candidate_dist<=rr;counts=vcnt-vcover[:,i]+hit
  newly=(origval<=0)&(counts==0)
  others=np.where(active & (np.arange(len(old))!=i))[0]
  gap=np.minimum((dv[:,others]-newr[others]).min(1),candidate_dist-rr) if len(others) else candidate_dist-rr
  area_loss=100*(np.mean(origval[:220000]<=0)-np.mean(gap[:220000]<=0))
  patch=dv[:,i]<=r[i]+.003
  patchloss=100*(np.mean(origval[patch]<=0)-np.mean(gap[patch]<=0)) if patch.any() else 0
  newlygap=max(0,float(gap[origval<=0].max()))
  if area_loss>.10 or patchloss>.20 or newlygap>.0005:continue
  dc=np.linalg.norm(pts-cc,axis=1) if rr else np.full(len(pts),np.inf)
  cnt+= (dc<=rr).astype(int)-cover[:,i];scnt+=(dc<=rr-.0005).astype(int)-strong[:,i]
  cover[:,i]=dc<=rr;strong[:,i]=dc<=rr-.0005;d[:,i]=dc
  vcnt=counts;vcover[:,i]=hit;dv[:,i]=candidate_dist;newc[i]=cc;newr[i]=rr;active[i]=rr>0
  changes.append({'index':i,'old_radius_mm':float(r[i]*1000),'new_radius_mm':float(rr*1000),'center_shift_mm':float(np.linalg.norm(cc-c[i])*1000),'action':'remove' if rr==0 else 'refit'})
 balls=[{'center':newc[i].tolist(),'radius':float(newr[i])} for i in range(len(old)) if active[i]]
 gap=(dv[:,active]-newr[active]).min(1)
 record={'count_before':len(old),'count_after':len(balls),'mean_radius_before_mm':float(r.mean()*1000),'mean_radius_after_mm':float(newr[active].mean()*1000),'coverage_before_pct':float(np.mean(origval[:220000]<=0)*100),'coverage_after_pct':float(np.mean(gap[:220000]<=0)*100),'newly_uncovered_max_mm':float(max(0,gap[origval<=0].max())*1000),'changes':changes}
 report['links'][name]=record;k['collision_spheres'][name]=balls
 print(name,json.dumps({a:b for a,b in record.items() if a!='changes'}),'changed',len(changes),flush=True)
(ROOT/'rm75_individual_refined.yml').write_text(yaml.safe_dump(config,sort_keys=False));(ROOT/'individual_refined_metrics.json').write_text(json.dumps(report,indent=2))
print('TOTAL',sum(map(len,k['collision_spheres'].values())),flush=True)
