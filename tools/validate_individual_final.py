from pathlib import Path
import json,numpy as np,trimesh,yaml
from build_sphere_comparison import load_mesh
R=Path('/home/fang/robot_ws/collision_models/hybrid');d=json.loads((R/'individual_refined_metrics.json').read_text());old=yaml.safe_load(Path(d['source']).read_text());new=yaml.safe_load((R/'rm75_individual_refined.yml').read_text())
def gap(p,s):
 c=np.array([x['center'] for x in s]);r=np.array([x['radius'] for x in s]);return np.concatenate([(np.linalg.norm(p[i:i+5000,None]-c,axis=2)-r).min(1) for i in range(0,len(p),5000)])
report={}
for n,v in d['links'].items():
 m=load_mesh(Path(new['kinematics']['urdf_path']),n);np.random.seed(9811);p=np.vstack([trimesh.sample.sample_surface(m,300000)[0],m.vertices]);a=old['kinematics']['collision_spheres'][n];b=new['kinematics']['collision_spheres'][n];ga=gap(p,a);gb=gap(p,b)
 row={'before_coverage_pct':float(np.mean(ga[:300000]<=0)*100),'after_coverage_pct':float(np.mean(gb[:300000]<=0)*100),'newly_uncovered_max_mm':float(max(0,gb[ga<=0].max())*1000),'patches':[]}
 for ch in v['changes']:
  sphere=a[ch['index']];mask=np.linalg.norm(p-np.array(sphere['center']),axis=1)<=sphere['radius']+.003
  row['patches'].append({'index':ch['index'],'before_pct':float(np.mean(ga[mask]<=0)*100),'after_pct':float(np.mean(gb[mask]<=0)*100),'new_max_gap_mm':float(max(0,gb[mask].max())*1000)})
 # Explicit arm 3-to-2 deletion test; best possible deletion with other balls fixed.
 if len(a)==3 and '_link' in n:
  row['best_two_ball_coverage_pct']=max(float(np.mean(gap(p[:300000],[s for i,s in enumerate(b) if i!=j])<=0)*100) for j in range(len(b)))
 report[n]=row
 print(n,json.dumps({k:v for k,v in row.items() if k!='patches'}),flush=True)
(R/'individual_final_coverage.json').write_text(json.dumps(report,indent=2))
