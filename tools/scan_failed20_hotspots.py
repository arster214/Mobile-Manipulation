from pathlib import Path
import json,collections,numpy as np,yaml,trimesh
from build_sphere_comparison import load_mesh
R=Path('/home/fang/robot_ws');O=R/'collision_models/hybrid/failed20_audit';d=json.loads((O/'report.json').read_text());k=yaml.safe_load(Path(d['source']).read_text())['kinematics'];changes=collections.defaultdict(dict)
# Diagnostic only: assign each measured deficit to the body-side sphere.
for row in d['rows']:
 for c in row['checks']:
  a,b=c['pair']
  if a not in ['body_base_link','platform_base_link','r_base_link1']:continue
  for i,j,gap in c['overlapping_sphere_pairs']:changes[a][i]=max(changes[a].get(i,0),-gap/1000+.001)
report={}
for name,delta in changes.items():
 balls=k['collision_spheres'][name];cent=np.array([s['center'] for s in balls]);rad=np.array([s['radius'] for s in balls]);new=rad.copy()
 for i,v in delta.items():new[i]-=v
 m=load_mesh(Path(k['urdf_path']),name);np.random.seed(20261025);pts=trimesh.sample.sample_surface(m,120000)[0];before=[];after=[];local=[]
 for ps in np.array_split(pts,30):
  dist=np.linalg.norm(ps[:,None]-cent,axis=2);before.extend((dist-rad).min(1));after.extend((dist-new).min(1));local.extend(((dist-rad)[:,list(delta)]<=0).any(1))
 before=np.array(before);after=np.array(after);local=np.array(local)
 report[name]={'sphere_changes':[{'index':i,'old_radius_mm':rad[i]*1000,'diagnostic_radius_mm':new[i]*1000,'shrink_mm':v*1000} for i,v in delta.items()],'whole_surface_coverage_before_pct':float(np.mean(before<=0)*100),'whole_surface_coverage_after_pct':float(np.mean(after<=0)*100),'hotspot_surface_coverage_after_pct':float(np.mean(after[local]<=0)*100),'max_new_gap_on_previously_covered_surface_mm':float(max(0,after[before<=0].max())*1000),'note':'Counterfactual shrink to clear these 20 samples by 1mm, NOT approved model; uniform area surface sampling only'}
 print(name,report[name],flush=True)
(O/'shrink_diagnostic.json').write_text(json.dumps(report,indent=2))
