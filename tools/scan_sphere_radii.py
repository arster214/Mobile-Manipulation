"""Fixed-center radius sensitivity using independent area-weighted mesh samples."""
from pathlib import Path
import json,copy,csv,html,re
import numpy as np,trimesh,yaml
from build_sphere_comparison import load_mesh
ROOT=Path('/home/fang/robot_ws/collision_models/hybrid')
OUT=Path('/home/fang/Documents/Codex/2026-10-24/new-chat/outputs/radius_scan');OUT.mkdir(parents=True,exist_ok=True)
source=max(ROOT.glob('rm75_hybrid_edited_*.yml'),key=lambda p:p.stat().st_mtime)
config=yaml.safe_load(source.read_text());k=config['kinematics'];candidate=copy.deepcopy(config)
scales=[1.,.99,.98,.97,.95,.925,.90,.85,.80,.75,.70]
report={'source':str(source),'samples_per_link_per_split':100000,'training_seed':1561,'validation_seed':2561,'criterion':'coverage loss <=1 percentage point per link; cautious candidate additionally max uncovered gap <=2mm on samples AND mesh vertices; baseline coverage <99% left unchanged','links':{}}

def stats(mesh,balls,seed,vertices=False):
 np.random.seed(seed);pts=trimesh.sample.sample_surface(mesh,100000)[0]
 if vertices:pts=np.vstack([pts,mesh.vertices])
 c=np.array([s['center'] for s in balls]);r=np.array([s['radius'] for s in balls]);ratios=[];dist=[]
 for off in range(0,len(pts),10000):
  d=np.linalg.norm(pts[off:off+10000,None]-c[None],axis=2);ratios.append((d/r).min(1));dist.append(d)
 return np.concatenate(ratios),dist,r

def gap_at(dist,r,scale):return max(0,max(float((d-scale*r).min(1).max()) for d in dist))*1000
for n,balls in k['collision_spheres'].items():
 mesh=load_mesh(Path(k['urdf_path']),n)
 train,td,r=stats(mesh,balls,1561)
 val,vd,_=stats(mesh,balls,2561,True);surface=val[:100000]
 base=float(np.mean(surface<=1)*100)
 thresholds={}
 for loss in [.1,.5,1.]:
  target=max(0,float(np.mean(train<=1))-loss/100)
  scale=min(1,float(np.quantile(train,target,method='higher')))
  thresholds[str(loss)]={'scale':scale,'validation_coverage_pct':float(np.mean(surface<=scale)*100),'validation_gap_mm':gap_at(vd,r,scale)}
 # Independently check a 0.1% grid candidate, keeping a gap cap and high baseline.
 trial=max(.5,min(1,np.ceil(thresholds['0.5']['scale']*1000)/1000)) if base>=99 else 1.
 if base>=99:
  while trial<1 and (base-np.mean(surface<=trial)*100>1 or gap_at(vd,r,trial)>2):trial=round(trial+.001,3)
 newballs=[{'center':s['center'],'radius':float(s['radius']*trial)} for s in balls]
 candidate['kinematics']['collision_spheres'][n]=newballs
 rows=[{'scale':s,'coverage_pct':float(np.mean(surface<=s)*100),'max_gap_mm':gap_at(vd,r,s)} for s in scales]
 report['links'][n]={'area_m2':float(mesh.area),'sphere_count':len(balls),'baseline_coverage_pct':base,'thresholds_pp':thresholds,'scan':rows,'candidate_scale':trial,'candidate_coverage_pct':float(np.mean(surface<=trial)*100),'candidate_max_gap_mm':gap_at(vd,r,trial),'old_radii_mm':(r*1000).tolist(),'candidate_radii_mm':(r*trial*1000).tolist()}
 print(n,'base',round(base,2),'1pp',round(thresholds['1.0']['scale'],3),'candidate',trial,'gap',round(report['links'][n]['candidate_max_gap_mm'],2),flush=True)
(ROOT/'rm75_radius_scan_candidate.yml').write_text(yaml.safe_dump(candidate,sort_keys=False));(OUT/'rm75_radius_scan_candidate.yml').write_text(yaml.safe_dump(candidate,sort_keys=False))
(OUT/'radius_scan.json').write_text(json.dumps(report,indent=2))
with (OUT/'radius_scan.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['link','scale','coverage_pct','max_gap_mm'])
 for n,v in report['links'].items():
  for row in v['scan']:w.writerow([n,row['scale'],row['coverage_pct'],row['max_gap_mm']])
def table(headers,rows):return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
body='<h1>碰撞球半径与表面覆盖率扫描</h1><p>固定球心，同一 link 的所有球按相同比例缩放。每个 link 使用 10 万训练表面点和另外 10 万验证点，均按三角面面积采样。最大遗漏深度同时计入所有网格顶点。表面覆盖仅是几何指标，不等于自碰撞或规划安全验收。</p><p>以覆盖率相对当前值下降不超过 1 个百分点为主要比较标准；另给出更保守候选：训练覆盖下降不超过 0.5 个百分点起步，独立验证下降不超过 1 个百分点，最大遗漏不超过 2 mm。原始覆盖不足 99% 的 link 不缩小。候选不是已批准替换，正式配置和当前页面不变。</p>'
body+='<h2>全身统一缩放的影响</h2>'
aggregate=[]
for i,s in enumerate(scales):
 vals=list(report['links'].values());area=sum(x['area_m2'] for x in vals)
 cov=sum(x['area_m2']*x['scan'][i]['coverage_pct'] for x in vals)/area
 arm=[x for n,x in report['links'].items() if re.fullmatch('[lr]_link[1-7]',n)]
 aggregate.append([s,round(cov,3),round(min(x['scan'][i]['coverage_pct'] for x in arm),3),round(max(x['scan'][i]['max_gap_mm'] for x in arm),3)])
body+=table(['半径比例','全身面积加权覆盖 %','最差手臂 link 覆盖 %','手臂最大遗漏 mm'],aggregate)
body+='<h2>逐连杆阈值</h2><p>1pp 比例由训练集估计，右侧显示独立验证结果。半径列表与配置中的球编号顺序一致。不能把同一 link 内各球独立缩小到任意值而仍引用这些覆盖结果。</p>'
body+=table(['link','原覆盖 %','1pp 阈值比例','阈值验证覆盖 %','阈值遗漏 mm','保守候选比例','候选半径 mm','候选覆盖 %','候选遗漏 mm'],[[n,round(v['baseline_coverage_pct'],3),round(v['thresholds_pp']['1.0']['scale'],4),round(v['thresholds_pp']['1.0']['validation_coverage_pct'],3),round(v['thresholds_pp']['1.0']['validation_gap_mm'],3),v['candidate_scale'],[round(x,2) for x in v['candidate_radii_mm']],round(v['candidate_coverage_pct'],3),round(v['candidate_max_gap_mm'],3)] for n,v in report['links'].items()])
body+='<h2>复现</h2><pre>/home/fang/isaacsim/python.sh /home/fang/robot_ws/tools/scan_sphere_radii.py</pre><p>源文件：'+html.escape(str(source))+'</p><p>radius_scan.json 包含 0.1、0.5、1 个百分点阈值、完整半径和扫描结果；CSV 包含所有比例的覆盖率与遗漏深度。候选 YAML 仅供后续碰撞验证，不覆盖现用配置。</p>'
(OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Radius scan</title><style>body{font-family:system-ui;max-width:1400px;margin:30px auto;line-height:1.6}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:7px}th{background:#eaf2f7}</style>'+body)
print('AGGREGATE',aggregate,flush=True)
