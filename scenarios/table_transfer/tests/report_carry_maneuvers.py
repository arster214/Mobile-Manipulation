"""Report short-reverse feasibility and re-score using the calibrated soft time."""
from pathlib import Path
import json,statistics,html
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tests/artifacts/carry_maneuver_scan'
settings=json.loads((ROOT/'planning/config.json').read_text());soft=settings['carry_reverse_soft_time_s']
rows=json.loads((OUT/'results.json').read_text());records=[]
for row in rows:
 folder=OUT/f"case_{row['case']:02}";candidates=json.loads((folder/'candidates.json').read_text())
 for r in candidates:
  n=r['reverse_bursts'];assert n<=2
  bursts=[] if n==0 else [r['max_reverse_burst_s']]
  if n==2:bursts.append(r['reverse_time_s']-r['max_reverse_burst_s'])
  r['calibrated_cost']=r['cost']-.5*r.get('time_penalty_s2',r['reverse_time_s']**2)+.5*sum(max(t-soft,0.)**2 for t in bursts)
  records.append(dict(case=row['case'],**r))
 feasible=[r for r in candidates if r['success']]
 if feasible:
  best=min(feasible,key=lambda r:r['calibrated_cost']);keys=['before_m','after_m','full_reverse','route','upper_timing']
  assert all(best[k]==row['best'][k] for k in keys), (row['case'],best,row['best'])
  row['best']['calibrated_cost']=best['calibrated_cost']
  row['soft_reverse_time_s']=soft
(OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
summary=[]
for d in [0.,.05,.1,.15,.2,.3,.4]:
 group=[r for r in records if r['before_m']==d and r['after_m']==d and not r['full_reverse']]
 summary.append({'each_adjustment_m':d,'candidates':len(group),'passed':sum(r['success'] for r in group),'median_single_reverse_s':statistics.median(r['max_reverse_burst_s'] for r in group)})
report={'carry_cases':len(rows),'tested_candidates':len(records),'skipped_case':9,'skip_reason':'No complete grasp path in the original random-start trial; no carry start exists.', 'scan_policy':'Two reverse adjustments (departure and arrival) at independently selected 0/5/10/15/20/30/40 cm, three routes and three body timing choices, plus nine full-reverse comparisons per case.', 'equal_adjustment_comparison':summary,'soft_threshold_s':soft,'selected_reverse_distances_m':[r['best']['reverse_distance_m'] for r in rows],'limitation':'All available carry endpoints admit forward-only paths. This supports a provisional short-adjustment soft knee, not an optimal required reverse allowance or a collision safety threshold. The frozen endpoints do not fix original grasp or withdrawal failures.'}
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
table='<table><tr><th>每次短退</th><th>单次连续倒车时长中位数</th><th>通过候选/总数</th></tr>'+''.join(f"<tr><td>{r['each_adjustment_m']*100:.0f} cm</td><td>{r['median_single_reverse_s']:.3f} s</td><td>{r['passed']} / {r['candidates']}</td></tr>" for r in summary)+'</table>'
(OUT/'report.html').write_text('<meta charset="utf-8"><title>短距离腾挪扫描</title><style>body{font:17px sans-serif;max-width:1000px;margin:40px auto;line-height:1.7}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:right}table{border-collapse:collapse;width:100%}</style><h1>短距离腾挪：软阈值实验</h1><p>'+str(len(rows))+'组搬运起终点，共'+str(len(records))+'条候选。每组比较出发前短退、到位前短退、两次短退及全程倒车。沿用机器人和附着物碰撞、差速底盘、关节及既有奇异性检查。</p>'+table+'<p>暂取单次连续倒车 <b>1.1 秒</b>为软分界；当前加速度下约对应20 cm短退。代价为 2 × 累计倒车距离 + 0.5 × Σ max(单次连续倒车时长 − 1.1, 0)²。短退仍付距离代价；长退不被硬禁止。</p><p>所有可用搬运起终点都找到更便宜的不倒车方案。因此该阈值是当前配置的试用值，不是必要倒车量的最优解，也不是安全边界。第9组没有抓取连接轨迹，未伪造搬运起点；第3组原抽离失败仍未修复。</p><p><a href="http://127.0.0.1:8107/">查看重新选择的轨迹</a> · <a href="../random_room_resident/report.html">查看旧规则10组完整耗时</a></p>')
print(json.dumps(report,indent=2))
