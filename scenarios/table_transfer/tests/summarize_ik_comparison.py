from pathlib import Path
import json,csv,statistics,html
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'ik_test/comparison';d=json.loads((OUT/'result.json').read_text())
summary={}
for b in ('pink','curobo'):
 rows=[r for r in d['rows'] if r['backend']==b];times=[r['wall_ms'] for r in rows];valid=[r for r in rows if r['accepted']]
 summary[b]={'accepted':len(valid),'runs':len(rows),'mean_ms':float(np.mean(times)),'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'min_ms':min(times),'max_ms':max(times),'mean_position_error_mm':float(np.mean([r['audit']['position_error_m']*1000 for r in valid])),'max_position_error_mm':max(r['audit']['position_error_m']*1000 for r in valid),'median_self_gap_mm':float(np.median([r['audit']['self_gap_m']*1000 for r in valid])),'max_abs_heading_deg':max(abs(r['audit']['heading_deg']) for r in valid)}
 if b=='curobo':summary[b]['mean_synchronized_gpu_call_ms']=float(np.mean([r['gpu_solve_ms'] for r in rows]))
paired=[]
for i in range(1,21):
 row={'index':i}
 for b in ('pink','curobo'):
  group=[r for r in d['rows'] if r['backend']==b and r['index']==i]
  row[b+'_median_ms']=statistics.median(r['wall_ms'] for r in group);row[b+'_accepted']=sum(r['accepted'] for r in group)
 paired.append(row)
summary['gpu_faster_cases']=sum(r['curobo_median_ms']<r['pink_median_ms'] for r in paired)
summary['mean_wall_speed_ratio']=summary['pink']['mean_ms']/summary['curobo']['mean_ms']
summary['cpu']='AMD Ryzen 7 9700X';summary['gpu']='NVIDIA GeForce RTX 5080';summary['cpu_policy']='single process; BLAS/OpenMP threads limited to 1; no CPU seed parallelism'
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (OUT/'per_start.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
with (OUT/'all_trials.csv').open('w') as f:
 fields=['backend','repeat','index','accepted','wall_ms','seeds_tried'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(d['rows'])
rows=''.join(f"<tr><td>{r['index']}</td><td>{r['pink_median_ms']:.2f}</td><td>{r['curobo_median_ms']:.2f}</td><td>{'cuRobo' if r['curobo_median_ms']<r['pink_median_ms'] else 'Pink'}</td></tr>" for r in paired)
text=f'''<!doctype html><meta charset="utf-8"><title>全身 IK 同条件对比</title><style>body{{font:17px/1.65 system-ui;max-width:1000px;margin:40px auto;padding:0 25px;color:#203043}}h1{{font-size:30px}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:9px;border-bottom:1px solid #ccd4de;text-align:left}}.note{{background:#eef3f7;padding:18px;border-radius:10px}}img{{width:100%}}</style>
<h1>Pinocchio + Pink 与 cuRobo V2：全身 IK 对比</h1><p>20 个原始起点，每个重复 3 次。两边均为 60/60 通过；cuRobo 完整流程平均快 {summary['mean_wall_speed_ratio']:.2f} 倍。这是当前实现与该目标的测试，不是库的普遍性能排名。</p>
<table><tr><th>指标</th><th>Pink（CPU）</th><th>cuRobo（GPU）</th></tr>'''
for label,key in [('平均完整耗时 / ms','mean_ms'),('中位耗时 / ms','median_ms'),('P95 耗时 / ms','p95_ms'),('最大耗时 / ms','max_ms'),('平均位置误差 / mm','mean_position_error_mm'),('自碰撞最小间隙的中位数 / mm','median_self_gap_mm')]:text+=f"<tr><td>{label}</td><td>{summary['pink'][key]:.6g}</td><td>{summary['curobo'][key]:.6g}</td></tr>"
text+='''</table><div class="note">统一要求：11 自由度；464 个球及相同忽略对；同一瓶子、桌子和墙；0.1 mm 位置误差、0.001 rad 姿态误差；瓶子在底盘前方 ±30°；自碰撞及环境间隙不小于零（1 微米数值容差）；关节限位及房间边界。固定地面支撑不参加球障碍检查。</div>
<h2>时间口径与算法差异</h2><p>两边最多 160 个相同初值、每组 1 秒软预算；cuRobo 每轮 GPU 并行 32 个，最多 5 轮，Pink 单进程逐个尝试，每初值最多 100 次 QP 迭代并提前终止停滞。达到合格解即结束。三次重复用于测量时间波动，不代表 60 个独立场景。</p>
<p>Pink 每步对最近 64 个自碰撞球对、32 个球与环境对加入线性化距离约束；cuRobo 使用自身碰撞优化。两边都使用相同的最终朝向筛选和 Pinocchio 全量独立验收。两边初值池相同，但内部算法和并行方式不相同；这是一项应用方案对比。</p>
<p>完整时间含候选处理、重试和独立验收，cuRobo 还包括本地工作进程通信。模型加载、预热不计入；GPU 同步求解调用平均 23.17 ms，不能把它单独与 Pink 的完整耗时直接相比。硬件：Ryzen 7 9700X / RTX 5080。Pink 未做多核初值并行优化。</p>
<p>成功解未强制额外安全裕量，部分解贴近自碰撞边界；没有验证差速路径、中间碰撞、动力学或抓取执行。当前结论不代表某个库无法进一步调优。</p>
<img src="timing.png" alt="各起点的三次耗时中位数"><h2>每组耗时中位数</h2><table><tr><th>起点编号</th><th>Pink / ms</th><th>cuRobo / ms</th><th>更快</th></tr>'''+rows+'</table>'
(OUT/'report.html').write_text(text)
print(json.dumps(summary,indent=2))
