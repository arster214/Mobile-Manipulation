"""Summarize measured random-start trials and independently audit completed tasks."""
from pathlib import Path
import json, html, subprocess, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tests/artifacts/random_room_resident'
rows=[]
for folder in sorted(OUT.glob('case_*')):
 if not (folder/'timing.json').exists():continue
 timing=json.loads((folder/'timing.json').read_text());trials=timing['trials']
 if len(trials)<2:continue
 warm,run=trials[0],trials[-1];grasp=json.loads((folder/'replay/result.json').read_text())
 side=grasp['selected_arm'];transport=json.loads((folder/'transport/result.json').read_text()) if (folder/'transport/result.json').exists() else {}
 stage_failure='; '.join(f"{name}: {stage.get('reason', '')}" for name,stage in transport.get('stages',{}).items() if stage['validation']!='passed')
 reason=run.get('failure') or stage_failure or ('; '.join(f"{a}: {r['reason']}" for a,r in grasp['arms'].items()) if side is None else '')
 row={'case':int(folder.name[-2:]),'success':run['success'],'arm':side,'warmup_s':warm['wall_s'],'grasp_s':run['grasp_s'],'transport_s':run['transport_s'],'planning_s':run['wall_s'],'trajectory_s':transport.get('total_execution_duration_s'),'failure':reason,'zero_initialization':run['initialization']['models_created']==run['initialization']['ik_created']==0,'same_trajectory_as_warmup':run['same_trajectory_as_warmup']}
 if run['success']:
  audit=subprocess.run([sys.executable,str(ROOT/'tests/audit_table_transport.py'),'--placement','--directory',str(folder/'transport'),'--arm',side],capture_output=True,text=True)
  row['independent_audit_passed']=audit.returncode==0
  if audit.returncode:row['audit_error']=audit.stderr
 rows.append(row)
(OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
def cell(x):return html.escape(f'{x:.2f}' if isinstance(x,float) else str(x if x is not None else '—'))
headers=['case','success','arm','grasp_s','transport_s','planning_s','trajectory_s','warmup_s']
titles=['编号','完整成功','抓取臂','抓取规划','后续搬运','计算总时长','运动时长','首轮预热']
table='<table><tr>'+''.join('<th>'+h+'</th>' for h in titles)+'</tr>'+''.join('<tr>'+''.join('<td>'+cell(r.get(h))+'</td>' for h in headers)+'</tr>' for r in rows)+'</table>'
detail=''.join(f"<p>组 {r['case']}: {html.escape(r['failure'])}</p>" for r in rows if r['failure'])
(OUT/'report.html').write_text('<meta charset="utf-8"><title>随机起点常驻规划测试</title><style>body{font:16px sans-serif;max-width:1150px;margin:40px auto;background:#f5f7fa;color:#182638}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:right}table{background:white;border-collapse:collapse;width:100%}p{line-height:1.6;overflow-wrap:anywhere}</style><h1>10组随机起点 · 预热后规划</h1><p>每组同起点完整首轮预热，再重新搜索轨迹。计时包含验证和文件读写，不包含首轮初始化与预热；不是首次看到新场景的在线耗时。固定双臂/升降初始构型，只随机底盘位置和朝向。时间单位：秒。</p><p><a href="http://127.0.0.1:8105/">打开可切换轨迹回放</a></p>'+table+detail)
print(json.dumps(rows,indent=2))
