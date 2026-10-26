from pathlib import Path
import json,csv,html,re,hashlib,yaml
root=Path('/home/fang/robot_ws/collision_models/hybrid');out=Path('/home/fang/Documents/Codex/2026-10-24/new-chat/outputs/collision_audit')
d=json.loads((root/'three_refined_collision_audit.json').read_text());m=json.loads((root/'three_refined_metrics.json').read_text())
assert all(d['visual_collision_geometry_match'].values())
assert not d['adjacent_not_ignored']
new=yaml.safe_load((root/'rm75_hybrid_three_refined.yml').read_text());old=yaml.safe_load(Path(m['source']).read_text())
assert new['kinematics']['self_collision_ignore']==old['kinematics']['self_collision_ignore']
for n,s in new['kinematics']['collision_spheres'].items():
 if n not in m['links']:assert s==old['kinematics']['collision_spheres'][n]
base=Path('/home/fang/isaacsim_rm75/config/rm75_dual.yml')
assert hashlib.sha256(base.read_bytes()).hexdigest()==json.loads((root/'evaluation.json').read_text())['source_sha256']
for filename in ['rm75_hybrid_three_refined.yml','three_refined_metrics.json','three_refined_collision_audit.json']:(out/filename).write_bytes((root/filename).read_bytes())
with (out/'pair_results.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['link1','link2','curobo_ignore','srdf_reason','urdf_adjacent','mesh_hits','sphere_tp','sphere_fp','sphere_fn','sphere_tn'])
 for p in d['pairs']:w.writerow([*p['pair'],p['ignored_curobo'],p['srdf_reason'],p['adjacent_urdf'],p['mesh_hits'],*[p['versions']['three_refined'][x] for x in ['tp','fp','fn','tn']]])
with (out/'ignore_differences.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['link1','link2','status']);w.writerows([*p,'ignored_by_old_SRDF_but_checked_by_curobo'] for p in d['srdf_only_ignored'])
def table(headers,rows):return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
body='''<h1>RM75 三球版半径优化与自碰撞对照</h1><p>结论：三球布局保留，平均半径由 50.60 mm 降至 43.36 mm（约 14.3%）。当前全身 378 球。修改的 14 个连杆各自独立抽样 60000 个表面点，均未发现遗漏。优化限制球心每轴移动不超过 20 mm，并约束 AABB 外凸不比原版更大。</p><p>不能判定球模型与 URDF 的碰撞结果完全一致：有限样本中未见漏检，但仍有误报，尤其来自平台和躯干的规则球。</p><h2>测试范围</h2><p>使用当前 cuRobo URDF 的 collision 网格（已核对所有 38 个 link 的 visual 与 collision 几何和原点相同），通过 Python-FCL 的 BVH 三角面相交作为参考。共 203 个姿态：初始姿态、4 个升降位置、14 个关节各 5 点扫描、128 个关节限位内随机姿态。随机姿态升降为 0.5 m，种子 20261025。</p><p>38 个 link 共 703 对；cuRobo 排除 367 对后，实际比较 336 对 × 203 姿态 = 68208 次。球体以 link 局部坐标经 URDF 正运动学变换到世界系，按球心距与半径和检测，当前相关缓冲均为零。此为 CPU 几何对照，没有执行 cuRobo CUDA 自碰撞内核或规划器。</p><h2>有效检查对的结果</h2><p>计数单位为“连杆对 × 姿态”，不是不同连杆对数量。误报：球相交、三角网格未相交；漏检：反之。</p>'''
body+=table(['版本','同时检测到碰撞','球误报','球漏检','双方无碰撞'],[[v,*[d['confusion_after_curobo_ignore'][v][x] for x in ['tp','fp','fn','tn']]] for v in ['original313','three_before','three_refined']])
body+='<p>仅手臂 link1～link7 之间：优化前误报 39 次，优化后 14 次；参考碰撞 21 次全部检出。</p><h2>初始姿态仍有的误报</h2>'
body+=table(['link1','link2','球重叠深度 mm'],[[*e['pair'],round(-e['sphere_gap_mm'],2)] for e in d['examples'] if e['pose']=='default'])
body+='<p>这些误报说明当前全身候选不能直接作为已验收的规划模型。应先改进躯干/平台局部球形包络，而不是直接把这些连杆对全部加入忽略表。</p><h2>与旧 SRDF 关系核对</h2><p>旧 MoveIt SRDF 在这 38 个碰撞 link 内排除 471 对；cuRobo 排除 367 对，全部属于 SRDF 排除对，少排除 104 对，即 cuRobo 检查范围更大。两者并不完全相同。所有有碰撞网格的 URDF 相邻 link 对已排除；无碰撞几何的 tool0 不计入。测试中没有发现旧 SRDF 标为 Never 的连杆对发生三角面相交，但有限采样不能证明它们永不相撞。SRDF 来自旧 MoveIt 配置迁移，并非本次重新生成。</p><p>没有改动任何碰撞排除关系，也没有替换正式规划配置。</p><h2>逐连杆变化</h2>'
body+=table(['link','原半径 mm','新半径 mm','原 AABB 外凸下界 mm','新下界 mm'],[[n,v['old_radii_mm'],v['new_radii_mm'],round(v['before']['aabb_excess_mm'],2),round(v['after']['aabb_excess_mm'],2)] for n,v in m['links'].items()])
body+='<h2>边界</h2><p>AABB 外凸只是实际网格外凸的下界。部分原始网格不封闭，因此 FCL 三角面相交不是实体包含检查；完整包裹但表面不相交的情况不由此参考覆盖。无采样遗漏不是连续曲面的覆盖证明，有限姿态零漏检也不是全关节空间证明。尚未测试环境碰撞、运动连续碰撞或执行安全。</p><h2>复现与文件</h2><pre>/home/fang/isaacsim/python.sh /home/fang/robot_ws/tools/audit_sphere_collisions.py\n/home/fang/isaacsim/python.sh /home/fang/robot_ws/tools/edit_hybrid_spheres.py --port 8085 --focus-link l_link4 --config /home/fang/robot_ws/collision_models/hybrid/rm75_hybrid_three_refined.yml</pre><p>pair_results.csv：全部连杆对统计。ignore_differences.csv：104 对排除差异。three_refined_collision_audit.json：测试条件、网格封闭性和可复现异常姿态关节值。</p>'
(out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>RM75 collision audit</title><style>body{font-family:system-ui;max-width:1100px;margin:40px auto;line-height:1.65;color:#223}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ddd;padding:8px;text-align:left}th{background:#eaf1f5}pre{background:#eee;padding:16px;overflow:auto}</style>'+body)
print('Exported',out)
