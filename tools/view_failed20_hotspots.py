"""Interactive sphere overlap witnesses for the 20 supplied IK terminal poses."""
from pathlib import Path
from functools import partial
import time,json,threading
import numpy as np,yaml,yourdfpy,viser
from viser.extras import ViserUrdf
R=Path('/home/fang/robot_ws');D=R/'collision_models/hybrid';data=json.loads(Path('/home/fang/.codex/attachments/cdafce39-2e4b-4352-b574-c35f49a662e6/已粘贴的文本.txt').read_text());config=yaml.safe_load(Path(data['robot_config']).read_text());k=config['kinematics'];candidate=yaml.safe_load((D/'symmetric_adjustment/candidate_2.yml').read_text())['kinematics']['collision_spheres'];p=Path(k['urdf_path']);urdf=yourdfpy.URDF.load(p,load_meshes=True,filename_handler=partial(yourdfpy.filename_handler_magic,dir=p.parent))
server=viser.ViserServer(host='127.0.0.1',port=8090,label='身体—手臂碰撞热点 · 20组终态对比');server.scene.set_up_direction('+z');robot=ViserUrdf(server,urdf,root_node_name='/robot',mesh_color_override=(.55,.60,.65,.65));server.scene.add_grid('/floor',width=3,height=3,plane='xy',position=(0,0,-.15))
pairs=[('body_base_link','l_base_link1'),('body_base_link','r_base_link1'),('platform_base_link','r_link2'),('platform_base_link','r_link3'),('r_base_link1','r_link2')];titles=['① 躯干—左肩座','② 躯干—右肩座','③ 平台—右上臂 r_link2','④ 平台—右肘根部 r_link3','⑤ 右肩座—右上臂'];base=dict(zip(k['cspace']['joint_names'],k['cspace']['default_joint_position']));base.update(k['lock_joints']);handles=[];lock=threading.RLock()
server.gui.add_markdown('**红球：身体/肩座一侧；橙球：手臂一侧。黄色点与引线：球体重叠位置。**\n\n显示的是 IK 终态，不是已确认的初始姿态。网格参考检查未发现这些连杆表面相交。球编号从 0 开始。')
pose=server.gui.add_dropdown('失败样本',options=[str(i) for i in range(1,21)],initial_value='20');version=server.gui.add_dropdown('碰撞球版本',options=['原始模型','对称调整候选（强度2）'],initial_value='原始模型');focus=server.gui.add_dropdown('重点区域',options=['全部热点']+titles,initial_value='全部热点');mesh=server.gui.add_checkbox('显示机器人网格',initial_value=True);details=server.gui.add_markdown('');server.gui.add_markdown('约95%覆盖作为候选评估目标；完整报告保留局部遗漏数据。候选版本未覆盖正式配置。')
def update(_=None):
 with lock,server.atomic():
  for h in handles:h.remove()
  handles.clear();q=dict(base);v=data['results'][int(pose.value)-1]['joint_positions'];q['platform_joint']=v[4];q.update(dict(zip([f'r_joint{i}' for i in range(1,8)],v[5:])));robot.update_cfg(np.array([q.get(n,0) for n in robot.get_actuated_joint_names()]));robot.show_visual=mesh.value
  bs=k['collision_spheres'] if version.value=='原始模型' else candidate;rows=[]
  for j,(a,b) in enumerate(pairs):
   ta=urdf.get_transform(a);tb=urdf.get_transform(b);ca=np.array([s['center'] for s in bs[a]])@ta[:3,:3].T+ta[:3,3];cb=np.array([s['center'] for s in bs[b]])@tb[:3,:3].T+tb[:3,3];ra=np.array([s['radius'] for s in bs[a]]);rb=np.array([s['radius'] for s in bs[b]]);g=np.linalg.norm(ca[:,None]-cb[None,:],axis=2)-ra[:,None]-rb[None,:];i,t=np.unravel_index(g.argmin(),g.shape);rows.append(f'| {titles[j]} | {g[i,t]*1000:.2f} | {i} / {t} |')
   if focus.value!='全部热点' and focus.value!=titles[j]:continue
   # In candidate view also show nearest pairs, with green clearance markers.
   hits=list(zip(*np.where(g<0))) or [(i,t)]
   for n,cent,radii,ids,color in [(a,ca,ra,sorted(set(x for x,y in hits)),(255,65,65)),(b,cb,rb,sorted(set(y for x,y in hits)),(255,170,35))]:
    for idx in ids:handles.append(server.scene.add_icosphere(f'/hot/{j}/{n}/{idx}',position=cent[idx],radius=float(radii[idx]),color=color,opacity=.35,subdivisions=2))
   direction=(cb[t]-ca[i]);direction/=np.linalg.norm(direction);point=(ca[i]+direction*ra[i]+cb[t]-direction*rb[t])/2
   label=np.array([.42 if j==0 else -.48,-.32,[1.20,1.13,.98,.68,.83][j]]);color=(255,230,20) if g[i,t]<0 else (30,230,120)
   handles.append(server.scene.add_icosphere(f'/hot/{j}/point',radius=.008,position=point,color=color))
   handles.append(server.scene.add_line_segments(f'/hot/{j}/leader',points=np.array([[point,label]]),colors=np.array([[color,color]],dtype=np.uint8),line_width=2))
   handles.append(server.scene.add_label(f'/hot/{j}/label',text=f'{titles[j]}: '+(f'重叠 {-g[i,t]*1000:.1f} mm' if g[i,t]<0 else f'间隙 {g[i,t]*1000:.1f} mm'),position=label))
  details.content='**间隙为负表示球体重叠**\n\n| 区域 | 间隙 mm | 双方最深球编号 |\n|---|---:|---|\n'+'\n'.join(rows)
for c in [pose,version,focus,mesh]:c.on_update(update)
@server.on_client_connect
def camera(client):
 client.camera.position=(-1.2,-1.8,1.25);client.camera.look_at=(0,-.12,.8);client.camera.up_direction=(0,0,1)
update();print('READY http://127.0.0.1:8090/',flush=True)
while True:time.sleep(.5)
