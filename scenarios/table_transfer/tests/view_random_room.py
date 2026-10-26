"""Switch between random room starts and their actually generated trajectories."""
from pathlib import Path
from functools import partial
import json,time,sys,xml.etree.ElementTree as ET
import numpy as np,yaml,trimesh,yourdfpy,viser
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tests/artifacts/random_room_resident'
scene=json.loads((ROOT/'scene_config.json').read_text());cfg=yaml.safe_load(Path(scene['robot']['curobo_config']).read_text())['kinematics']
reference=json.loads((ROOT/'tests/artifacts/grasp_priority/result.json').read_text())
starts=json.loads((OUT/'starts.json').read_text())['base_xy_yaw']
server=viser.ViserServer(host='127.0.0.1',port=8107 if '--maneuver' in sys.argv else (8106 if '--penalty' in sys.argv else 8105));server.scene.set_up_direction('+z')
server.gui.add_markdown('## 搬运倒车软罚与腾挪对照\n保留原抓取、放置，仅替换搬运段。界面中的完整规划耗时属于旧规则基准。失败组只播放已验证部分。' if '--maneuver' in sys.argv or '--penalty' in sys.argv else '## 10组随机起点 · 预热后规划\n固定上肢初始构型，随机底盘位置和朝向。失败组展示起点及已生成的部分轨迹。夹持为运动学附着假设。')
case=server.gui.add_dropdown('起点编号',options=[str(i+1) for i in range(10)],initial_value='1')
play=server.gui.add_checkbox('播放',initial_value=False)
slider=server.gui.add_slider('时间 (s)',min=0,max=1,step=.025,initial_value=0)
speed=server.gui.add_dropdown('速度',options=['0.5','1','2'],initial_value='1')
info=server.gui.add_text('规划结果',initial_value='',multiline=True,disabled=True)
urdf=Path(cfg['urdf_path']);robot=yourdfpy.URDF.load(urdf,load_meshes=False,build_scene_graph=True,filename_handler=partial(yourdfpy.filename_handler_magic,dir=urdf.parent))

for name,box in yaml.safe_load((ROOT/'static_world.yml').read_text())['cuboid'].items():
 mesh=trimesh.creation.box(box['dims']);color=(50,160,80) if 'Tray' in name else ((155,115,75) if 'Table' in name else (210,215,220))
 server.scene.add_mesh_simple('/world/'+name,mesh.vertices,mesh.faces,position=np.array(box['pose'][:3]),wxyz=np.array(box['pose'][3:]),color=color,opacity=.05 if 'Wall' in name else 1.)
nodes={};base=server.scene.add_frame('/robot',show_axes=False)
for link in ET.parse(urdf).findall('link'):
 name=link.get('name');nodes[name]=server.scene.add_frame('/robot/'+name,show_axes=False)
 for i,v in enumerate(link.findall('visual')):
  m=v.find('geometry/mesh')
  if m is None:continue
  mesh=trimesh.load(urdf.parent/m.get('filename'),force='mesh');mesh.apply_scale(np.fromstring(m.get('scale','1 1 1'),sep=' '));o=v.find('origin')
  if o is not None:
   t=trimesh.transformations.euler_matrix(*np.fromstring(o.get('rpy','0 0 0'),sep=' '));t[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ');mesh.apply_transform(t)
  server.scene.add_mesh_simple(f'/robot/{name}/mesh{i}',mesh.vertices,mesh.faces,color=(150,165,180))
mesh=trimesh.creation.cylinder(radius=scene['cup']['radius_m'],height=scene['cup']['height_m'])
bottle=server.scene.add_mesh_simple('/bottle',mesh.vertices,mesh.faces,color=(230,130,35))

for i,start in enumerate(starts):
 server.scene.add_frame(f'/starts/{i}',position=(*start[:2],.03),wxyz=Rotation.from_euler('z',start[2]).as_quat(scalar_first=True),axes_length=.25,axes_radius=.01)
 server.scene.add_label(f'/starts/{i}/label',text=str(i+1),position=(0,0,.35))
old_carry=server.gui.add_checkbox('对照：播放旧搬运轨迹',initial_value=False) if '--penalty' in sys.argv or '--maneuver' in sys.argv else None
data={}
path_handle=server.scene.add_line_segments('/path',points=np.zeros((1,2,3)),colors=(40,160,240),line_width=3)
def load(_=None):
 play.value=False;i=int(case.value)-1;folder=OUT/f'case_{i+1:02}'
 result_path=folder/'replay/result.json';report=json.loads(result_path.read_text()) if result_path.exists() else reference
 state=report['start_state'];names=state['joint_names'];side=report.get('selected_arm') if result_path.exists() else None
 parts=[];object_pose=report['object_pose'];base0=np.array(starts[i]);initial=np.array(state['positions'])
 if side:
  pre=dict(np.load(folder/f'replay/{side}/pregrasp.npz'));parts.append(('预抓取',pre))
  insert=dict(np.load(folder/f'replay/{side}/approach.npz'));insert['base_poses']=np.tile(pre['base_poses'][-1],(len(insert['times']),1));parts.append(('深入',insert))
  for stage in ['lift','carry','place','withdraw']:
   p=folder/f'transport/{stage}.npz'
   if stage=='carry' and old_carry is not None and not old_carry.value:
    p=ROOT/('tests/artifacts/carry_maneuver_scan' if '--maneuver' in sys.argv else 'tests/artifacts/carry_reverse_penalty')/folder.name/'carry.npz'
    if not p.exists():break
   if p.exists():parts.append((stage,dict(np.load(p))))
 times=[0.];qs=[initial.copy()];bases=[base0];objects=[object_pose];labels=['起点'];elapsed=0.;current=initial.copy()
 for label,part in parts:
  for n,t in enumerate(part['times']):
   current=current.copy()
   for j,name in enumerate(part['joint_names']):current[names.index(name)]=part['positions'][n,j]
   times.append(elapsed+float(t));qs.append(current);bases.append(part['base_poses'][n]);objects.append(part.get('object_poses',np.tile(object_pose,(len(part['times']),1)))[n]);labels.append(label)
  elapsed+=float(part['times'][-1])
 timing=json.loads((folder/'timing.json').read_text()) if (folder/'timing.json').exists() else {}
 trials=timing.get('trials',[]);measured=next((t for t in trials if t['phase']=='resident'),None)
 status=f"预热后计算：{measured['wall_s']:.2f}s；完整任务：{'成功' if measured['success'] else '失败'}" if measured else '测量中或未完成'
 if ('--penalty' in sys.argv or '--maneuver' in sys.argv):
  results_path=ROOT/('tests/artifacts/carry_maneuver_scan/results.json' if '--maneuver' in sys.argv else 'tests/artifacts/carry_reverse_penalty/results.json')
  rows=json.loads(results_path.read_text()) if results_path.exists() else []
  row=next((r for r in rows if r['case']==i+1),None)
  status=('当前播放：旧搬运轨迹\n' if old_carry.value else '当前播放：新搬运轨迹\n')+'旧规则完整规划：'+status
  if row:
   if '--maneuver' in sys.argv:
    best=row['best']
    status+=f"\n腾挪扫描：{row['valid_candidates']}/{row['candidates']}条通过"
    if best:status+=f"；选定倒车{best['reverse_distance_m']:.3f}m / {best['reverse_time_s']:.2f}s"
   else:
    status+=f"\n新倒车罚项：搬运重规划{'成功' if row['success'] else '失败'}，{row['warm_planning_s']:.2f}s"
    if row['success']:status+=f"；倒车{row['reverse_distance_m']:.3f}m / {row['reverse_time_s']:.2f}s"
 reasons=''
 if (folder/'transport/result.json').exists():
  transport=json.loads((folder/'transport/result.json').read_text())
  reasons='\n'+'\n'.join(f"{n}: {v['reason']}" for n,v in transport.get('stages',{}).items() if v['validation']!='passed')
 if measured and measured.get('failure'):reasons+='\n'+measured['failure']
 if result_path.exists() and not side:
  reasons='\n'+'\n'.join(f"{arm}: {row.get('reason','')}" for arm,row in report['arms'].items())
 data.update(times=np.array(times),qs=np.array(qs),bases=np.array(bases),objects=np.array(objects),labels=labels,names=names,z=state['base_pose'][2],status=status+reasons)
 slider.max=max(elapsed,.025);slider.value=0.
 jump_carry.disabled='carry' not in labels
 line=np.array(bases);points=np.column_stack([line[:,:2],np.full(len(line),.035)])
 path_handle.points=np.stack([points[:-1],points[1:]],axis=1) if len(points)>1 else np.array([[points[0],points[0]+[0,0,.01]]])
 update()
def update(_=None):
 if not data:return
 i=min(int(np.searchsorted(data['times'],slider.value,side='right')),len(data['times'])-1)
 robot.update_cfg(dict(zip(data['names'],data['qs'][i])))
 base.position=np.r_[data['bases'][i,:2],data['z']];base.wxyz=Rotation.from_euler('z',data['bases'][i,2]).as_quat(scalar_first=True)
 for name,h in nodes.items():
  t=robot.get_transform(name);h.position=t[:3,3];h.wxyz=Rotation.from_matrix(t[:3,:3]).as_quat(scalar_first=True)
 bottle.position=data['objects'][i,:3];bottle.wxyz=data['objects'][i,3:]
 info.value=data['status']+f"\n阶段：{data['labels'][i]}；已生成轨迹 {data['times'][-1]:.2f}s"
pending={'load':False,'update':False}
def request_load(_):pending['load']=True
def request_update(_):pending['update']=True
case.on_update(request_load);slider.on_update(request_update)
if old_carry is not None:old_carry.on_update(request_load)
refresh=server.gui.add_button('刷新测试结果')
refresh.on_click(request_load)
jump_carry=server.gui.add_button('跳到搬运开始')
@jump_carry.on_click
def goto_carry(_):
 play.value=False
 indices=[i for i,label in enumerate(data['labels']) if label=='carry']
 if indices:slider.value=float(data['times'][indices[0]])
@server.on_client_connect
def connect(client):client.camera.position=(6,-6,6);client.camera.look_at=(0,0,.5)
load();print('READY random-room replay',flush=True)
last=time.monotonic()
while True:
 now=time.monotonic()
 if pending['load']:
  pending['load']=False;load()
 if pending['update']:
  pending['update']=False;update()
 if play.value:
  slider.value=min(slider.max,slider.value+(now-last)*float(speed.value))
  if slider.value>=slider.max:play.value=False
 last=now;time.sleep(.025)
