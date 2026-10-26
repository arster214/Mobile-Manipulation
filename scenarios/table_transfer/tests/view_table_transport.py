"""Full kinematic task playback: approach, insert, lift, carry, place, withdraw."""
from pathlib import Path
from functools import partial
import sys,json,time,xml.etree.ElementTree as ET
import numpy as np,yaml,trimesh,yourdfpy,viser
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/('tests/artifacts/arm_placement' if '--placement' in sys.argv else ('tests/artifacts/transport_priority' if '--priority' in sys.argv else 'tests/artifacts/transport'));OLD=ROOT/('tests/artifacts/grasp_priority' if '--priority' in sys.argv or '--placement' in sys.argv else 'tests/artifacts/grasp_first')
report=json.loads((OUT/'result.json').read_text());assert report['complete']
scene=json.loads((ROOT/'scene_config.json').read_text());cfg=yaml.safe_load(Path(scene['robot']['curobo_config']).read_text())['kinematics']
pre=np.load(OLD/'r/pregrasp.npz');insert=np.load(OLD/'r/approach.npz');names=pre['joint_names'].tolist()
positions=np.tile(pre['positions'][-1],(len(insert['times']),1))
for j,n in enumerate(insert['joint_names']):positions[:,names.index(n)]=insert['positions'][:,j]
parts=[('到预抓取',dict(pre)),('深入',dict(times=insert['times'],positions=positions,base_poses=np.tile(pre['base_poses'][-1],(len(positions),1)),object_poses=np.tile(report['object_start_pose'],(len(positions),1))))]
parts[0][1]['object_poses']=np.tile(report['object_start_pose'],(len(pre['times']),1))
for stage,label in [('lift','抬升10cm'),('carry','移动搬运'),('place','放到托盘'),('withdraw','释放后抽离')]:parts.append((label,dict(np.load(OUT/f'{stage}.npz'))))
times=[];qs=[];bases=[];objects=[];labels=[];boundaries={};elapsed=0.
for label,p in parts:
 boundaries[label]=elapsed
 idx=[p.get('joint_names',np.array(names)).tolist().index(n) for n in names]
 start=1 if times else 0
 times.extend(elapsed+p['times'][start:]);qs.extend(p['positions'][start:][:,idx]);bases.extend(p['base_poses'][start:]);objects.extend(p['object_poses'][start:]);labels.extend([label]*(len(p['times'])-start));elapsed+=p['times'][-1]
times=np.array(times);qs=np.array(qs);bases=np.array(bases);objects=np.array(objects)
np.savez_compressed(OUT/'full_playback.npz',times=times,positions=qs,joint_names=names,base_poses=bases,object_poses=objects,stages=labels)
urdf=Path(cfg['urdf_path']);robot=yourdfpy.URDF.load(urdf,load_meshes=False,build_scene_graph=True,filename_handler=partial(yourdfpy.filename_handler_magic,dir=urdf.parent))
server=viser.ViserServer(host='127.0.0.1',port=8104 if '--placement' in sys.argv else (8103 if '--priority' in sys.argv else 8101));server.scene.set_up_direction('+z')
if '--placement' in sys.argv:
 server.gui.add_markdown('## 右臂下放与抽离\n下放/抽离固定底盘和升降；完整轨迹 **'+str(round(report['total_execution_duration_s'],2))+'秒**。\n'+ '\n'.join(f'{name}: {duration:.2f}s' for name,duration in report['execution_durations_s'].items()))
server.gui.add_markdown('## 双桌搬运完整运动学回放\n预抓取 → 深入 → 抬升10cm → 搬运 → 放置 → 抽离。\n瓶子搬运时随TCP刚性附着，释放后留在托盘。**夹持/释放为任务事件假设，未验证夹爪接触物理。**')
play=server.gui.add_checkbox('播放',initial_value=False);slider=server.gui.add_slider('时间 (s)',min=0,max=float(times[-1]),step=.025,initial_value=0)
speed=server.gui.add_dropdown('速度',options=['0.5','1','2'],initial_value='1');jump=server.gui.add_dropdown('跳转阶段',options=list(boundaries),initial_value='到预抓取');info=server.gui.add_markdown('')
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
fixed=dict(zip(report['start_state']['joint_names'],report['start_state']['positions']))
def update(_=None):
 i=min(int(np.searchsorted(times,slider.value)),len(times)-1);values=fixed.copy();values.update(zip(names,qs[i]));robot.update_cfg(values)
 base.position=np.r_[bases[i,:2],report['start_state']['base_pose'][2]];base.wxyz=Rotation.from_euler('z',bases[i,2]).as_quat(scalar_first=True)
 for n,h in nodes.items():
  t=robot.get_transform(n);h.position=t[:3,3];h.wxyz=Rotation.from_matrix(t[:3,:3]).as_quat(scalar_first=True)
 bottle.position=objects[i,:3];bottle.wxyz=objects[i,3:];info.content=f'阶段：**{labels[i]}**；总时长 {times[-1]:.2f}s。\n抓取前瓶子静止；抬升至放置阶段附着；抽离阶段已释放。'
slider.on_update(update)
@jump.on_update
def goto(_):play.value=False;slider.value=float(boundaries[jump.value]);update()
@server.on_client_connect
def connected(client):client.camera.position=(4,-4,3.5);client.camera.look_at=(0,1,.65)
update();print('READY transport replay',flush=True)
last=time.monotonic()
while True:
 now=time.monotonic()
 if play.value:
  slider.value=min(float(times[-1]),slider.value+(now-last)*float(speed.value))
  if slider.value>=times[-1]:play.value=False
 last=now;time.sleep(.025)
