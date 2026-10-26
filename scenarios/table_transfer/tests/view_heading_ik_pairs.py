"""Interactive paired IK endpoint viewer; deliberately no trajectory playback."""
from pathlib import Path
import json, time, math, xml.etree.ElementTree as ET
import numpy as np
import trimesh, viser, yaml
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'ik_test/gpu_heading20'
r=json.loads((OUT/'result.json').read_text());frames=json.loads((OUT/'pair_frames.json').read_text());scene=json.loads((ROOT/'scene_config.json').read_text());k=yaml.safe_load((OUT/'source_robot.yml').read_text())['kinematics']
server=viser.ViserServer(host='127.0.0.1',port=8093);server.scene.set_up_direction('+z')
server.gui.add_markdown('## GPU 全身 IK：朝向 ±30°\n蓝色半透明＝起点；绿色＝通过检测的终点；红色＝失败候选。\n**显示姿态对，不表示两者之间存在无碰撞差速路径。**')
options=[f"{x['index']:02d} | {'通过' if x['accepted'] else '失败'} | {x['wall_ms']:.2f} ms" for x in r['rows']]
select=server.gui.add_dropdown('选择起终点对',options=options)
show_start=server.gui.add_checkbox('显示起点',initial_value=True);show_end=server.gui.add_checkbox('显示终点 / 失败候选',initial_value=True)
show_spheres=server.gui.add_checkbox('显示最新碰撞球',initial_value=False)
show_map=server.gui.add_checkbox('显示全部起点编号',initial_value=True)
info=server.gui.add_markdown('')
handles={'start':{},'end':{}}; sphere_nodes=[]; end_meshes=[]
urdf=ET.parse(r['urdf']).getroot()
for kind in ('start','end'):
 for link in urdf.findall('link'):
  name=link.attrib['name']
  if name not in frames[0][kind]:continue
  handle=server.scene.add_frame(f'/{kind}/{name}',show_axes=False);handles[kind][name]=handle
  for i,v in enumerate(link.findall('visual')):
   g=v.find('geometry/mesh')
   if g is None:continue
   m=trimesh.load(g.attrib['filename'],force='mesh')
   if 'scale' in g.attrib:m.apply_scale([float(z) for z in g.attrib['scale'].split()])
   origin=v.find('origin');T=np.eye(4)
   if origin is not None:
    T[:3,3]=[float(z) for z in origin.attrib.get('xyz','0 0 0').split()]
    T[:3,:3]=Rotation.from_euler('xyz',[float(z) for z in origin.attrib.get('rpy','0 0 0').split()]).as_matrix()
   m.apply_transform(T)
   h=server.scene.add_mesh_simple(f'/{kind}/{name}/visual{i}',m.vertices,m.faces,color=(60,140,245) if kind=='start' else (65,185,115),opacity=.3 if kind=='start' else 1.)
   if kind=='end':end_meshes.append(h)
  balls=[]
  for s in k['collision_spheres'].get(name,[]):
   if s['radius']<=0:continue
   ball=trimesh.creation.icosphere(subdivisions=1,radius=s['radius']);ball.apply_translation(s['center']);balls.append(ball)
  if balls:
   m=trimesh.util.concatenate(balls);h=server.scene.add_mesh_simple(f'/{kind}/{name}/spheres',m.vertices,m.faces,color=(45,120,255) if kind=='start' else (255,165,40),opacity=.25,visible=False);sphere_nodes.append(h)

def addmesh(name,m,color):return server.scene.add_mesh_simple(name,m.vertices,m.faces,color=color)
world=yaml.safe_load((OUT/'world.yml').read_text())['cuboid']
for name,b in world.items():
 if name=='Bottle':continue
 m=trimesh.creation.box(b['dims'],transform=trimesh.transformations.translation_matrix(b['pose'][:3]))
 h=addmesh('/environment/'+name,m,(165,120,75) if 'Table' in name else (200,205,215))
 if 'Wall' in name:h.opacity=.12
addmesh('/environment/floor',trimesh.creation.box([7,7,.02],transform=trimesh.transformations.translation_matrix([0,0,-.01])),(220,225,230))
cup=r['cup'];addmesh('/environment/bottle',trimesh.creation.cylinder(radius=cup['radius_m'],height=cup['height_m'],transform=trimesh.transformations.translation_matrix(cup['center_m'])),(245,100,30))
T=np.array(r['target_tcp_world']);server.scene.add_frame('/target_tcp',position=T[:3,3],wxyz=Rotation.from_matrix(T[:3,:3]).as_quat(scalar_first=True),axes_length=.1,axes_radius=.003)
server.scene.add_line_segments('/approach',points=np.array([[T[:3,3],cup['center_m']]]),colors=(240,140,25),line_width=4)
markers=[]
for row in r['rows']:
 markers.append(server.scene.add_label(f'/map/{row["index"]}',str(row['index']),position=(*row['start_q'][:2],.05)))

def update(_=None):
 i=options.index(select.value);row=r['rows'][i]
 with server.atomic():
  for kind in ('start','end'):
   for name,h in handles[kind].items():
    T=np.array(frames[i][kind][name]);h.position=T[:3,3];h.wxyz=Rotation.from_matrix(T[:3,:3]).as_quat(scalar_first=True);h.visible=show_start.value if kind=='start' else show_end.value
  for h in end_meshes:h.color=(65,185,115) if row['accepted'] else (230,65,65)
  for h in sphere_nodes:h.visible=show_spheres.value
  for h in markers:h.visible=show_map.value
  a=row['start_q'];b=row['end_q'];audit=row['end_audit']
  info.content=f"""### 第 {row['index']} 对：{'通过检测' if row['accepted'] else '失败候选，不可用'}
| | 起点 | 终点 |
|---|---:|---:|
| 底盘 X (m) | {a[0]:.3f} | {b[0]:.3f} |
| 底盘 Y (m) | {a[1]:.3f} | {b[1]:.3f} |
| URDF 底盘偏航 (°) | {math.degrees(a[2]):.1f} | {math.degrees(b[2]):.1f} |
| 升降 q (m，增大向下) | {a[3]:.3f} | {b[3]:.3f} |

瓶子相对底盘正前方：**{row['heading_error_deg']:.2f}°**（限制 ±30°）

尝试轮数：**{row['attempts']} / 5**

GPU 求解及筛选总耗时：**{row['wall_ms']:.2f} ms**（含同步与重试）

终点最小自碰撞间隙：**{audit['min_self_gap_m']*1000:.3f} mm**

终点最小环境间隙：**{audit['min_world_gap_m']*1000:.2f} mm**

独立 FK 位置误差：**{row['independent_position_error_m']*1000:.3f} mm**

只验证起终点；未验证中间路径。绿色解也未附加硬性安全裕量。
"""
for h in (select,show_start,show_end,show_spheres,show_map):h.on_update(update)
@server.on_client_connect
def connected(client):
 client.camera.position=(6,-8,7);client.camera.look_at=(0,0,.5)
update();print('PAIR_VIEW_READY http://127.0.0.1:8093/',flush=True)
while True:time.sleep(1)
