"""Compare the existing collision model and the local 10 cm lift candidate."""
from pathlib import Path
from functools import partial
import json,time,xml.etree.ElementTree as ET
import numpy as np,yaml,trimesh,yourdfpy,viser
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tests/artifacts/lift_spheres'
old=yaml.safe_load(Path('/home/fang/robot_ws/postures/rm75_mobile_manipulation.yml').read_text())['kinematics'];new=yaml.safe_load((OUT/'radius_only.yml').read_text())['kinematics']
s=json.loads((ROOT/'tests/artifacts/transport/result.json').read_text())['start_state'];values=dict(zip(s['joint_names'],s['positions']));initial=values['platform_joint']
urdf=Path(old['urdf_path']);robot=yourdfpy.URDF.load(urdf,load_meshes=False,build_scene_graph=True,filename_handler=partial(yourdfpy.filename_handler_magic,dir=urdf.parent))
server=viser.ViserServer(host='127.0.0.1',port=8098);server.scene.set_up_direction('+z')
server.gui.add_markdown('## 升降柱局部微调对照\n候选版只缩小6个柱体球，仍为 **464球**。\n正式配置未替换。此候选仅验证当前抓取姿态向上10cm，不能据此认为全行程无碰撞。\n柱体采样覆盖 **99.442% → 99.289%**；新增表面漏覆盖最深约 **4.8mm**。')
mode=server.gui.add_dropdown('模型',options=['原模型','局部缩球候选'],initial_value='局部缩球候选')
slider=server.gui.add_slider('向上抬升 (cm)',min=0,max=10,step=.1,initial_value=10)
show=server.gui.add_checkbox('显示其余柱体和平台球',initial_value=False)
nodes={};sets={'原模型':[],'局部缩球候选':[]};rest=[]
for link in ET.parse(urdf).findall('link'):
 name=link.get('name');nodes[name]=server.scene.add_frame('/robot/'+name,show_axes=False)
 for i,item in enumerate(link.findall('visual')):
  m=item.find('geometry/mesh')
  if m is None:continue
  mesh=trimesh.load(urdf.parent/m.get('filename'),force='mesh');mesh.apply_scale(np.fromstring(m.get('scale','1 1 1'),sep=' '));o=item.find('origin')
  if o is not None:
   t=trimesh.transformations.euler_matrix(*np.fromstring(o.get('rpy','0 0 0'),sep=' '));t[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ');mesh.apply_transform(t)
  server.scene.add_mesh_simple(f'/robot/{name}/visual{i}',mesh.vertices,mesh.faces,color=(150,160,170),opacity=.65)
for label,k in [('原模型',old),('局部缩球候选',new)]:
 for name in ('body_base_link','platform_base_link','l_base_link1','r_base_link1'):
  for i,b in enumerate(k['collision_spheres'][name]):
   changed=name=='body_base_link' and i in (85,86,93,94,144,145);root='base_link1' in name
   mesh=trimesh.creation.icosphere(subdivisions=2,radius=b['radius']);color=(230,110,35) if changed else ((70,140,230) if root else (80,180,130))
   h=server.scene.add_mesh_simple(f'/robot/{name}/{label}/{i}',mesh.vertices,mesh.faces,position=np.array(b['center']),color=color,opacity=.35)
   sets[label].append((h,changed or root))
def update(_=None):
 values['platform_joint']=initial-slider.value/100;robot.update_cfg(values)
 for name,node in nodes.items():
  t=robot.get_transform(name);node.position=t[:3,3];node.wxyz=Rotation.from_matrix(t[:3,:3]).as_quat(scalar_first=True)
 for label,items in sets.items():
  for handle,focus in items:handle.visible=label==mode.value and (focus or show.value)
slider.on_update(update);mode.on_update(update);show.on_update(update);update()
@server.on_client_connect
def connected(client):
 client.camera.position=(1.6,-2,1.7);client.camera.look_at=(0,-.05,1.0)
print('READY http://127.0.0.1:8098/',flush=True)
while True:time.sleep(1)
