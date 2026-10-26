"""Compare the existing collision model and the local 10 cm lift candidate."""
from pathlib import Path
from functools import partial
import json,time,xml.etree.ElementTree as ET
import numpy as np,yaml,trimesh,yourdfpy,viser
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tests/artifacts/column_grid'
old=yaml.safe_load(Path('/home/fang/robot_ws/postures/rm75_mobile_manipulation.yml').read_text())['kinematics'];new=yaml.safe_load((OUT/'candidate.yml').read_text())['kinematics']
s=json.loads((ROOT/'tests/artifacts/transport/result.json').read_text())['start_state'];values=dict(zip(s['joint_names'],s['positions']));initial=values['platform_joint']
urdf=Path(old['urdf_path']);robot=yourdfpy.URDF.load(urdf,load_meshes=False,build_scene_graph=True,filename_handler=partial(yourdfpy.filename_handler_magic,dir=urdf.parent))
server=viser.ViserServer(host='127.0.0.1',port=8099);server.scene.set_up_direction('+z')
server.gui.add_markdown('## 升降柱三排两列重建\n横截面X方向3排、Y方向2列，每层6球；沿高度16层，共96个柱体球（减少1/3）。\n整机464 → **510球**。新柱体球半径约 **52–58mm**。\n柱段表面采样覆盖 **98.295% → 99.983%**。\n仍有自碰撞，正式配置未替换。可关闭网格，查看完整球阵。')
mode=server.gui.add_dropdown('模型',options=['原模型','三排两列重建'],initial_value='三排两列重建')
slider=server.gui.add_slider('升降关节位置 (m)',min=0,max=.79,step=.001,initial_value=initial)
show=server.gui.add_checkbox('绿色：原柱体上下端/底座过渡球',initial_value=False)
platform=server.gui.add_checkbox('紫色：原升降平台球',initial_value=False)
meshes=[]
visuals=server.gui.add_checkbox('显示机器人网格',initial_value=True)
nodes={};sets={'原模型':[],'三排两列重建':[]};rest=[]
for link in ET.parse(urdf).findall('link'):
 name=link.get('name');nodes[name]=server.scene.add_frame('/robot/'+name,show_axes=False)
 for i,item in enumerate(link.findall('visual')):
  m=item.find('geometry/mesh')
  if m is None:continue
  mesh=trimesh.load(urdf.parent/m.get('filename'),force='mesh');mesh.apply_scale(np.fromstring(m.get('scale','1 1 1'),sep=' '));o=item.find('origin')
  if o is not None:
   t=trimesh.transformations.euler_matrix(*np.fromstring(o.get('rpy','0 0 0'),sep=' '));t[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ');mesh.apply_transform(t)
  meshes.append(server.scene.add_mesh_simple(f'/robot/{name}/visual{i}',mesh.vertices,mesh.faces,color=(150,160,170),opacity=.4))
for label,k in [('原模型',old),('三排两列重建',new)]:
 for name in ('body_base_link','platform_base_link','l_base_link1','r_base_link1'):
  for i,b in enumerate(k['collision_spheres'][name]):
   changed=name=='body_base_link' and .57<=b['center'][2]<=1.30;root='base_link1' in name
   mesh=trimesh.creation.icosphere(subdivisions=2,radius=b['radius']);color=(230,110,35) if changed else ((70,140,230) if root else ((170,100,205) if name=='platform_base_link' else (80,180,130)))
   h=server.scene.add_mesh_simple(f'/robot/{name}/{label}/{i}',mesh.vertices,mesh.faces,position=np.array(b['center']),color=color,opacity=.35)
   sets[label].append((h,'focus' if changed or root else ('platform' if name=='platform_base_link' else 'body')))
def update(_=None):
 values['platform_joint']=slider.value;robot.update_cfg(values)
 for name,node in nodes.items():
  t=robot.get_transform(name);node.position=t[:3,3];node.wxyz=Rotation.from_matrix(t[:3,:3]).as_quat(scalar_first=True)
 for mesh in meshes:mesh.visible=visuals.value
 for label,items in sets.items():
  for handle,group in items:handle.visible=label==mode.value and (group=='focus' or (platform.value if group=='platform' else show.value))
slider.on_update(update);mode.on_update(update);show.on_update(update);platform.on_update(update);visuals.on_update(update);update()
@server.on_client_connect
def connected(client):
 client.camera.position=(1,-1.2,1.45);client.camera.look_at=(0,-.05,.9)
print('READY http://127.0.0.1:8099/',flush=True)
while True:time.sleep(1)
