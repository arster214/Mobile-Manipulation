"""Inspect shoulder/platform/column sphere overlaps without changing collision filters."""
from pathlib import Path
from functools import partial
import json,time,xml.etree.ElementTree as ET
import numpy as np,yaml,trimesh,yourdfpy,viser
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tests/artifacts/column_grid'
configs={label:yaml.safe_load(path.read_text())['kinematics'] for label,path in [('修改前464球',ROOT/'tests/artifacts/platform_rear/baseline.yml'),('平台修改后511球',Path('/home/fang/robot_ws/postures/rm75_mobile_manipulation.yml')),('柱体候选557球',OUT/'candidate.yml')]}
s=json.loads((ROOT/'tests/artifacts/transport/result.json').read_text())['start_state'];values=dict(zip(s['joint_names'],s['positions']));initial=values['platform_joint']
urdf=Path(configs['平台修改后511球']['urdf_path']);robot=yourdfpy.URDF.load(urdf,load_meshes=False,build_scene_graph=True,filename_handler=partial(yourdfpy.filename_handler_magic,dir=urdf.parent))
server=viser.ViserServer(host='127.0.0.1',port=8100);server.scene.set_up_direction('+z')
server.gui.add_markdown('## 肩部安装座碰撞近景\n蓝色：左安装座；青色：右安装座；紫色：平台；橙色：柱体。\n**红球及红线**表示选中连杆对的几何重叠；是否参与规划检测见下方状态。\n初始姿态取已保存的抓取终点，并非实机实时状态。')
mode=server.gui.add_dropdown('模型',options=list(configs),initial_value='平台修改后511球')
pairs={'双肩—平台': [('l_base_link1','platform_base_link'),('r_base_link1','platform_base_link')],'左肩—平台':[('l_base_link1','platform_base_link')],'右肩—平台':[('r_base_link1','platform_base_link')],'双肩—柱体':[('l_base_link1','body_base_link'),('r_base_link1','body_base_link')]}
selection=server.gui.add_dropdown('查看碰撞对',options=list(pairs),initial_value='双肩—平台')
slider=server.gui.add_slider('升降关节 (m)',min=0,max=.79,step=.001,initial_value=initial)
only=server.gui.add_checkbox('只显示发生重叠的球',initial_value=True)
visual=server.gui.add_checkbox('显示URDF网格',initial_value=True)
info=server.gui.add_markdown('')
colors={'l_base_link1':(60,120,240),'r_base_link1':(30,190,200),'platform_base_link':(170,100,205),'body_base_link':(230,140,40)}
nodes={};meshes=[];handles={}
for link in ET.parse(urdf).findall('link'):
 name=link.get('name')
 if name not in colors:continue
 nodes[name]=server.scene.add_frame('/robot/'+name,show_axes=False)
 for i,item in enumerate(link.findall('visual')):
  m=item.find('geometry/mesh')
  if m is None:continue
  mesh=trimesh.load(urdf.parent/m.get('filename'),force='mesh');mesh.apply_scale(np.fromstring(m.get('scale','1 1 1'),sep=' '));o=item.find('origin')
  if o is not None:
   t=trimesh.transformations.euler_matrix(*np.fromstring(o.get('rpy','0 0 0'),sep=' '));t[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ');mesh.apply_transform(t)
  meshes.append(server.scene.add_mesh_simple(f'/robot/{name}/mesh{i}',mesh.vertices,mesh.faces,color=colors[name],opacity=.3))
 for label,k in configs.items():
  for i,b in enumerate(k['collision_spheres'][name]):
   mesh=trimesh.creation.icosphere(subdivisions=2,radius=b['radius'])
   handles[label,name,i]=server.scene.add_mesh_simple(f'/robot/{name}/{label}/{i}',mesh.vertices,mesh.faces,position=np.array(b['center']),color=colors[name],opacity=.45)
lines=None

def update(_=None):
 global lines
 values['platform_joint']=slider.value;robot.update_cfg(values);poses={n:robot.get_transform(n) for n in colors}
 for n,node in nodes.items():node.position=poses[n][:3,3];node.wxyz=Rotation.from_matrix(poses[n][:3,:3]).as_quat(scalar_first=True)
 for h in meshes:h.visible=visual.value
 k=configs[mode.value];active=set();hit=set();segments=[];rows=[];records=[]
 for a,b in pairs[selection.value]:
  active.update([a,b]);sa=k['collision_spheres'][a];sb=k['collision_spheres'][b]
  ca=np.array([x['center'] for x in sa])@poses[a][:3,:3].T+poses[a][:3,3];cb=np.array([x['center'] for x in sb])@poses[b][:3,:3].T+poses[b][:3,3]
  gaps=np.linalg.norm(ca[:,None]-cb[None],axis=2)-np.array([x['radius'] for x in sa])[:,None]-np.array([x['radius'] for x in sb])
  ij=np.argwhere(gaps<0);ignored=b in k['self_collision_ignore'].get(a,[]) or a in k['self_collision_ignore'].get(b,[])
  for i,j in ij:hit.update([(a,int(i)),(b,int(j))]);segments.append([ca[i],cb[j]])
  depth=max(0,float(-gaps.min()*1000));rows.append(f'**{a} ↔ {b}**\n检测：{"已忽略" if ignored else "启用"}；重叠球对：{len(ij)}；最大重叠：{depth:.2f} mm。')
  records.append({'a':a,'b':b,'ignored':ignored,'overlapping_sphere_pairs':len(ij),'max_overlap_mm':depth})
 for (label,n,i),h in handles.items():
  h.visible=label==mode.value and n in active and (not only.value or (n,i) in hit);h.color=(240,55,55) if (n,i) in hit else colors[n]
 if lines is not None:lines.remove()
 if segments:lines=server.scene.add_line_segments('/overlap_lines',points=np.array(segments),colors=(255,30,30),line_width=2)
 else:lines=None
 info.content='\n\n'.join(rows)
 (OUT/'shoulder_contacts_current.json').write_text(json.dumps({'model':mode.value,'platform_joint':slider.value,'pairs':records},indent=2))
for control in (mode,selection,slider,only,visual):control.on_update(update)
update()
@server.on_client_connect
def connected(client):
 client.camera.position=(.7,-1.1,1.15);client.camera.look_at=(0,-.20,.74)
print('READY http://127.0.0.1:8100/',flush=True)
while True:time.sleep(1)
