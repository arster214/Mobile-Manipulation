"""Inspect the configured tray and both tables, without running robot motion."""
from pathlib import Path
import json,time
import numpy as np,yaml,trimesh,viser
ROOT=Path(__file__).resolve().parents[1]
scene=json.loads((ROOT/'scene_config.json').read_text());tray=scene['tray']
server=viser.ViserServer(host='127.0.0.1',port=8097)
server.scene.set_up_direction('+z')
server.gui.add_markdown('## 目标桌托盘位置\n绿色托盘外尺寸 **25 × 18 cm**。\n25 cm 沿桌子长边；18 cm 沿短边。\n托盘前沿距桌子靠近机器人一侧的边缘 **3 cm**。\n此页面只检查摆放位置，不执行机器人运动。')
near=server.gui.add_button('托盘近景');overview=server.gui.add_button('查看两张桌子')
for name,box in yaml.safe_load((ROOT/'static_world.yml').read_text())['cuboid'].items():
 mesh=trimesh.creation.box(box['dims'])
 color=(45,150,85) if name.startswith('TargetTray') else ((90,145,160) if 'TargetTable' in name else ((170,120,75) if 'SourceTable' in name else (220,225,230)))
 server.scene.add_mesh_simple('/world/'+name,mesh.vertices,mesh.faces,position=np.array(box['pose'][:3]),wxyz=np.array(box['pose'][3:]),color=color,opacity=.07 if 'Wall' in name else 1.)
x,y=tray['center_xy_m'];z=.785
server.scene.add_label('/tray_label','目标托盘 25 × 18 cm',position=np.array([x,y+.10,z+.10]))
server.scene.add_label('/target_table','目标桌（第二张桌子）',position=np.array([1.2,1.65,.9]))
server.scene.add_label('/source_table','源桌',position=np.array([-1.2,1.5,.9]))
server.scene.add_line_segments('/dimension_x',points=np.array([[[x-.125,y-.12,z],[x+.125,y-.12,z]]]),colors=(30,90,180),line_width=3)
server.scene.add_label('/width','25 cm',position=np.array([x,y-.14,z]))
server.scene.add_line_segments('/dimension_y',points=np.array([[[x+.16,y-.09,z],[x+.16,y+.09,z]]]),colors=(30,90,180),line_width=3)
server.scene.add_label('/depth','18 cm',position=np.array([x+.20,y,z]))
# Show a transparent target bottle envelope, not a claimed completed placement.
mesh=trimesh.creation.cylinder(radius=scene['cup']['radius_m'],height=scene['cup']['height_m'])
target=server.scene.add_mesh_simple('/placement_preview',mesh.vertices,mesh.faces,position=np.array(scene['placement_center_m']),color=(245,130,30),opacity=.3)
show=server.gui.add_checkbox('显示瓶子放置目标（半透明）',initial_value=True)
@show.on_update
def toggle(_):target.visible=show.value

def camera(client,wide=False):
 client.camera.position=(4,-3,4) if wide else (1.85,.30,1.65)
 client.camera.look_at=(0,1.4,.65) if wide else (x,y,.76)
@server.on_client_connect
def connected(client):camera(client)
@near.on_click
def closeup(_):
 for client in server.get_clients().values():camera(client)
@overview.on_click
def wide(_):
 for client in server.get_clients().values():camera(client,True)
print('TRAY_READY http://127.0.0.1:8097/',flush=True)
while True:time.sleep(1)
