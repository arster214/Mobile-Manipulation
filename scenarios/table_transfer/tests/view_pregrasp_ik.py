"""Display saved offline IK result; no physics or robot control."""
from pathlib import Path
import json
import time
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
result = json.loads((ROOT/'ik_test/result.json').read_text())
selected = json.loads((ROOT/'ik_test/selected_posture.json').read_text())
scene = json.loads((ROOT/'scene_config.json').read_text())
server = viser.ViserServer(host='127.0.0.1', port=8091)
server.scene.set_up_direction('+z')
server.gui.add_markdown('## Pinocchio + Pink 预抓取 IK\n右臂 + 升降；底盘固定。橙色：瓶子；坐标轴：目标 TCP。\n\n**仅位姿求解结果，尚未验证碰撞或执行轨迹。**')

def mesh(name, m, T=None, color=(170,180,195), opacity=1.):
    if T is not None:
        m.apply_transform(T)
    return server.scene.add_mesh_simple(name, m.vertices, m.faces, color=color, opacity=opacity)

urdf = Path(result['urdf'])
for link in ET.parse(urdf).getroot().findall('link'):
    if link.attrib['name'] not in selected['frame_transforms_world']:
        continue
    T = np.array(selected['frame_transforms_world'][link.attrib['name']])
    for i, visual in enumerate(link.findall('visual')):
        g = visual.find('geometry/mesh')
        if g is None:
            continue
        path = urdf.parent / g.attrib['filename']
        m = trimesh.load(str(path), force='mesh')
        if 'scale' in g.attrib:
            m.apply_scale([float(x) for x in g.attrib['scale'].split()])
        origin = visual.find('origin')
        local = np.eye(4)
        if origin is not None:
            local[:3,3] = [float(x) for x in origin.attrib.get('xyz','0 0 0').split()]
            local[:3,:3] = Rotation.from_euler('xyz',[float(x) for x in origin.attrib.get('rpy','0 0 0').split()]).as_matrix()
        mesh(f'/robot/{link.attrib["name"]}/{i}',m,T@local)

for table in scene['tables']:
    x,y=table['center_xy_m']; lx,ly,h=table['size_m']; th=scene['tabletop_thickness_m']
    mesh('/'+table['name']+'/top',trimesh.creation.box([lx,ly,th],transform=trimesh.transformations.translation_matrix([x,y,h-th/2])),color=(150,104,65))
    for i,(sx,sy) in enumerate([(-1,-1),(-1,1),(1,-1),(1,1)]):
        mesh('/'+table['name']+f'/leg{i}',trimesh.creation.box([.06,.06,h-th],transform=trimesh.transformations.translation_matrix([x+sx*(lx/2-.08),y+sy*(ly/2-.08),(h-th)/2])),color=(70,75,85))
mesh('/floor',trimesh.creation.box([7,7,.02],transform=trimesh.transformations.translation_matrix([0,0,-.01])),color=(210,215,220))
cup=selected['cup']
mesh('/bottle',trimesh.creation.cylinder(radius=cup['radius_m'],height=cup['height_m'],transform=trimesh.transformations.translation_matrix(cup['center_m'])),color=(240,90,25))
T=np.array(result['selected']['target_tcp_world'])
server.scene.add_frame('/target_tcp',position=T[:3,3],wxyz=Rotation.from_matrix(T[:3,:3]).as_quat(scalar_first=True),axes_length=.12,axes_radius=.004)
server.scene.add_line_segments('/approach',points=np.array([[T[:3,3],cup['center_m']]]),colors=(40,190,80),line_width=4)
server.scene.add_label('/target_label','预抓取 TCP',position=T[:3,3]+[0,0,.16])
@server.on_client_connect
def connected(client):
    client.camera.position=(-3.2,-1.8,2.1)
    client.camera.look_at=(-1.1,1.0,.85)
print('PREGRASP_VIEW_READY http://127.0.0.1:8091/',flush=True)
while True:
    time.sleep(1)
