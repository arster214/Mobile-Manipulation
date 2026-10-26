"""Replay saved mobile trajectory and reproduced partial approach IK samples."""
from pathlib import Path
import json, time, sys
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tests/artifacts/baseline'


def export_frames():
    import pinocchio as pin
    motion = np.load(OUT / 'trajectory.npz')
    partial = np.load(OUT / 'approach_partial.npz')
    scene = json.loads((ROOT / 'scene_config.json').read_text())
    cfg = yaml.safe_load(Path(scene['robot']['curobo_config']).read_text())['kinematics']
    initial = json.loads(Path(scene['robot']['initial_posture']).read_text())
    values = dict(zip(initial['joint_names'], initial['joint_positions_rad']))
    values.update(cfg['lock_joints'])
    model = pin.buildModelFromUrdf(str(OUT / 'mobile.urdf'))
    data = model.createData()
    ids = [i for i, frame in enumerate(model.frames) if frame.type == pin.FrameType.BODY]
    frames = []
    def record():
        q = pin.neutral(model)
        for side in ('l', 'r'):
            values[f'{side}_Joint_finger2'] = values[f'{side}_Joint_finger1']
        for name, value in values.items():
            if model.existJointName(name):
                q[model.joints[model.getJointId(name)].idx_q] = value
        pin.framesForwardKinematics(model, data, q)
        frames.append(np.array([data.oMf[i].homogeneous for i in ids]))
    for body, base in zip(motion['positions'], motion['base_poses']):
        values.update(zip(motion['joint_names'], body))
        values.update(zip(['base_x', 'base_y', 'base_yaw'], base))
        record()
    for arm in partial['positions'][1:]:
        values.update(zip(partial['joint_names'], arm))
        record()
    np.savez_compressed(OUT / 'replay_frames.npz', frames=frames,
                        link_names=[model.frames[i].name for i in ids], mobile_count=len(motion['times']))
    print('Exported', len(frames), 'frames')


def show():
    import xml.etree.ElementTree as ET
    import trimesh
    import viser
    saved = np.load(OUT / 'replay_frames.npz')
    partial = np.load(OUT / 'approach_partial.npz')
    motion = np.load(OUT / 'trajectory.npz')
    frames, names = saved['frames'], saved['link_names'].tolist()
    count = int(saved['mobile_count'])
    scene = json.loads((ROOT / 'scene_config.json').read_text())
    cfg = yaml.safe_load(Path(scene['robot']['curobo_config']).read_text())['kinematics']
    server = viser.ViserServer(host='127.0.0.1', port=8094)
    server.scene.set_up_direction('+z')
    server.gui.add_markdown('## 失败深入段回放\n前段：保存的预抓取轨迹。后段：同种子复现的深入 IK 点。\n**深入点逐点展示，未通过完整轨迹验证；红色坐标轴仅是失败目标，没有对应有效机器人姿态。**')
    play = server.gui.add_checkbox('播放', initial_value=False)
    slider = server.gui.add_slider('帧', min=0, max=len(frames)-1, step=1, initial_value=count-1)
    speed = server.gui.add_dropdown('播放速度', options=['0.25×','0.5×','1×'],initial_value='0.5×')
    to_start = server.gui.add_button('回到全身运动起点')
    to_approach = server.gui.add_button('回到预抓取 / 深入起点')
    to_last = server.gui.add_button('查看最后一个已求解姿态')
    spheres = server.gui.add_checkbox('显示碰撞球', initial_value=False)
    info = server.gui.add_markdown('')
    nodes, balls = {}, []
    robot = ET.parse(OUT / 'mobile.urdf').getroot()
    for link in robot.findall('link'):
        name = link.get('name')
        if name not in names:
            continue
        nodes[name] = server.scene.add_frame('/robot/'+name,show_axes=False)
        for i, visual in enumerate(link.findall('visual')):
            mesh = visual.find('geometry/mesh')
            if mesh is None:
                continue
            m = trimesh.load(mesh.get('filename'), force='mesh')
            m.apply_scale([float(x) for x in mesh.get('scale','1 1 1').split()])
            origin = visual.find('origin')
            if origin is not None:
                transform = np.eye(4)
                transform[:3,3] = [float(x) for x in origin.get('xyz','0 0 0').split()]
                transform[:3,:3] = Rotation.from_euler('xyz',[float(x) for x in origin.get('rpy','0 0 0').split()]).as_matrix()
                m.apply_transform(transform)
            server.scene.add_mesh_simple(f'/robot/{name}/visual{i}',m.vertices,m.faces,color=(120,155,190))
        meshes=[]
        for sphere in cfg['collision_spheres'].get(name,[]):
            if sphere['radius'] > 0:
                m=trimesh.creation.icosphere(subdivisions=1,radius=sphere['radius'])
                m.apply_translation(sphere['center']);meshes.append(m)
        if meshes:
            m=trimesh.util.concatenate(meshes)
            balls.append(server.scene.add_mesh_simple(f'/robot/{name}/spheres',m.vertices,m.faces,color=(255,160,45),opacity=.3,visible=False))
    for name, box in yaml.safe_load((ROOT/'static_world.yml').read_text())['cuboid'].items():
        m=trimesh.creation.box(box['dims'])
        server.scene.add_mesh_simple('/world/'+name,m.vertices,m.faces,position=np.array(box['pose'][:3]),wxyz=np.array(box['pose'][3:]),color=(165,120,75) if 'Table' in name else (215,220,225),opacity=.12 if 'Wall' in name else 1.)
    m=trimesh.creation.cylinder(radius=scene['cup']['radius_m'],height=scene['cup']['height_m'])
    server.scene.add_mesh_simple('/cup',m.vertices,m.faces,position=partial['object_pose'][:3],color=(245,100,30))
    for label,key,color in [('预抓取','pregrasp_pose',(50,160,255)),('失败目标：73.3%','failed_target_pose',(255,40,40)),('最终抓取目标','grasp_pose',(40,200,90))]:
        pose=partial[key]
        server.scene.add_frame('/targets/'+key,position=pose[:3],wxyz=pose[3:],axes_length=.06,axes_radius=.002)
        server.scene.add_label('/labels/'+key,label,position=pose[:3]+[0,0,.09])
        m=trimesh.creation.icosphere(subdivisions=2,radius=.009)
        server.scene.add_mesh_simple('/markers/'+key,m.vertices,m.faces,position=pose[:3],color=color)
    server.scene.add_line_segments('/insertion',points=np.array([[partial['pregrasp_pose'][:3],partial['grasp_pose'][:3]]]),colors=(255,100,45),line_width=3)
    def update(_=None):
        index=int(slider.value)
        with server.atomic():
            for name,node in nodes.items():
                transform=frames[index,names.index(name)]
                node.position=transform[:3,3]
                node.wxyz=Rotation.from_matrix(transform[:3,:3]).as_quat(scalar_first=True)
            for node in balls:node.visible=spheres.value
            if index < count:
                info.content=f'**预抓取轨迹**：{motion["times"][index]:.2f} / {motion["times"][-1]:.2f} 秒。'
            else:
                progress=(index-count+1)/15
                info.content=f'**深入 IK 已解点：{progress:.1%}**。底盘、升降、左臂保持不动。\n\n下一失败目标：73.3%。深入播放每点停留 0.5 秒，仅用于检查姿态，无执行时间含义。'
    slider.on_update(update);spheres.on_update(update)
    @to_start.on_click
    def start(_):
        play.value=False;slider.value=0
    @to_approach.on_click
    def approach(_):
        play.value=False;slider.value=count-1
    @to_last.on_click
    def last(_):
        play.value=False;slider.value=len(frames)-1
    @server.on_client_connect
    def connected(client):
        client.camera.position=(1.5,-1.7,2.2)
        client.camera.look_at=(-1.05,1.05,.85)
    update()
    print('REPLAY_READY http://127.0.0.1:8094/',flush=True)
    while True:
        if play.value:
            index=int(slider.value)
            if index==len(frames)-1:
                play.value=False
            else:
                delay=float(motion['times'][index+1]-motion['times'][index]) if index<count-1 else .5
                time.sleep(delay/float(speed.value[:-1]));slider.value=index+1
        else:
            time.sleep(.05)

if __name__=='__main__':
    export_frames() if '--export' in sys.argv else show()
