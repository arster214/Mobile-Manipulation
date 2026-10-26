"""Replay the two constrained pregrasp paths, excluding insertion."""
from pathlib import Path
import json, time, sys
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tests/artifacts/arm_comparison'


def export_frames():
    import pinocchio as pin
    report = json.loads((OUT/'result.json').read_text())
    for side in ('l','r'):
        motion = np.load(OUT/side/'pregrasp.npz')
        values = dict(zip(report['start_state']['joint_names'],report['start_state']['positions']))
        model = pin.buildModelFromUrdf(str(OUT/side/'mobile.urdf'))
        data = model.createData()
        ids = [i for i,f in enumerate(model.frames) if f.type==pin.FrameType.BODY]
        frames=[]
        for body,base in zip(motion['positions'],motion['base_poses']):
            values.update(zip(motion['joint_names'],body))
            values.update(zip(['base_x','base_y','base_yaw'],base))
            for arm in ('l','r'):values[f'{arm}_Joint_finger2']=values[f'{arm}_Joint_finger1']
            q=pin.neutral(model)
            for name,value in values.items():
                if model.existJointName(name):q[model.joints[model.getJointId(name)].idx_q]=value
            pin.framesForwardKinematics(model,data,q)
            frames.append(np.array([data.oMf[i].homogeneous for i in ids]))
        np.savez_compressed(OUT/side/'replay_frames.npz',frames=frames,link_names=[model.frames[i].name for i in ids])
        print(side,len(frames),'frames')


def show():
    import xml.etree.ElementTree as ET
    import trimesh
    import viser
    report=json.loads((OUT/'result.json').read_text())
    saved={side:np.load(OUT/side/'replay_frames.npz') for side in ('l','r')}
    motions={side:np.load(OUT/side/'pregrasp.npz') for side in ('l','r')}
    names=saved['r']['link_names'].tolist()
    scene = json.loads((ROOT / 'scene_config.json').read_text())
    cfg = yaml.safe_load(Path(scene['robot']['curobo_config']).read_text())['kinematics']
    server = viser.ViserServer(host='127.0.0.1', port=8095)
    server.scene.set_up_direction('+z')
    server.gui.add_markdown('## 左右臂预抓取轨迹对比\n只显示起点到预抓取位姿，不包含深入。右臂为当前低代价候选。\n数值轨迹回放，非 PhysX 执行；奇异性阈值适用于新 IK 终点。')
    select=server.gui.add_dropdown('抓取臂',options=['右臂（优先候选）','左臂'],initial_value='右臂（优先候选）')
    play = server.gui.add_checkbox('播放', initial_value=False)
    slider = server.gui.add_slider('进度 (%)',min=0,max=100,step=.1,initial_value=0)
    speed = server.gui.add_dropdown('播放速度',options=['0.25×','0.5×','1×'],initial_value='0.5×')
    to_start=server.gui.add_button('回到起点')
    to_last=server.gui.add_button('查看预抓取终点')
    spheres = server.gui.add_checkbox('显示碰撞球', initial_value=False)
    info = server.gui.add_markdown('')
    nodes, balls = {}, []
    robot = ET.parse(OUT / 'r/mobile.urdf').getroot()
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
    server.scene.add_mesh_simple('/cup',m.vertices,m.faces,position=np.array(report['object_pose'][:3]),color=(245,100,30))
    pose=np.array(report['arms']['r']['target_pose'])
    target=server.scene.add_frame('/target',position=pose[:3],wxyz=pose[3:],axes_length=.06,axes_radius=.002)
    server.scene.add_label('/target_label','预抓取终点',position=pose[:3]+[0,0,.1])
    paths={}
    for side,color in [('l',(80,140,255)),('r',(50,195,100))]:
        xy=motions[side]['base_poses'][:,:2]
        xyz=np.column_stack([xy,np.full(len(xy),.015)])
        paths[side]=server.scene.add_line_segments('/path/'+side,points=np.stack([xyz[:-1],xyz[1:]],axis=1),colors=color,line_width=4)
    def current_side():return 'r' if select.value.startswith('右') else 'l'
    def update(_=None):
        side=current_side();frames=saved[side]['frames'];motion=motions[side]
        index=min(len(frames)-1,round(slider.value/100*(len(frames)-1)))
        row=report['arms'][side]
        with server.atomic():
            for name,node in nodes.items():
                transform=frames[index,names.index(name)]
                node.position=transform[:3,3]
                node.wxyz=Rotation.from_matrix(transform[:3,:3]).as_quat(scalar_first=True)
            for node in balls:node.visible=spheres.value
            for key,node in paths.items():node.visible=key==side
            pose=np.array(row['target_pose']);target.position=pose[:3];target.wxyz=pose[3:]
            metrics=row['constraint_metrics']['endpoint']
            info.content=(f"**{'右臂' if side=='r' else '左臂'}**：{motion['times'][index]:.2f} / {motion['times'][-1]:.2f} 秒\n\n"
                          f"路径代价：**{row['pregrasp_cost']:.3f}**\n\n"
                          f"最大单关节变化：{metrics['max_arm_joint_change_rad']:.3f} rad\n\n"
                          f"终点 σmin：{metrics['min_singular_value']:.4f}；条件数：{metrics['max_condition_number']:.2f}\n\n"
                          f"物体方向：{'左' if side=='l' else '右'}前方 {abs(row['heading_deg']):.2f}°")
    slider.on_update(update);spheres.on_update(update)
    @select.on_update
    def choose(_):
        play.value=False;slider.value=0;update()
    @to_start.on_click
    def start(_):
        play.value=False;slider.value=0
    @to_last.on_click
    def last(_):
        play.value=False;slider.value=100
    @server.on_client_connect
    def connected(client):
        client.camera.position=(.5,-.8,1.9)
        client.camera.look_at=(-1.05,1.05,.85)
    update()
    print('REPLAY_READY http://127.0.0.1:8095/',flush=True)
    while True:
        time.sleep(.025)
        if play.value:
            duration=float(motions[current_side()]['times'][-1])
            slider.value=min(100.,slider.value+100*.025*float(speed.value[:-1])/duration)
            if slider.value>=100:play.value=False

if __name__=='__main__':
    export_frames() if '--export' in sys.argv else show()
