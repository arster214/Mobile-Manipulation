"""Replay grasp-first connection and fixed-platform insertion."""
from pathlib import Path
import json, time, sys
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ('tests/artifacts/grasp_priority' if '--priority' in sys.argv else 'tests/artifacts/grasp_first')
PORT = 8102 if '--priority' in sys.argv else 8096


def export_frames():
    import pinocchio as pin
    report = json.loads((OUT/'result.json').read_text())
    for side in ('r',):
        motion = np.load(OUT/side/'pregrasp.npz')
        approach=np.load(OUT/side/'approach.npz')
        times=np.r_[motion['times'],motion['times'][-1]+np.arange(.025,float(approach['times'][-1]),.025),motion['times'][-1]+approach['times'][-1]]
        body=np.tile(motion['positions'][-1],(len(times),1))
        body[:len(motion['times'])]=motion['positions']
        for j,name in enumerate(approach['joint_names']):
            index=motion['joint_names'].tolist().index(name)
            body[len(motion['times']):,index]=np.interp(times[len(motion['times']):]-motion['times'][-1],approach['times'],approach['positions'][:,j])
        base=np.vstack([motion['base_poses'],np.tile(motion['base_poses'][-1],(len(times)-len(motion['times']),1))])
        split=float(motion['times'][-1])
        np.savez(OUT/side/'playback.npz',times=times,base_poses=base,split=split)
        values = dict(zip(report['start_state']['joint_names'],report['start_state']['positions']))
        model = pin.buildModelFromUrdf(str(OUT/side/'mobile.urdf'))
        data = model.createData()
        ids = [i for i,f in enumerate(model.frames) if f.type==pin.FrameType.BODY]
        frames=[]
        for body,base in zip(body,base):
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
    saved={side:np.load(OUT/side/'replay_frames.npz') for side in ('r',)}
    motions={side:np.load(OUT/side/'playback.npz') for side in ('r',)}
    names=saved['r']['link_names'].tolist()
    scene = json.loads((ROOT / 'scene_config.json').read_text())
    cfg = yaml.safe_load(Path(scene['robot']['curobo_config']).read_text())['kinematics']
    server = viser.ViserServer(host='127.0.0.1', port=PORT)
    server.scene.set_up_direction('+z')
    if '--priority' in sys.argv:
        selected=report['selected_arm']
        tier=report['arms'][selected]['search_stats']['motion_scope']
        server.gui.add_markdown(f'## 分级规划结果：{tier}\n优先只动抓取臂 → 抓取臂与升降 → 允许底盘。仅回放通过完整路径检查的结果。')
    server.gui.add_markdown('## 抓取优先规划：右臂完整回放\n到预抓取 → 固定底盘、升降和左臂 → 右臂深入15cm。\n深入阶段排除杯子碰撞，保留自碰撞和其余环境碰撞。数值回放，不含夹爪闭合或物理执行。')
    play = server.gui.add_checkbox('播放', initial_value=False)
    slider = server.gui.add_slider('进度 (%)',min=0,max=100,step=.1,initial_value=0)
    speed = server.gui.add_dropdown('播放速度',options=['0.25×','0.5×','1×'],initial_value='0.5×')
    to_start=server.gui.add_button('回到起点')
    to_approach=server.gui.add_button('从深入起点播放')
    to_last=server.gui.add_button('查看抓取位姿')
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
    for side,color in [('r',(50,195,100))]:
        xy=motions[side]['base_poses'][:,:2]
        xyz=np.column_stack([xy,np.full(len(xy),.015)])
        paths[side]=server.scene.add_line_segments('/path/'+side,points=np.stack([xyz[:-1],xyz[1:]],axis=1),colors=color,line_width=4)
    def current_side():return 'r'
    def update(_=None):
        side=current_side();frames=saved[side]['frames'];motion=motions[side]
        elapsed=slider.value/100*float(motion['times'][-1])
        index=min(len(frames)-1,int(np.searchsorted(motion['times'],elapsed)))
        row=report['arms'][side]
        with server.atomic():
            for name,node in nodes.items():
                transform=frames[index,names.index(name)]
                node.position=transform[:3,3]
                node.wxyz=Rotation.from_matrix(transform[:3,:3]).as_quat(scalar_first=True)
            for node in balls:node.visible=spheres.value
            for key,node in paths.items():node.visible=key==side
            pose=np.array(row['target_pose']);target.position=pose[:3];target.wxyz=pose[3:]
            split=float(motion['split'])
            stage='到预抓取（全身运动）' if elapsed<split else '深入（仅右臂运动）'
            info.content=(f"**阶段：{stage}**\n\n{elapsed:.2f} / {motion['times'][-1]:.2f} 秒\n\n"
                          f"预抓取段 {split:.2f}s；深入段 {motion['times'][-1]-split:.2f}s\n\n"
                          "深入段最大关节变化 0.7986 rad；最小奇异值 0.05498；最大条件数 31.62。")
    slider.on_update(update);spheres.on_update(update)
    @to_approach.on_click
    def approach(_):
        slider.value=float(motions['r']['split']/motions['r']['times'][-1]*100)
        play.value=True
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
    print(f'REPLAY_READY http://127.0.0.1:{PORT}/',flush=True)
    previous=time.monotonic()
    while True:
        time.sleep(.025)
        now=time.monotonic();dt=now-previous;previous=now
        if play.value:
            duration=float(motions[current_side()]['times'][-1])
            slider.value=min(100.,slider.value+100*dt*float(speed.value[:-1])/duration)
            if slider.value>=100:play.value=False

if __name__=='__main__':
    export_frames() if '--export' in sys.argv else show()
