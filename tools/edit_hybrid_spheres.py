"""Edit a hybrid draft in link-local coordinates; save only versioned candidates."""
from functools import partial
from pathlib import Path
import argparse
import threading
import datetime
import json
import trimesh
from build_sphere_comparison import load_mesh, metrics
import time

import numpy as np
import viser
import viser.transforms as tf
from viser.extras import ViserUrdf
import yaml
import yourdfpy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8082)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'collision_models/hybrid/rm75_hybrid_draft.yml')
    parser.add_argument('--focus-link', default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    k = config['kinematics']
    sphere_data = k['collision_spheres']
    total = sum(len(s) for s in sphere_data.values())
    urdf_path = Path(k['urdf_path'])
    urdf = yourdfpy.URDF.load(
        urdf_path, build_scene_graph=True, load_meshes=True,
        filename_handler=partial(yourdfpy.filename_handler_magic, dir=urdf_path.parent),
    )
    server = viser.ViserServer(host=args.host, port=args.port, label='RM75 混合布球 · 手动编辑草稿')
    server.scene.set_up_direction('+z')
    robot = ViserUrdf(server, urdf, root_node_name='/robot', mesh_color_override=(0.68, 0.70, 0.73, 0.65))
    server.scene.add_grid('/ground', width=4, height=4, plane='xy', position=(0, 0, -0.15))
    frames, balls = {}, {}
    for link, spheres in sphere_data.items():
        if link not in urdf.link_map:
            raise ValueError(f'Collision link missing from URDF: {link}')
        frames[link] = server.scene.add_frame(f'/spheres/{link}', show_axes=False)
        color = (60, 155, 255) if link.startswith('l_') else (255, 150, 45) if link.startswith('r_') else (155, 100, 230)
        balls[link] = [server.scene.add_icosphere(
            f'/spheres/{link}/{i}', radius=float(s['radius']),
            position=tuple(s['center']), color=color, opacity=0.45, subdivisions=2,
        ) for i, s in enumerate(spheres)]
    names = list(robot.get_actuated_joint_names())
    active = k['cspace']['joint_names']
    start = dict(zip(active, k['cspace']['default_joint_position']))
    start.update(k.get('lock_joints') or {})
    initial = np.array([start.get(n, 0.0) for n in names])
    lock = threading.RLock()
    sliders = {}
    with server.gui.add_folder('显示 / Display'):
        server.gui.add_markdown(f'**{total} 个碰撞球 · {len(sphere_data)} 个 link**\n\n蓝：左臂；橙：右臂；紫：身体与底盘。\n\n仅查看配置，滑块不会发送机器人命令。')
        show_robot = server.gui.add_checkbox('机器人网格', initial_value=True)
        show_spheres = server.gui.add_checkbox('碰撞球', initial_value=True)
        alpha = server.gui.add_slider('球透明度', min=0.05, max=1.0, step=0.05, initial_value=0.45)
        wireframe = server.gui.add_checkbox('球线框', initial_value=False)
        selected = server.gui.add_dropdown('查看 link', options=['全身'] + list(sphere_data), initial_value=args.focus_link or '全身')
        count = server.gui.add_markdown(f'当前显示：**{total} 个球**')
        reset = server.gui.add_button('恢复配置初始姿态')
    with server.gui.add_folder('手动编辑 · link 局部坐标 / 米'):
        edit_link = server.gui.add_dropdown('编辑 link', options=list(sphere_data), initial_value=args.focus_link or 'l_link2')
        index = server.gui.add_dropdown('球编号', options=[str(i) for i in range(len(sphere_data[args.focus_link or 'l_link2']))])
        coords = [server.gui.add_number(axis, initial_value=0.0, step=.001) for axis in ('X / m','Y / m','Z / m')]
        radius = server.gui.add_number('半径 / m', initial_value=.05, min=.001, step=.001)
        isolate = server.gui.add_button('仅显示此 link 的球')
        check = server.gui.add_button('检查此 link 表面覆盖')
        check_result = server.gui.add_markdown('编辑后请重新检查覆盖。球心坐标以对应 link 为基准。')
        save = server.gui.add_button('保存为新候选配置')
        saved = server.gui.add_markdown('不会写入当前规划配置；保存文件带时间戳。')
    with server.gui.add_folder('双臂及升降关节'):

        limits = robot.get_actuated_joint_limits()
        for name in active + ['platform_joint']:
            lo, hi = limits[name]
            sliders[name] = server.gui.add_slider(name, min=float(lo), max=float(hi), step=0.01, initial_value=float(start[name]))
    with server.gui.add_folder('配置详情'):
        server.gui.add_markdown(f'配置：`{args.config}`\n\n球缓冲：`{k.get("collision_sphere_buffer", 0)}` m\n\n升降初始 0.5 m，可调节检查；头部与夹爪按配置。此草稿未验证自碰撞及规划。\n\n球心使用各 link 局部坐标，并随关节正运动学更新。')
        server.gui.add_markdown('| link | 球数 |\n|---|---:|\n' + '\n'.join(f'| {n} | {len(s)} |' for n, s in sphere_data.items()))

    def update_pose(_) -> None:
        with lock, server.atomic():
            q = initial.copy()
            for name, slider in sliders.items():
                q[names.index(name)] = slider.value
            robot.update_cfg(q)
            for name, frame in frames.items():
                matrix = urdf.get_transform(name)
                frame.position = matrix[:3, 3]
                frame.wxyz = tf.SO3.from_matrix(matrix[:3, :3]).wxyz

    def update_display(_) -> None:
        with lock, server.atomic():
            robot.show_visual = show_robot.value
            for name, frame in frames.items():
                frame.visible = show_spheres.value and selected.value in ('全身', name)
                for sphere in balls[name]:
                    sphere.opacity = alpha.value
                    sphere.wireframe = wireframe.value
            n = total if selected.value == '全身' else len(sphere_data[selected.value])
            count.content = f'当前显示：**{n if show_spheres.value else 0} 个球** / 总计 {total}'

    editing = {'loading': False}
    def load_fields(_=None):
        with lock:
            editing['loading'] = True
            options = tuple(str(i) for i in range(len(sphere_data[edit_link.value])))
            if tuple(index.options) != options: index.options = options
            if index.value not in index.options: index.value = '0'
            ball = sphere_data[edit_link.value][int(index.value)]
            for control, value in zip(coords, ball['center']): control.value = float(value)
            radius.value = float(ball['radius'])
            editing['loading'] = False

    def edit_ball(_):
        with lock, server.atomic():
            if editing['loading']: return
            link=edit_link.value; i=int(index.value)
            ball={'center':[float(c.value) for c in coords], 'radius':float(radius.value)}
            sphere_data[link][i]=ball
            balls[link][i].remove()
            color=(60,155,255) if link.startswith('l_') else (255,150,45) if link.startswith('r_') else (155,100,230)
            balls[link][i]=server.scene.add_icosphere(f'/spheres/{link}/{i}',radius=ball['radius'],position=tuple(ball['center']),color=color,opacity=alpha.value,subdivisions=2,wireframe=wireframe.value)
            check_result.content='已修改。请点击检查，重新评估覆盖与外凸。'

    edit_link.on_update(load_fields)
    index.on_update(load_fields)
    for c in [*coords,radius]: c.on_update(edit_ball)
    @isolate.on_click
    def isolate_link(_):
        selected.value=edit_link.value
        update_display(None)
    @check.on_click
    def evaluate(_):
        with lock:
            link=edit_link.value
            mesh=load_mesh(urdf_path,link)
            np.random.seed(109)
            samples,_=trimesh.sample.sample_surface(mesh,20000)
            result=metrics(mesh,sphere_data[link],samples)
            check_result.content=(f"**{link} · {result['count']} 球**\n\n表面抽样覆盖 {result['sampled_surface_coverage_pct']:.2f}% · 最大遗漏 {result['sampled_max_gap_mm']:.1f} mm\n\nAABB 外凸下界 {result['aabb_excess_mm']:.1f} mm。仅表面抽样，不等于完整碰撞验证。")
    @save.on_click
    def save_candidate(_):
        with lock:
            stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            path=args.config.parent / f'rm75_hybrid_edited_{stamp}.yml'
            with path.open('x') as f: yaml.safe_dump(config,f,sort_keys=False)
            saved.content=f'已保存：`{path}`\n\n自碰撞排除表沿用基线，尚未验证。'
    load_fields()

    for slider in sliders.values():
        slider.on_update(update_pose)
    for control in (show_robot, show_spheres, alpha, wireframe, selected):
        control.on_update(update_display)

    @reset.on_click
    def reset_pose(_) -> None:
        for name, slider in sliders.items():
            slider.value = float(start[name])
        update_pose(None)

    @server.on_client_connect
    def camera(client: viser.ClientHandle) -> None:
        client.camera.position = (3.1, 3.1, 2.1)
        client.camera.look_at = (0, 0, 0.85)
        client.camera.up_direction = (0, 0, 1)
        if args.focus_link:
            mesh = load_mesh(urdf_path, args.focus_link)
            matrix = urdf.get_transform(args.focus_link)
            target = matrix[:3,:3] @ mesh.bounds.mean(axis=0) + matrix[:3,3]
            distance = max(float(mesh.extents.max()) * 2.7, .4)
            client.camera.look_at = tuple(target)
            client.camera.position = tuple(target + np.array([1,1,.55]) * distance)


    update_pose(None)
    update_display(None)
    print(f'READY: http://{args.host}:{args.port} | {total} spheres / {len(frames)} links', flush=True)
    while True:
        time.sleep(0.5)


if __name__ == '__main__':
    main()
