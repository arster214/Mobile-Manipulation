"""Inspect the existing RM75 sphere configuration without changing it."""
from functools import partial
from pathlib import Path
import argparse
import threading
import time

import numpy as np
import viser
import viser.transforms as tf
from viser.extras import ViserUrdf
import yaml
import yourdfpy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'realman_rm75/config/rm75_dual.yml')
    args = parser.parse_args()
    k = yaml.safe_load(args.config.read_text())['kinematics']
    sphere_data = k['collision_spheres']
    total = sum(len(s) for s in sphere_data.values())
    urdf_path = Path(k['urdf_path'])
    urdf = yourdfpy.URDF.load(
        urdf_path, build_scene_graph=True, load_meshes=True,
        filename_handler=partial(yourdfpy.filename_handler_magic, dir=urdf_path.parent),
    )
    server = viser.ViserServer(host=args.host, port=args.port, label='RM75 全身碰撞球')
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
        selected = server.gui.add_dropdown('查看 link', options=['全身'] + list(sphere_data), initial_value='全身')
        count = server.gui.add_markdown(f'当前显示：**{total} 个球**')
        reset = server.gui.add_button('恢复配置初始姿态')
    with server.gui.add_folder('双臂关节 / radians'):
        limits = robot.get_actuated_joint_limits()
        for name in active:
            lo, hi = limits[name]
            sliders[name] = server.gui.add_slider(name, min=float(lo), max=float(hi), step=0.01, initial_value=float(start[name]))
    with server.gui.add_folder('配置详情'):
        server.gui.add_markdown(f'配置：`{args.config}`\n\n球缓冲：`{k.get("collision_sphere_buffer", 0)}` m\n\n升降固定 0.5 m，头部固定，夹爪按配置张开。\n\n球心使用各 link 局部坐标，并随关节正运动学更新。')
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

    update_pose(None)
    print(f'READY: http://{args.host}:{args.port} | {total} spheres / {len(frames)} links', flush=True)
    while True:
        time.sleep(0.5)


if __name__ == '__main__':
    main()
