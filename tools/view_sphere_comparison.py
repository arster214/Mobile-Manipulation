"""Three-column Viser comparison of saved RM75 collision-sphere experiments."""
from pathlib import Path
import argparse
import json
import threading
import time

import numpy as np
import trimesh
import viser

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'collision_models/comparison'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8081)
    args = parser.parse_args()
    report = json.loads((DATA/'comparison.json').read_text())
    server = viser.ViserServer(host='127.0.0.1', port=args.port, label='RM75 碰撞球方案对比')
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/ground', width=5, height=3, plane='xy', position=(0, 0, -.43))
    link_options = {d['label']: name for name, d in report['links'].items()}
    meshes = {name: trimesh.load(DATA/(name+'.ply'), force='mesh') for name in report['links']}
    samples = {}
    for name, mesh in meshes.items():
        np.random.seed(93)
        samples[name], _ = trimesh.sample.sample_surface(mesh, 4000)
    lock = threading.RLock()
    handles = []
    with server.gui.add_folder('对比设置'):
        server.gui.add_markdown('**左：现有配置 · 中：规则布球 · 右：NVIDIA 自动**\n\n仅对三个 link 做实验，正式配置未替换。')
        selected = server.gui.add_dropdown('部位', options=list(link_options), initial_value='底盘主体')
        quality = server.gui.add_dropdown('球数档位', options=['紧凑', '更细'], initial_value='紧凑')
        show_mesh = server.gui.add_checkbox('显示原始网格', initial_value=True)
        show_spheres = server.gui.add_checkbox('显示碰撞球', initial_value=True)
        alpha = server.gui.add_slider('球透明度', min=.05, max=1.0, step=.05, initial_value=.45)
        mesh_alpha = server.gui.add_slider('网格透明度', min=.05, max=1.0, step=.05, initial_value=.65)
        wireframe = server.gui.add_checkbox('球线框', initial_value=False)
        gaps = server.gui.add_checkbox('红色显示遗漏表面点', initial_value=False)
        proxy = server.gui.add_checkbox('显示 NVIDIA 输入代理网格', initial_value=False)
        reset = server.gui.add_button('重置观察视角')
    with server.gui.add_folder('定量对比'):
        stats = server.gui.add_markdown('')
    with server.gui.add_folder('如何解读'):
        server.gui.add_markdown('覆盖：每个 link 原始三角表面独立抽样 30,000 点。\n\n遗漏：网格表面到最近球实体的最大正距离。\n\n越界下界：球超出该 link 原始网格轴对齐包围盒的量，不是精确网格外凸。\n\n规则版：沿规则网格分割三角形，球心固定在格点，半径向上取整到 5 mm；臂段使用轴向球链。覆盖表面不等于认证的实体体积包络。\n\nNVIDIA：在封闭分区代理上生成相同球数的中轴球，额外半径 5 mm；没有再做覆盖修补。臂段代理是该单 link 的凸包。\n\n三列等比例显示；切换部位会归一化观察尺寸。')

    def camera(client: viser.ClientHandle) -> None:
        client.camera.position = (0, -2.6, 1.65)
        client.camera.look_at = (0, 0, 0)
        client.camera.up_direction = (0, 0, 1)

    server.on_client_connect(camera)

    @reset.on_click
    def reset_camera(_) -> None:
        for client in server.get_clients().values():
            camera(client)

    def render(_) -> None:
        with lock, server.atomic():
            for handle in handles:
                handle.remove()
            handles.clear()
            name = link_options[selected.value]
            entry = report['links'][name]
            mesh = meshes[name]
            center = mesh.bounds.mean(axis=0)
            scale = .65 / mesh.extents.max()
            fine = quality.value == '更细'
            methods = ['baseline', 'structured_fine' if fine else 'structured', 'native_fine' if fine else 'native']
            titles = ['现有配置', '规则布球', 'NVIDIA 自动']
            colors = [(80, 145, 240), (50, 190, 130), (245, 160, 65)]
            for col, (method, title, color) in enumerate(zip(methods, titles, colors)):
                offset = np.array([(col-1)*.95, 0, 0])
                prefix = f'/comparison/{col}'
                handles.append(server.scene.add_mesh_simple(prefix+'/mesh', vertices=(mesh.vertices-center)*scale+offset,
                    faces=mesh.faces, color=(155, 163, 175), opacity=mesh_alpha.value, visible=show_mesh.value))
                balls = entry['spheres'][method]
                for i, ball in enumerate(balls):
                    handles.append(server.scene.add_icosphere(prefix+f'/sphere/{i}', radius=ball['radius']*scale,
                        position=(np.array(ball['center'])-center)*scale+offset, color=color,
                        opacity=alpha.value, wireframe=wireframe.value, subdivisions=2, visible=show_spheres.value))
                m = entry['metrics'][method]
                label = f"{title}\n{m['count']} 球 · 表面覆盖 {m['sampled_surface_coverage_pct']:.1f}%"
                handles.append(server.scene.add_label(prefix+'/title', label, position=offset+np.array([0, 0, .40]), anchor='center'))
                pts = samples[name]
                c = np.array([b['center'] for b in balls]); r = np.array([b['radius'] for b in balls])
                uncovered = (np.linalg.norm(pts[:, None]-c[None], axis=2)-r).min(axis=1) > .0005
                if uncovered.any():
                    handles.append(server.scene.add_point_cloud(prefix+'/gaps', points=(pts[uncovered]-center)*scale+offset,
                        colors=(230, 25, 45), point_size=.005, visible=gaps.value))
                if col == 2:
                    file = DATA/(name+('_proxy_fine.ply' if fine else '_proxy.ply'))
                    pm = trimesh.load(file, force='mesh')
                    handles.append(server.scene.add_mesh_simple(prefix+'/proxy', vertices=(pm.vertices-center)*scale+offset,
                        faces=pm.faces, color=(30, 90, 160), wireframe=True, opacity=.5, visible=proxy.value))
            rows = ['| 指标 | 现有 | 规则 | NVIDIA |', '|---|---:|---:|---:|']
            for key, title, fmt in [('count', '球数', '.0f'), ('sampled_surface_coverage_pct', '表面覆盖 %', '.1f'),
                                    ('sampled_max_gap_mm', '最大遗漏 mm', '.1f'), ('aabb_excess_mm', '越界下界 mm', '.1f')]:
                rows.append('| '+title+' | '+' | '.join(format(entry['metrics'][m][key], fmt) for m in methods)+' |')
            rows.append(f'\n原始尺寸：{mesh.extents[0]*1000:.0f} × {mesh.extents[1]*1000:.0f} × {mesh.extents[2]*1000:.0f} mm。')
            rows.append('\n**实验方案：未通过全身碰撞与规划验收。**')
            stats.content = '\n'.join(rows)

    for control in [selected, quality, show_mesh, show_spheres, alpha, mesh_alpha, wireframe, gaps, proxy]:
        control.on_update(render)
    render(None)
    print(f'READY: http://127.0.0.1:{args.port}', flush=True)
    while True:
        time.sleep(.5)


if __name__ == '__main__':
    main()
