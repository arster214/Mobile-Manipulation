"""Build three-link experimental sphere alternatives; never overwrite the active config."""
from pathlib import Path
import copy
import hashlib
import itertools
import json
import time
import xml.etree.ElementTree as ET

import cumotion
import manifold3d
import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'realman_rm75/config/rm75_dual.yml'
OUTPUT = ROOT / 'collision_models/comparison'
GRIDS = {'base_link_underpan': (3, 5, 2), 'platform_base_link': (4, 3, 2), 'l_link2': (1, 1, 5)}
FINE_GRIDS = {'base_link_underpan': (4, 6, 3), 'platform_base_link': (5, 4, 3), 'l_link2': (1, 1, 7)}
LABELS = {'base_link_underpan': '底盘主体', 'platform_base_link': '升降平台', 'l_link2': '左臂连杆 l_link2'}


def load_mesh(urdf: Path, name: str) -> trimesh.Trimesh:
    link = next(x for x in ET.parse(urdf).findall('link') if x.attrib['name'] == name)
    meshes = []
    for visual in link.findall('visual'):
        item = visual.find('geometry/mesh')
        if item is None:
            continue
        mesh = trimesh.load(urdf.parent / item.attrib['filename'], force='mesh')
        mesh.apply_scale(np.fromstring(item.attrib.get('scale', '1 1 1'), sep=' '))
        origin = visual.find('origin')
        transform = np.eye(4)
        if origin is not None:
            transform = trimesh.transformations.euler_matrix(*np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' '))
            transform[:3, 3] = np.fromstring(origin.attrib.get('xyz', '0 0 0'), sep=' ')
        mesh.apply_transform(transform)
        meshes.append(mesh)
    return trimesh.util.concatenate(meshes)


def clip_polygon(points: np.ndarray, axis: int, boundary: float, lower: bool) -> np.ndarray:
    if len(points) == 0:
        return points
    result = []
    a = points[-1]
    da = (a[axis] - boundary) * (1 if lower else -1)
    for b in points:
        db = (b[axis] - boundary) * (1 if lower else -1)
        if (da >= 0) != (db >= 0):
            result.append(a + da / (da - db) * (b - a))
        if db >= 0:
            result.append(b)
        a, da = b, db
    return np.asarray(result).reshape(-1, 3)


def regular_spheres(mesh: trimesh.Trimesh, grid: tuple[int, int, int]) -> list[dict]:
    """Cover every clipped triangle with a lattice-centered ball, rounding radii upward.

    This covers the input triangle surface, not a certified solid volume for open meshes.
    The lattice partition is deterministic. The one-column arm grid forms an axial chain.
    """
    low, high = mesh.bounds
    count = np.asarray(grid)
    step = (high - low) / count
    edges = [np.linspace(low[a], high[a], grid[a]+1) for a in range(3)]
    axial_centers = None
    if grid[0] == grid[1] == 1:
        end_inset = min(mesh.extents[:2]) / 2
        axial_centers = np.linspace(low[2] + end_inset, high[2] - end_inset, grid[2])
        edges[2] = np.r_[low[2], (axial_centers[:-1]+axial_centers[1:])/2, high[2]]
    radii = {}
    centers = {}
    for triangle in mesh.triangles:
        first = [int(np.clip(np.searchsorted(edges[a], triangle[:, a].min(), side='right')-1, 0, grid[a]-1)) for a in range(3)]
        last = [int(np.clip(np.searchsorted(edges[a], triangle[:, a].max(), side='right')-1, 0, grid[a]-1)) for a in range(3)]
        for index in itertools.product(*(range(first[d], last[d] + 1) for d in range(3))):
            lo = np.array([edges[a][index[a]] for a in range(3)])
            hi = np.array([edges[a][index[a]+1] for a in range(3)])
            poly = triangle
            for axis in range(3):
                poly = clip_polygon(poly, axis, lo[axis], True)
                poly = clip_polygon(poly, axis, hi[axis], False)
            if len(poly):
                center = (lo + hi) / 2
                if axial_centers is not None:
                    center[2] = axial_centers[index[2]]
                radii[index] = max(radii.get(index, 0), float(np.linalg.norm(poly - center, axis=1).max()))
                centers[index] = center
    # A 5 mm radius palette produces smoother size transitions. 1 mm numerical margin.
    return [{'center': centers[i].tolist(), 'radius': float(np.ceil((r + .001) / .005) * .005)}
            for i, r in sorted(radii.items())]


def closed_proxy(mesh: trimesh.Trimesh, grid: tuple[int, int, int], spheres: list[dict]) -> trimesh.Trimesh:
    # Every occupied lattice cell encloses its clipped source triangles. Boolean-union
    # the cells to remove internal faces; do not feed overlapping boxes to the generator.
    if grid[0] == grid[1] == 1:
        return mesh.convex_hull
    step = mesh.extents / np.asarray(grid)
    boxes = [manifold3d.Manifold.cube(tuple(step + 2e-5), center=True).translate(s['center']) for s in spheres]
    result = manifold3d.Manifold.batch_boolean(boxes, manifold3d.OpType.Add).to_mesh64()
    proxy = trimesh.Trimesh(vertices=np.asarray(result.vert_properties)[:, :3], faces=np.asarray(result.tri_verts), process=False)
    assert proxy.is_watertight and proxy.is_volume
    return proxy


def native_spheres(proxy: trimesh.Trimesh, budget: int) -> tuple[list[dict], dict]:
    generator = cumotion.create_collision_sphere_generator(proxy.vertices, proxy.faces)
    output = generator.generate_spheres(budget, .005)
    spheres = [{'center': np.asarray(x.center).reshape(3).tolist(), 'radius': float(x.radius)} for x in output]
    return spheres, {'budget': budget, 'radius_offset_m': .005, 'watertight_after_cleanup': bool(proxy.is_watertight),
                     'validated_triangles': int(generator.num_triangles()),
                     'note': 'Native medial-axis output on a watertight union of occupied grid cells, plus 5 mm radius offset. No coverage post-correction.'}


def metrics(mesh: trimesh.Trimesh, spheres: list[dict], samples: np.ndarray) -> dict:
    c = np.array([s['center'] for s in spheres]); r = np.array([s['radius'] for s in spheres])
    assert len(r) and np.isfinite(c).all() and np.isfinite(r).all() and (r > 0).all()
    gap = (np.linalg.norm(samples[:, None] - c[None], axis=2) - r).min(axis=1)
    vertex_gap = (np.linalg.norm(mesh.vertices[:, None] - c[None], axis=2) - r).min(axis=1)
    aabb_excess = np.maximum(np.maximum(mesh.bounds[0] - (c - r[:, None]), c + r[:, None] - mesh.bounds[1]), 0)
    return {'count': len(r), 'sampled_surface_coverage_pct': float(100 * np.mean(gap <= 1e-8)),
            'sampled_max_gap_mm': float(max(0, gap.max()) * 1000),
            'vertex_max_gap_mm': float(max(0, vertex_gap.max()) * 1000),
            'aabb_excess_mm': float(aabb_excess.max() * 1000),
            'radius_min_mm': float(r.min()*1000), 'radius_max_mm': float(r.max()*1000)}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original = SOURCE.read_bytes()
    config = yaml.safe_load(original)
    k = config['kinematics']
    report = {'source': str(SOURCE), 'source_sha256': hashlib.sha256(original).hexdigest(),
              'status': 'experimental_comparison_only', 'samples_per_link': 30000, 'evaluation_seed': 42,
              'metric_notes': 'Surface coverage uses area-weighted samples of original visual triangles. AABB excess is a lower bound on protrusion relative to the individual link, not exact mesh overshoot. No solid-volume or collision-planning certification.',
              'links': {}}
    variants = {method: copy.deepcopy(config) for method in ['structured', 'native', 'structured_fine', 'native_fine']}
    for name, grid in GRIDS.items():
        t = time.monotonic()
        mesh = load_mesh(Path(k['urdf_path']), name)
        mesh.export(OUTPUT / (name + '.ply'))
        structured = regular_spheres(mesh, grid)
        proxy = closed_proxy(mesh, grid, structured)
        proxy.export(OUTPUT / (name + '_proxy.ply'))
        native, native_info = native_spheres(proxy, len(structured))
        fine = regular_spheres(mesh, FINE_GRIDS[name])
        fine_proxy = closed_proxy(mesh, FINE_GRIDS[name], fine)
        fine_proxy.export(OUTPUT / (name + '_proxy_fine.ply'))
        native_fine, fine_info = native_spheres(fine_proxy, len(fine))
        np.random.seed(42)
        samples, _ = trimesh.sample.sample_surface(mesh, 30000)
        balls = {'baseline': k['collision_spheres'][name], 'structured': structured, 'native': native,
                 'structured_fine': fine, 'native_fine': native_fine}
        stats = {method: metrics(mesh, s, samples) for method, s in balls.items()}
        assert stats['structured']['sampled_max_gap_mm'] < 1e-5
        assert stats['structured']['vertex_max_gap_mm'] < 1e-5
        assert stats['structured_fine']['sampled_max_gap_mm'] < 1e-5
        assert stats['structured_fine']['vertex_max_gap_mm'] < 1e-5
        report['links'][name] = {'label': LABELS[name], 'bounds': mesh.bounds.tolist(), 'grid': grid,
                                 'fine_grid': FINE_GRIDS[name], 'native_info': native_info, 'native_fine_info': fine_info,
                                 'proxy_method': 'convex hull of this small arm link' if name == 'l_link2' else 'watertight boolean union of occupied lattice cells',
                                 'spheres': balls, 'metrics': stats}
        for method, variant in variants.items():
            variant['kinematics']['collision_spheres'][name] = balls[method]
        print(name, json.dumps(stats), 'seconds', round(time.monotonic()-t, 2), flush=True)
    for method, variant in variants.items():
        (OUTPUT / f'rm75_{method}_experimental.yml').write_text(yaml.safe_dump(variant, sort_keys=False))
    (OUTPUT/'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    assert SOURCE.read_bytes() == original
    print('Saved experimental outputs:', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
