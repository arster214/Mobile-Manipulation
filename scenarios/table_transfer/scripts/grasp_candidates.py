"""Geometric candidates only: no IK, collision acceptance or robot commands.

Grasp frame: X finger closing axis, Y up, Z approach into the cup.
This abstract frame is NOT r_tool0; its calibrated tool transform is required.
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def add(a, b, scale=1.0):
    return [x + scale * y for x, y in zip(a, b)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def generate(table, cup, policy):
    """Upright cylinder on a horizontal rectangular table; world coordinates, SI."""
    length, width, top = table['size_m']
    if length <= width:
        raise ValueError('Table local X must be its long axis')
    yaw = math.radians(table.get('yaw_deg', 0.0))
    tangent = [math.cos(yaw), math.sin(yaw), 0.0]
    normal = [-math.sin(yaw), math.cos(yaw), 0.0]
    center = [*table['center_xy_m'], top]
    p = cup['center_m']
    radius, height = cup['radius_m'], cup['height_m']
    relative = add(p, center, -1)
    u, v = dot(relative, tangent), dot(relative, normal)
    if radius <= 0 or height <= 0 or abs(u) + radius > length / 2 or abs(v) + radius > width / 2:
        raise ValueError('Cup footprint must fit on tabletop')
    # Freeze this side for a grasp attempt; re-evaluate when object pose changes.
    side = 1 if v > policy['side_tie_tolerance_m'] else -1
    if abs(v) <= policy['side_tie_tolerance_m']:
        side = policy['tie_side_sign']
    outward = [side * x for x in normal]
    approach = [-x for x in outward]
    closing = [-approach[1], approach[0], 0.0]  # up cross approach
    up = [0.0, 0.0, 1.0]
    rotation = [[closing[i], up[i], approach[i]] for i in range(3)]
    edge = add(center, outward, width / 2)
    n = policy['height_segments']
    if n < 1 or n % 2 != 1:
        raise ValueError('Use an odd positive segment count for a central segment')
    backoff = policy['pregrasp_backoff_m']
    if backoff <= radius:
        raise ValueError('Pregrasp backoff must exceed cup radius')
    ordered = sorted(range(n), key=lambda i: (abs(i - n // 2), i))
    candidates = []
    for index in ordered:
        z_offset = ((index + 0.5) / n - 0.5) * height
        margin = height / 2 - abs(z_offset)
        if margin < policy['end_margin_m']:
            continue
        grasp = add(p, up, z_offset)
        pre = add(grasp, outward, backoff)
        candidates.append({
            'rank': len(candidates), 'height_segment': index,
            'grasp_center_world_m': grasp,
            'pregrasp_center_world_m': pre,
            'contact_pair_world_m': [add(grasp, closing, radius), add(grasp, closing, -radius)],
            'rotation_world_from_grasp': rotation,
            'pregrasp_center_outside_table_edge': dot(add(pre, edge, -1), outward) > 0,
            'status': 'geometry_only_requires_tool_transform_ik_and_swept_collision_check',
        })
    if not candidates:
        raise ValueError('No candidates remain after end margins')
    return {
        'table': table['name'], 'cup_pose_source': cup['pose_source'],
        'units': 'meters', 'frame': 'world',
        'sampling_assumption': 'upright_cylinder_height_segments_center_outward',
        'selected_long_edge_sign_local_y': side,
        'selected_long_edge_midpoint_world_m': edge,
        'outward_normal_world': outward, 'approach_direction_world': approach,
        'base_region_rule': 'base footprint must stay outside selected long edge; check full robot against table and room',
        'tool_from_grasp_transform': None,
        'execution_ready': False, 'candidates': candidates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, default=ROOT / 'grasp_policy.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'grasp_candidates.json')
    parser.add_argument('--cup-center', nargs=3, type=float, help='Current world ground-truth center in meters')
    args = parser.parse_args()
    scene = json.loads((ROOT / 'scene_config.json').read_text())
    policy = json.loads(args.policy.read_text())
    table = next(t for t in scene['tables'] if t['name'] == policy['source_table'])
    cup = dict(scene['cup'])
    cup['center_m'] = args.cup_center or cup['initial_center_m']
    cup['pose_source'] = 'supplied_world_ground_truth' if args.cup_center else 'scene_config_initial_pose_not_live_measurement'
    result = generate(table, cup, policy)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'candidate_count': len(result['candidates']), 'approach': result['approach_direction_world'], 'execution_ready': False}))


if __name__ == '__main__':
    main()
