#!/usr/bin/env python3
"""Turn RobotBuilder output into the fixed-base 14-DOF RM75 dual-arm config."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "config" / "rm75_generated.yml"
OUTPUT = ROOT / "config" / "rm75_dual.yml"

JOINTS = [f"l_joint{i}" for i in range(1, 8)] + [f"r_joint{i}" for i in range(1, 8)]


def main() -> None:
    data = yaml.safe_load(SOURCE.read_text())
    kin = data["kinematics"]

    # These joints remain present in the full robot geometry but are not optimized.
    kin["lock_joints"] = {
        "platform_joint": 0.5,
        "head_joint1": 0.0,
        "head_joint2": 0.0,
        "l_Joint_finger1": 0.0325,
        "r_Joint_finger1": 0.0325,
    }
    kin["cspace"] = {
        "joint_names": JOINTS,
        "default_joint_position": [0.0] * 14,
        "null_space_weight": [1.0] * 14,
        "cspace_distance_weight": [1.0] * 14,
        # Conservative initial values; replace with verified RM75 limits before hardware use.
        "max_acceleration": [3.0] * 14,
        "max_jerk": [30.0] * 14,
    }
    kin["urdf_path"] = str(ROOT / "assets" / "urdf" / "rm75_dual_rmg24_curobo.urdf")
    kin["asset_root_path"] = str(ROOT / "assets" / "urdf")
    kin["tool_frames"] = ["l_tool0", "r_tool0"]
    kin["format_version"] = 2.0

    OUTPUT.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
