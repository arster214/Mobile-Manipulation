#!/usr/bin/env python3
"""Load the migrated robot through cuRobo and verify its active kinematic model."""

from pathlib import Path

import yaml

from curobo._src.types.robot import RobotCfg

from rm75_planner import LEFT_JOINTS, RIGHT_JOINTS, make_mode_robot_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "rm75_dual.yml"
EXPECTED = [f"l_joint{i}" for i in range(1, 8)] + [f"r_joint{i}" for i in range(1, 8)]


def main() -> None:
    data = yaml.safe_load(CONFIG.read_text())
    robot = RobotCfg.create(data)
    actual = list(robot.cspace.joint_names)
    if actual != EXPECTED:
        raise RuntimeError(f"Active joint mismatch:\nexpected={EXPECTED}\nactual={actual}")
    if robot.kinematics.kinematics_config.tool_frames != ["l_tool0", "r_tool0"]:
        raise RuntimeError("Tool-frame mismatch")

    for mode, expected_joints, expected_tools in (
        ("left", LEFT_JOINTS, ["l_tool0"]),
        ("right", RIGHT_JOINTS, ["r_tool0"]),
        ("dual", EXPECTED, ["l_tool0", "r_tool0"]),
    ):
        mode_data = make_mode_robot_config(data, mode)
        mode_robot = RobotCfg.create(mode_data)
        if list(mode_robot.cspace.joint_names) != expected_joints:
            raise RuntimeError(f"{mode} active joint mismatch")
        if mode_robot.kinematics.kinematics_config.tool_frames != expected_tools:
            raise RuntimeError(f"{mode} tool-frame mismatch")

    print("cuRobo configuration valid")
    print(f"DOF: {len(actual)}")
    print(f"Joints: {actual}")
    print("Tool frames: l_tool0, r_tool0")
    print("Planning modes: left (7 DOF), right (7 DOF), dual (14 DOF)")


if __name__ == "__main__":
    main()
