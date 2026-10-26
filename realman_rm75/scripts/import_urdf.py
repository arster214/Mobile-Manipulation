#!/usr/bin/env python3
"""Import the RM75 + RMG24 simulation URDF with the Isaac Sim 6 importer."""

from pathlib import Path

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
simulation_app = SimulationApp({"headless": True})

from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402
from pxr import Usd  # noqa: E402


SCENE = ROOT / "assets" / "usd" / "scenes" / "rm75_simple_obstacles.usda"


def _update_scene_robot_reference(output: str) -> None:
    """Point the launch scene at the package created by this import."""
    stage = Usd.Stage.Open(str(SCENE))
    if stage is None:
        raise RuntimeError(f"Failed to open launch scene: {SCENE}")
    robot = stage.GetPrimAtPath("/World/Robot")
    if not robot.IsValid():
        raise RuntimeError(f"Launch scene has no /World/Robot prim: {SCENE}")

    relative_output = Path(output).relative_to(SCENE.parent.parent)
    references = robot.GetReferences()
    references.ClearReferences()
    references.AddReference(f"../{relative_output.as_posix()}")
    stage.GetRootLayer().Save()
    print(f"Updated scene robot reference: {relative_output}")


def main() -> None:
    config = URDFImporterConfig(
        urdf_path=str(ROOT / "assets" / "urdf" / "rm75_dual_rmg24_sim.urdf"),
        usd_path=str(ROOT / "assets" / "usd"),
        merge_fixed_joints=False,
        merge_mesh=False,
        collision_from_visuals=False,
        collision_type="Convex Decomposition",
        allow_self_collision=True,
        fix_base=True,
        joint_drive_type="force",
        joint_target_type="position",
        override_joint_stiffness=400.0,
        override_joint_damping=40.0,
    )
    output = URDFImporter(config).import_urdf()
    _update_scene_robot_reference(output)
    print(f"Imported USD: {output}")


try:
    main()
finally:
    simulation_app.close()
