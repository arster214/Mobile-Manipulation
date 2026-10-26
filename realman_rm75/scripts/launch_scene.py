#!/usr/bin/env python3
"""Launch Isaac Sim GUI and open the RM75 + obstacle scene."""

from pathlib import Path
import os
import sys

import yaml

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets" / "usd" / "scenes" / "rm75_simple_obstacles.usda"

headless = "--headless" in sys.argv
test_frames = int(os.environ.get("RM75_TEST_FRAMES", "0"))
diagnose_head = os.environ.get("RM75_DIAG_HEAD", "0") == "1"
diagnose_head_frames = int(os.environ.get("RM75_DIAG_HEAD_FRAMES", "120"))
enable_isaac_ghost = os.environ.get("RM75_ENABLE_ISAAC_GHOST", "0") == "1"
ARM_DRIVE_STIFFNESS = float(os.environ.get("RM75_ARM_DRIVE_STIFFNESS", "1000"))
ARM_DRIVE_DAMPING = float(os.environ.get("RM75_ARM_DRIVE_DAMPING", "200"))
ARM_DRIVE_MAX_FORCE = float(os.environ.get("RM75_ARM_DRIVE_MAX_FORCE", "1000"))
HEAD_DRIVE_STIFFNESS = float(os.environ.get("RM75_HEAD_DRIVE_STIFFNESS", "200"))
HEAD_DRIVE_DAMPING = float(os.environ.get("RM75_HEAD_DRIVE_DAMPING", "20"))
HEAD_DRIVE_MAX_FORCE = float(os.environ.get("RM75_HEAD_DRIVE_MAX_FORCE", "50"))
PLATFORM_DRIVE_STIFFNESS = float(os.environ.get("RM75_PLATFORM_DRIVE_STIFFNESS", "1500"))
PLATFORM_DRIVE_DAMPING = float(os.environ.get("RM75_PLATFORM_DRIVE_DAMPING", "200"))
PLATFORM_DRIVE_MAX_FORCE = float(os.environ.get("RM75_PLATFORM_DRIVE_MAX_FORCE", "500"))
RENDERER = os.environ.get("RM75_RENDERER", "RayTracedLighting")
DISABLE_SELF_COLLISIONS = os.environ.get("RM75_DISABLE_SELF_COLLISIONS", "0") != "0"
from rm75_process_guard import ensure_single_instance

ensure_single_instance(["launch_scene.py"], ROOT)
simulation_app = SimulationApp({"headless": headless, "renderer": RENDERER})

import omni.usd  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.timeline  # noqa: E402


if not omni.usd.get_context().open_stage(str(SCENE)):
    simulation_app.close()
    raise RuntimeError(f"Failed to open Stage: {SCENE}")

def _find_robot_link(stage, name: str):
    """Find a link by its authored name below the composed robot reference."""
    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith("/World/Robot/") and prim.GetName() == name:
            return prim
    return None


def _install_robot_posers() -> None:
    """Create two Isaac Robot Poser named poses for the RM75 arms."""
    from isaacsim.robot.poser import robot_poser
    from isaacsim.robot.poser.ui.utils.fk_helpers import build_pose_from_current_joints

    stage = omni.usd.get_context().get_stage()
    robot = stage.GetPrimAtPath("/World/Robot")
    if not robot.IsValid() or not robot.HasAPI("IsaacRobotAPI"):
        raise RuntimeError("/World/Robot does not expose IsaacRobotAPI")

    for pose_name, base_name, tool_name in (
        ("left_arm_poser", "l_base_link1", "l_tool0"),
        ("right_arm_poser", "r_base_link1", "r_tool0"),
    ):
        start = _find_robot_link(stage, base_name)
        end = _find_robot_link(stage, tool_name)
        if start is None or end is None:
            raise RuntimeError(f"Cannot find Robot Poser chain: {base_name} -> {tool_name}")
        result = build_pose_from_current_joints(stage, robot, str(start.GetPath()), str(end.GetPath()))
        if result is None or not result.success:
            raise RuntimeError(f"Failed to initialize Robot Poser: {pose_name}")
        robot_poser.store_named_pose(stage, robot, pose_name, result)
        print(f"Initialized Robot Poser: {pose_name} ({base_name} -> {tool_name})")


def _synchronize_initial_joint_state() -> None:
    """Match Isaac articulation state to the complete cuRobo initial state."""
    import numpy as np
    from isaacsim.core.experimental.prims import Articulation

    config_path = ROOT / "config" / "rm75_dual.yml"
    config = yaml.safe_load(config_path.read_text())
    kinematics = config["kinematics"]
    cspace = kinematics["cspace"]
    initial_joints = dict(kinematics["lock_joints"])
    initial_joints.update(
        zip(cspace["joint_names"], cspace["default_joint_position"])
    )
    articulation = Articulation("/World/Robot")
    dof_names = list(articulation.dof_names)
    names = [name for name in initial_joints if name in dof_names]
    indices = [dof_names.index(name) for name in names]
    positions = np.array([[initial_joints[name] for name in names]], dtype=np.float32)
    articulation.set_dof_positions(positions, dof_indices=indices)
    articulation.set_dof_position_targets(positions, dof_indices=indices)
    print(f"Synchronized cuRobo initial joints: {names}")


def _configure_head_drives() -> None:
    """Increase head joint gravity-hold gains in the composed runtime stage."""
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    patched = []
    platform_patched = []
    for prim in stage.Traverse():
        if prim.GetName() in {"head_joint1", "head_joint2"} and prim.IsA(UsdPhysics.RevoluteJoint):
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(HEAD_DRIVE_STIFFNESS)
            drive.CreateDampingAttr(HEAD_DRIVE_DAMPING)
            drive.CreateMaxForceAttr(HEAD_DRIVE_MAX_FORCE)
            patched.append(prim.GetName())
        elif prim.GetName() == "platform_joint" and prim.IsA(UsdPhysics.PrismaticJoint):
            drive = UsdPhysics.DriveAPI.Get(prim, "linear")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(PLATFORM_DRIVE_STIFFNESS)
            drive.CreateDampingAttr(PLATFORM_DRIVE_DAMPING)
            drive.CreateMaxForceAttr(PLATFORM_DRIVE_MAX_FORCE)
            platform_patched.append(prim.GetName())

    if sorted(patched) != ["head_joint1", "head_joint2"]:
        raise RuntimeError(f"Expected both head joints, patched={sorted(patched)}")
    print(
        "Configured head drives: "
        f"stiffness={HEAD_DRIVE_STIFFNESS}, damping={HEAD_DRIVE_DAMPING}, "
        f"maxForce={HEAD_DRIVE_MAX_FORCE}; "
        "platform drives: "
        f"stiffness={PLATFORM_DRIVE_STIFFNESS}, damping={PLATFORM_DRIVE_DAMPING}, "
        f"maxForce={PLATFORM_DRIVE_MAX_FORCE}"
    )
    if platform_patched != ["platform_joint"]:
        raise RuntimeError(f"Expected platform_joint, patched={platform_patched}")


def _configure_robot_self_collisions() -> None:
    """Keep robot self-collision and disable only head_link1 physics shapes."""
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    changed = []
    robot = stage.GetPrimAtPath("/World/Robot")
    for prim in Usd.PrimRange(robot):
        attribute = prim.GetAttribute("newton:selfCollisionEnabled")
        if attribute and attribute.IsValid():
            attribute.Set(False if DISABLE_SELF_COLLISIONS else True)
            changed.append(str(prim.GetPath()))
    if not changed:
        raise RuntimeError("Could not find robot self-collision setting")
    if DISABLE_SELF_COLLISIONS:
        print(f"Disabled robot self-collisions by override: {changed}")
        return

    head_collision_root = stage.GetPrimAtPath(
        "/World/Robot/Geometry/base_link_underpan/body_base_link/platform_base_link/head_link1/head_link1_1"
    )
    if not head_collision_root.IsValid():
        raise RuntimeError("Could not find head_link1 collision root")
    shapes = [
        prim
        for prim in Usd.PrimRange(head_collision_root, Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not shapes:
        raise RuntimeError("Could not find head_link1 collision shapes")
    for shape in shapes:
        collision_enabled = shape.GetAttribute("physics:collisionEnabled")
        if collision_enabled and collision_enabled.IsValid():
            collision_enabled.Set(False)
    print(
        "Enabled robot self-collisions; disabled head_link1 collision shapes: "
        f"{[str(shape.GetPath()) for shape in shapes]}"
    )


def _install_tcp_visuals() -> None:
    """Add RGB axes parented to each tool frame as current-TCP references."""
    from math import sqrt

    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    axis_length = 0.06
    axis_radius = 0.0025
    half_sqrt = 1.0 / sqrt(2.0)
    axes = (
        ("X", Gf.Vec3f(1.0, 0.1, 0.1), Gf.Vec3d(axis_length / 2, 0, 0), Gf.Quatf(half_sqrt, 0, half_sqrt, 0)),
        ("Y", Gf.Vec3f(0.1, 1.0, 0.1), Gf.Vec3d(0, axis_length / 2, 0), Gf.Quatf(half_sqrt, -half_sqrt, 0, 0)),
        ("Z", Gf.Vec3f(0.1, 0.3, 1.0), Gf.Vec3d(0, 0, axis_length / 2), Gf.Quatf(1, 0, 0, 0)),
    )

    for tool_name in ("l_tool0", "r_tool0"):
        tool = _find_robot_link(stage, tool_name)
        if tool is None:
            raise RuntimeError(f"Cannot find TCP frame: {tool_name}")
        visual_path = tool.GetPath().AppendChild("Current_TCP_Axes")
        UsdGeom.Xform.Define(stage, visual_path)

        origin = UsdGeom.Sphere.Define(stage, visual_path.AppendChild("Origin"))
        origin.CreateRadiusAttr(0.004)
        origin.CreateDisplayColorAttr([Gf.Vec3f(1.0, 1.0, 1.0)])

        for axis_name, color, translation, orientation in axes:
            cylinder = UsdGeom.Cylinder.Define(stage, visual_path.AppendChild(axis_name))
            cylinder.CreateRadiusAttr(axis_radius)
            cylinder.CreateHeightAttr(axis_length)
            cylinder.CreateDisplayColorAttr([color])
            xformable = UsdGeom.Xformable(cylinder.GetPrim())
            xformable.AddTranslateOp().Set(translation)
            xformable.AddOrientOp().Set(orientation)

        print(f"Installed current TCP visual: {tool_name}")


# Allow referenced USD payloads to resolve before discovering the arm links.
for _ in range(8):
    simulation_app.update()

_configure_head_drives()
_configure_robot_self_collisions()

extension_manager = omni.kit.app.get_app().get_extension_manager()
extension_manager.set_extension_enabled_immediate("isaacsim.robot.poser", True)
extension_manager.set_extension_enabled_immediate("isaacsim.robot.poser.ui", True)
for _ in range(4):
    simulation_app.update()

from rm75_pose_gizmo import install_combined_pose_gizmo  # noqa: E402

pose_gizmo = install_combined_pose_gizmo()

# Let configured drive targets (notably the lift) settle before capturing the
# initial end-effector targets for Robot Poser.
timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(2):
    simulation_app.update()
_synchronize_initial_joint_state()
for _ in range(2):
    simulation_app.update()

head_diagnostic = None
if diagnose_head:
    from isaacsim.core.experimental.prims import Articulation

    head_diagnostic = Articulation("/World/Robot")
    head_dof_indices = {
        name: index
        for index, name in enumerate(head_diagnostic.dof_names)
        if name in {"head_joint1", "head_joint2", "platform_joint"}
    }
    print(
        "Head diagnostic DOFs: "
        f"names={list(head_diagnostic.dof_names)}, indices={head_dof_indices}",
        flush=True,
    )
    if set(head_dof_indices) != {"head_joint1", "head_joint2", "platform_joint"}:
        raise RuntimeError(f"Head diagnostic could not find target joints: {head_dof_indices}")

_install_robot_posers()
_install_tcp_visuals()
for _ in range(4):
    simulation_app.update()
pose_gizmo.refresh()

from rm75_planner import RM75MotionPlanner  # noqa: E402
from rm75_joint_lock import RM75ArmJointLock  # noqa: E402
stage = omni.usd.get_context().get_stage()
robot = stage.GetPrimAtPath("/World/Robot")
motion_planner = RM75MotionPlanner(stage, robot, build_window=not headless)
from isaacsim.core.experimental.prims import Articulation  # noqa: E402

arm_lock = RM75ArmJointLock(
    stage,
    robot,
    Articulation(str(robot.GetPath())),
    stiffness=ARM_DRIVE_STIFFNESS,
    damping=ARM_DRIVE_DAMPING,
    max_force_cap=ARM_DRIVE_MAX_FORCE,
)
arm_lock.lock_current()
ik_preview = None
if enable_isaac_ghost:
    from rm75_ik_preview import RM75IKPreview  # noqa: E402

    ik_preview = RM75IKPreview(stage, robot)
    print("Isaac IK ghost preview enabled: /World/IKPreview")
else:
    print("Isaac IK ghost preview disabled: showing only the real robot")

# Open the built-in Robot Poser window; it will list both named poses.
import omni.ui  # noqa: E402
poser_window = omni.ui.Workspace.get_window("Robot Poser")
if poser_window is not None:
    poser_window.visible = True

print(f"Opened Stage: {SCENE}")
frame = 0
test_plans = [
    mode.strip()
    for mode in os.environ.get("RM75_TEST_PLAN", "").split(",")
    if mode.strip()
]
test_plan_index = 0
test_execute = os.environ.get("RM75_TEST_EXECUTE") == "1"
test_ik_preview = os.environ.get("RM75_TEST_IK_PREVIEW", "").strip()
ik_preview_tested = False
while simulation_app.is_running():
    simulation_app.update()
    if ik_preview is not None:
        ik_preview.update()
    motion_planner.update()
    arm_lock.update(motion_planner.is_executing)
    frame += 1
    if head_diagnostic is not None and frame <= diagnose_head_frames:
        positions = head_diagnostic.get_dof_positions().numpy()[0]
        targets = head_diagnostic.get_dof_position_targets().numpy()[0]
        values = {
            name: (float(positions[index]), float(targets[index]))
            for name, index in head_dof_indices.items()
        }
        print(f"Head diagnostic frame={frame}: position,target={values}", flush=True)
    if test_ik_preview and not ik_preview_tested and frame >= 2:
        import numpy as np
        from isaacsim.core.experimental.prims import Articulation
        from pxr import UsdGeom

        if test_ik_preview not in ("left", "right"):
            raise ValueError("RM75_TEST_IK_PREVIEW must be 'left' or 'right'")

        timeline.pause()
        articulation = Articulation("/World/Robot")
        real_before = articulation.get_dof_positions().numpy().copy()
        prefix = "l" if test_ik_preview == "left" else "r"
        ghost_tool = _find_robot_link(stage, f"{prefix}_tool0")
        ghost_tool = stage.GetPrimAtPath(
            str(ghost_tool.GetPath()).replace("/World/Robot", "/World/IKPreview", 1)
        )
        ghost_before = UsdGeom.Xformable(ghost_tool).ComputeLocalToWorldTransform(0).ExtractTranslation()

        real_start = _find_robot_link(stage, f"{prefix}_base_link1")
        ghost_start = stage.GetPrimAtPath(
            str(real_start.GetPath()).replace("/World/Robot", "/World/IKPreview", 1)
        )
        real_start_position = UsdGeom.Xformable(real_start).ComputeLocalToWorldTransform(0).ExtractTranslation()
        ghost_start_position = UsdGeom.Xformable(ghost_start).ComputeLocalToWorldTransform(0).ExtractTranslation()
        start_alignment_error = float((ghost_start_position - real_start_position).GetLength())
        if start_alignment_error > 1e-6:
            raise RuntimeError(
                f"{test_ik_preview} IK preview base is misaligned: "
                f"{start_alignment_error:.9f} m"
            )
        if ik_preview is None:
            raise RuntimeError("IK preview test requested, but Isaac ghost preview is disabled")
        initial_ik_success = ik_preview.last_solve_succeeded(test_ik_preview)
        print(f"Initial IK preview solve: side={test_ik_preview}, success={initial_ik_success}")

        pose_path = f"/World/Robot/NamedPoses/{test_ik_preview}_arm_poser"
        pose_prim = stage.GetPrimAtPath(pose_path)
        translate_op = next(
            op
            for op in UsdGeom.Xformable(pose_prim).GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
        )
        target = translate_op.Get()
        inward = np.asarray(real_start_position, dtype=np.float64) - np.asarray(
            target, dtype=np.float64
        )
        inward /= np.linalg.norm(inward)
        shifted_target = np.asarray(target, dtype=np.float64) + 0.002 * inward
        translate_op.Set(type(target)(*shifted_target))
        ik_preview.update()
        simulation_app.update()

        ghost_after = UsdGeom.Xformable(ghost_tool).ComputeLocalToWorldTransform(0).ExtractTranslation()
        real_after = articulation.get_dof_positions().numpy().copy()
        ghost_delta = float((ghost_after - ghost_before).GetLength())
        real_delta = float(np.max(np.abs(real_after - real_before)))
        if ik_preview.last_solve_succeeded(test_ik_preview) is not True:
            raise RuntimeError(f"{test_ik_preview} IK preview test failed to solve")
        if ghost_delta <= 1e-5:
            raise RuntimeError(f"{test_ik_preview} IK preview did not move")
        if real_delta > 1e-6:
            raise RuntimeError(f"Real robot moved during IK preview: {real_delta:.9f}")
        print(
            f"IK preview test passed: side={test_ik_preview}, "
            f"base_error={start_alignment_error:.9f} m, "
            f"ghost_delta={ghost_delta:.6f} m, real_joint_delta={real_delta:.9f}"
        )
        ik_preview_tested = True
    if test_plan_index < len(test_plans) and frame >= 2:
        planned = motion_planner.plan(test_plans[test_plan_index])
        test_plan_index += 1
        if planned is not None and test_execute:
            motion_planner.execute()
    if test_frames and frame >= test_frames:
        break

motion_planner.destroy()
arm_lock.destroy()
if ik_preview is not None:
    ik_preview.destroy()
pose_gizmo.destroy()
simulation_app.close()
