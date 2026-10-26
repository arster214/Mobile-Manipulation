#!/usr/bin/env python3
"""Launch Isaac Sim as the cuRobo backend for the ROS2/RViz frontend.

Transport note: this script uses a localhost JSON socket instead of an
Isaac-side rclpy.Node because the tested Isaac Sim 6 internal Humble rclpy
crashes when constructing a Node in this environment.
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import os
import socket
import sys

import yaml

from rm75_obstacles import DEFAULT_CONFIG as DEFAULT_OBSTACLE_CONFIG, load_obstacle_scene
from rm75_process_guard import ensure_single_instance

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets" / "usd" / "scenes" / "rm75_simple_obstacles.usda"
SAFE_MODE = os.environ.get("RM75_SAFE_MODE", "G").strip().upper()
SAFE_MODE_RANK = {name: index for index, name in enumerate("ABCDEFG")}
SAFE_RANK = SAFE_MODE_RANK.get(SAFE_MODE, SAFE_MODE_RANK["G"])
ENABLE_IK_PREVIEW = os.environ.get("RM75_ENABLE_IK_PREVIEW", "0") == "1"
BACKEND_HOST = os.environ.get("RM75_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("RM75_BACKEND_PORT", "50075"))
LOCK_PATH = Path(os.environ.get("RM75_BACKEND_LOCK", "/tmp/rm75_isaac_backend.lock"))
ARM_DRIVE_STIFFNESS = float(os.environ.get("RM75_ARM_DRIVE_STIFFNESS", "1000"))
ARM_DRIVE_DAMPING = float(os.environ.get("RM75_ARM_DRIVE_DAMPING", "100"))
ARM_DRIVE_MAX_FORCE = float(os.environ.get("RM75_ARM_DRIVE_MAX_FORCE", "1000"))
HEAD_DRIVE_STIFFNESS = float(os.environ.get("RM75_HEAD_DRIVE_STIFFNESS", "200"))
HEAD_DRIVE_DAMPING = float(os.environ.get("RM75_HEAD_DRIVE_DAMPING", "20"))
HEAD_DRIVE_MAX_FORCE = float(os.environ.get("RM75_HEAD_DRIVE_MAX_FORCE", "50"))
PLATFORM_DRIVE_STIFFNESS = float(os.environ.get("RM75_PLATFORM_DRIVE_STIFFNESS", "1500"))
PLATFORM_DRIVE_DAMPING = float(os.environ.get("RM75_PLATFORM_DRIVE_DAMPING", "200"))
PLATFORM_DRIVE_MAX_FORCE = float(os.environ.get("RM75_PLATFORM_DRIVE_MAX_FORCE", "500"))
PLATFORM_INITIAL_POSITION = float(os.environ.get("RM75_PLATFORM_INITIAL_POSITION", "0.5"))
RENDERER = os.environ.get("RM75_RENDERER", "RayTracedLighting")
DISABLE_SELF_COLLISIONS = os.environ.get("RM75_DISABLE_SELF_COLLISIONS", "0") != "0"

ensure_single_instance(["launch_ros2_curobo_scene.py"], ROOT)


def _backend_socket_is_alive() -> bool:
    try:
        with socket.create_connection((BACKEND_HOST, BACKEND_PORT), timeout=0.3) as sock:
            sock.settimeout(0.5)
            sock.sendall(b'{"command":"get_state"}\n')
            data = b""
            while not data.endswith(b"\n"):
                chunk = sock.recv(65536)
                if not chunk:
                    break
                data += chunk
        return bool(data and json.loads(data.decode("utf-8")).get("success"))
    except Exception:
        return False


LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
_lock_file = LOCK_PATH.open("w")
try:
    fcntl.flock(_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    if _backend_socket_is_alive():
        print(
            f"Refusing to start a second Isaac backend: existing backend is alive at "
            f"{BACKEND_HOST}:{BACKEND_PORT}.",
            file=sys.stderr,
        )
    else:
        print(
            f"Refusing to start Isaac backend: lock is held at {LOCK_PATH}. "
            "If this is stale, confirm no Isaac backend is running, then remove it.",
            file=sys.stderr,
        )
    sys.exit(3)

if _backend_socket_is_alive() and os.environ.get("RM75_FORCE_START_ISAAC") != "1":
    print(
        f"Refusing to start Isaac backend: socket {BACKEND_HOST}:{BACKEND_PORT} "
        "already serves an RM75 backend. Set RM75_FORCE_START_ISAAC=1 only if you are sure.",
        file=sys.stderr,
    )
    sys.exit(3)

from isaacsim import SimulationApp

headless = "--headless" in sys.argv
simulation_app = SimulationApp({"headless": headless, "renderer": RENDERER})

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402


def _find_robot_link(stage, name: str):
    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith("/World/Robot/") and prim.GetName() == name:
            return prim
    return None


def _synchronize_obstacles_from_config() -> None:
    """Author the canonical YAML cuboids into the live Isaac stage."""
    from pxr import Gf, UsdGeom, UsdPhysics

    obstacle_scene = load_obstacle_scene()
    if obstacle_scene.frame_id != "base_link_underpan":
        raise RuntimeError(
            "Isaac obstacle synchronization currently requires frame_id "
            f"'base_link_underpan', got {obstacle_scene.frame_id!r}"
        )

    stage = omni.usd.get_context().get_stage()
    root_path = "/World/Obstacles"
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    configured_names = {item.name for item in obstacle_scene.cuboids}

    for child in root.GetChildren():
        if child.GetName() not in configured_names:
            child.SetActive(False)

    for item in obstacle_scene.cuboids:
        cube = UsdGeom.Cube.Define(stage, f"{root_path}/{item.name}")
        cube.CreateSizeAttr(1.0).Set(1.0)
        cube.CreateDisplayColorAttr().Set([Gf.Vec3f(*item.color_rgba[:3])])

        xformable = UsdGeom.Xformable(cube.GetPrim())
        ordered_ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
        translate = ordered_ops.get("xformOp:translate") or xformable.AddTranslateOp()
        orient = ordered_ops.get("xformOp:orient") or xformable.AddOrientOp()
        scale = ordered_ops.get("xformOp:scale") or xformable.AddScaleOp()
        translate.Set(Gf.Vec3d(*item.position))
        qw, qx, qy, qz = item.quaternion_wxyz
        orient.Set(Gf.Quatf(qw, qx, qy, qz))
        scale.Set(Gf.Vec3f(*item.dims))
        xformable.SetXformOpOrder([translate, orient, scale])

        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        cube.GetPrim().SetCustomDataByKey("rm75:source", str(DEFAULT_OBSTACLE_CONFIG))

    print(
        f"Synchronized {len(obstacle_scene.cuboids)} Isaac obstacles from "
        f"{DEFAULT_OBSTACLE_CONFIG} in frame {obstacle_scene.frame_id}; "
        f"fingerprint={obstacle_scene.fingerprint}"
    )


def _synchronize_initial_joint_state() -> None:
    """Match Isaac articulation state to the complete cuRobo initial state."""
    import numpy as np
    from isaacsim.core.experimental.prims import Articulation

    config_path = ROOT / "config" / "rm75_dual.yml"
    config = yaml.safe_load(config_path.read_text())
    kinematics = config["kinematics"]
    cspace = kinematics["cspace"]
    initial_joints = dict(kinematics["lock_joints"])
    initial_joints["platform_joint"] = PLATFORM_INITIAL_POSITION
    initial_joints.update(zip(cspace["joint_names"], cspace["default_joint_position"]))
    articulation = Articulation("/World/Robot")
    dof_names = list(articulation.dof_names)
    names = [name for name in initial_joints if name in dof_names]
    indices = [dof_names.index(name) for name in names]
    positions = np.array([[initial_joints[name] for name in names]], dtype=np.float32)
    articulation.set_dof_positions(positions, dof_indices=indices)
    articulation.set_dof_position_targets(positions, dof_indices=indices)
    print(f"Synchronized cuRobo initial joints: {names}")


def _configure_drive(prim, drive_name: str, stiffness: float, damping: float, max_force: float) -> None:
    """Create/update a USD PhysicsDriveAPI on a joint prim."""
    from pxr import UsdPhysics

    drive = UsdPhysics.DriveAPI.Get(prim, drive_name)
    if not drive:
        drive = UsdPhysics.DriveAPI.Apply(prim, drive_name)
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(stiffness))
    drive.CreateDampingAttr(float(damping))
    drive.CreateMaxForceAttr(float(max_force))


def _configure_joint_drives() -> None:
    """Make Isaac articulation drives stiff enough to hold against gravity.

    The URDF importer generated angular stiffness around 6.98 and damping around
    0.70 for the RM75 joints, with wrist maxForce as low as 10. That is too soft
    for gravity-holding in this simulation. Patch the composed stage at runtime
    instead of editing generated USD assets.
    """
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    arm_names = {f"l_joint{i}" for i in range(1, 8)} | {f"r_joint{i}" for i in range(1, 8)}
    patched = []
    for prim in stage.Traverse():
        name = prim.GetName()
        if name in arm_names and prim.IsA(UsdPhysics.RevoluteJoint):
            _configure_drive(prim, "angular", ARM_DRIVE_STIFFNESS, ARM_DRIVE_DAMPING, ARM_DRIVE_MAX_FORCE)
            patched.append(name)
        elif name in {"head_joint1", "head_joint2"} and prim.IsA(UsdPhysics.RevoluteJoint):
            _configure_drive(prim, "angular", HEAD_DRIVE_STIFFNESS, HEAD_DRIVE_DAMPING, HEAD_DRIVE_MAX_FORCE)
            patched.append(name)
        elif name == "platform_joint" and prim.IsA(UsdPhysics.PrismaticJoint):
            _configure_drive(prim, "linear", PLATFORM_DRIVE_STIFFNESS, PLATFORM_DRIVE_DAMPING, PLATFORM_DRIVE_MAX_FORCE)
            patched.append(name)

    print(
        "Configured joint drives: "
        f"arm(k={ARM_DRIVE_STIFFNESS}, d={ARM_DRIVE_DAMPING}, f={ARM_DRIVE_MAX_FORCE}), "
        f"head(k={HEAD_DRIVE_STIFFNESS}, d={HEAD_DRIVE_DAMPING}, f={HEAD_DRIVE_MAX_FORCE}), "
        f"platform(k={PLATFORM_DRIVE_STIFFNESS}, d={PLATFORM_DRIVE_DAMPING}, f={PLATFORM_DRIVE_MAX_FORCE}); "
        f"patched={sorted(patched)}"
    )


def _configure_robot_self_collisions() -> None:
    """Keep robot self-collision and disable only head_link1 physics shapes."""
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    changed = []
    robot = stage.GetPrimAtPath("/World/Robot")
    for prim in Usd.PrimRange(robot):
        attribute = prim.GetAttribute("newton:selfCollisionEnabled")
        if attribute and attribute.IsValid():
            attribute.Set(not DISABLE_SELF_COLLISIONS)
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


def _install_tcp_goal_axes() -> None:
    """Add small static TCP axes to help compare targets and current TCP in Isaac."""
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
        for axis_name, color, translation, orientation in axes:
            cylinder = UsdGeom.Cylinder.Define(stage, visual_path.AppendChild(axis_name))
            cylinder.CreateRadiusAttr(axis_radius)
            cylinder.CreateHeightAttr(axis_length)
            cylinder.CreateDisplayColorAttr([color])
            xformable = UsdGeom.Xformable(cylinder.GetPrim())
            xformable.AddTranslateOp().Set(translation)
            xformable.AddOrientOp().Set(orientation)


if not omni.usd.get_context().open_stage(str(SCENE)):
    simulation_app.close()
    raise RuntimeError(f"Failed to open Stage: {SCENE}")

for _ in range(8):
    simulation_app.update()

_synchronize_obstacles_from_config()
_configure_joint_drives()
_configure_robot_self_collisions()

if SAFE_RANK >= SAFE_MODE_RANK["D"] and ENABLE_IK_PREVIEW:
    app_utils.enable_extension("isaacsim.robot.poser")
    for _ in range(4):
        simulation_app.update()

timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(2):
    simulation_app.update()
_synchronize_initial_joint_state()
for _ in range(2):
    simulation_app.update()
if os.environ.get("RM75_INSTALL_TCP_AXES", "0") == "1":
    _install_tcp_goal_axes()

from rm75_planner import RM75MotionPlanner  # noqa: E402
from rm75_socket_goal_server import RM75SocketGoalServer  # noqa: E402
from rm75_joint_lock import RM75ArmJointLock  # noqa: E402


class _DisabledMotionPlanner:
    """Lightweight planner facade for SAFE_MODE levels before planning is enabled."""

    is_executing = False
    has_plan = False

    def __init__(self) -> None:
        self._status_callback = None

    def set_status_callback(self, callback) -> None:
        self._status_callback = callback

    def plan_to_poses(self, mode, pose_by_frame):
        del mode, pose_by_frame
        if self._status_callback is not None:
            self._status_callback("planning disabled by SAFE_MODE")
        return None

    def execute(self) -> bool:
        if self._status_callback is not None:
            self._status_callback("execution disabled by SAFE_MODE")
        return False

    def clear_plan(self) -> None:
        if self._status_callback is not None:
            self._status_callback("Plan cleared")

    def update(self) -> None:
        pass

    def destroy(self) -> None:
        pass


stage = omni.usd.get_context().get_stage()
robot = stage.GetPrimAtPath("/World/Robot")
if SAFE_RANK >= SAFE_MODE_RANK["F"]:
    motion_planner = RM75MotionPlanner(stage, robot, build_window=not headless)
else:
    motion_planner = _DisabledMotionPlanner()
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
goal_server = RM75SocketGoalServer(
    stage,
    robot,
    motion_planner,
    host=BACKEND_HOST,
    port=BACKEND_PORT,
    enable_planning=SAFE_RANK >= SAFE_MODE_RANK["F"],
    enable_execution=SAFE_RANK >= SAFE_MODE_RANK["G"],
    enable_ik_preview=SAFE_RANK >= SAFE_MODE_RANK["D"] and ENABLE_IK_PREVIEW,
)

print(f"Opened cuRobo backend Stage: {SCENE}")
print(f"Socket backend: {BACKEND_HOST}:{BACKEND_PORT}")
print(
    f"SAFE_MODE={SAFE_MODE} "
    f"ik_preview={SAFE_RANK >= SAFE_MODE_RANK['D'] and ENABLE_IK_PREVIEW} "
    f"planning={SAFE_RANK >= SAFE_MODE_RANK['F']} "
    f"execution={SAFE_RANK >= SAFE_MODE_RANK['G']}"
)
print("ROS2/RViz frontend still uses /rm75/* topics/services via scripts/rm75_rviz_interactive_markers.py")

frame = 0
test_frames = int(os.environ.get("RM75_TEST_FRAMES", "0"))
try:
    while simulation_app.is_running():
        simulation_app.update()
        goal_server.update()
        motion_planner.update()
        arm_lock.update(motion_planner.is_executing)
        frame += 1
        if test_frames and frame >= test_frames:
            break
finally:
    goal_server.destroy()
    arm_lock.destroy()
    motion_planner.destroy()
    simulation_app.close()
