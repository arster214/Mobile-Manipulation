#!/usr/bin/env python3
"""Author three D435-like virtual cameras into the RM75 Isaac Sim scene.

The STL/URDF camera geometry is only the visual model.  Isaac Sim image/depth
data comes from USD ``Camera`` prims, so this script creates one camera under
the existing head camera link and one under each wrist tool frame.

Run with Isaac Sim's Python, for example:

    ./python.sh /home/fang/isaacsim_rm75/scripts/add_rm75_d435_cameras.py

If a preview points backwards, edit that camera's axes in
``D435_CAMERA_POSES`` below. The ``--forward-axis`` and ``--up-axis`` options
can be used to override all three cameras for diagnostics.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "assets" / "usd" / "scenes" / "rm75_simple_obstacles.usda"


@dataclass(frozen=True)
class CameraPose:
    """A camera mount relative to an existing robot link/site prim."""

    name: str
    parent_link_name: str
    camera_prim_name: str
    # Relative position of the USD camera optical frame in the parent frame.
    # Keep this at zero first, then adjust to the real D435 optical-center
    # extrinsic if you have calibration data.
    xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Optional extra Euler rotation in degrees after the --forward-axis
    # alignment.  Useful for small calibration corrections.
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    forward_axis: str = "-Y"
    up_axis: str = "+Z"


# These are the three logical D435 sensors on this RM75 model.  The head one
# attaches to the existing camera_link mesh.  The wrist cameras attach to the
# imported tool sites; if your CAD has a more exact wrist camera plate/link,
# replace parent_link_name with that prim's name and/or fill xyz_m/rpy_deg with
# the measured extrinsic.
D435_CAMERA_POSES = (
    CameraPose("head_d435", "camera_link", "head_camera"),
    CameraPose(
        "left_wrist_d435",
        "l_tool0",
        "left_camera",
        xyz_m=(0.0, 0.0, 0.15),
        forward_axis="+Z",
        up_axis="-X",
    ),
    CameraPose(
        "right_wrist_d435",
        "r_tool0",
        "right_camera",
        xyz_m=(0.0, 0.0, 0.15),
        forward_axis="+Z",
        up_axis="-X",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE,
        help=f"USD scene to edit. Default: {DEFAULT_SCENE}",
    )
    parser.add_argument(
        "--robot-root",
        default="/World/Robot",
        help="Root prim of the robot reference inside the composed scene.",
    )
    parser.add_argument(
        "--forward-axis",
        default=None,
        choices=("+X", "-X", "+Y", "-Y", "+Z", "-Z"),
        help=(
            "Override the parent-link viewing axis for every camera. USD "
            "cameras look down local -Z; by default each mount uses its "
            "configured axis."
        ),
    )
    parser.add_argument(
        "--up-axis",
        default=None,
        choices=("+X", "-X", "+Y", "-Y", "+Z", "-Z"),
        help="Override the image-up axis for every camera.",
    )
    parser.add_argument(
        "--image-roll-deg",
        type=float,
        default=0.0,
        help=(
            "Optional calibration roll around the camera optical axis. "
            "Default 0 keeps image up aligned with --up-axis."
        ),
    )
    parser.add_argument("--width", type=int, default=640, help="Nominal image width.")
    parser.add_argument("--height", type=int, default=480, help="Nominal image height.")
    parser.add_argument(
        "--horizontal-fov-deg",
        type=float,
        default=87.0,
        help="Approximate D435 depth horizontal FOV used to set USD focal length.",
    )
    parser.add_argument(
        "--near",
        type=float,
        default=0.05,
        help="Near clipping distance in meters.",
    )
    parser.add_argument(
        "--far",
        type=float,
        default=10.0,
        help="Far clipping distance in meters.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Open the stage and print target paths, but do not save edits.",
    )
    return parser.parse_args()


def _axis_to_tuple(axis: str) -> tuple[float, float, float]:
    sign = 1.0 if axis[0] == "+" else -1.0
    dim = axis[1].upper()
    if dim == "X":
        return (sign, 0.0, 0.0)
    if dim == "Y":
        return (0.0, sign, 0.0)
    if dim == "Z":
        return (0.0, 0.0, sign)
    raise ValueError(f"Unsupported axis: {axis}")


def _dot(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(v, v))


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    length = _norm(v)
    if length < 1e-12:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return tuple(x / length for x in v)  # type: ignore[return-value]


def _quat_from_matrix3(m: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]):
    """Convert a row-major 3x3 rotation matrix to (w, x, y, z)."""

    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)


def _camera_align_quat(forward_axis: str, up_axis: str, image_roll_deg: float):
    """Return Gf.Quatf that maps a USD Camera to the requested parent axes.

    USD cameras look along local -Z, with local +Y as image up.  We build a
    local-to-parent rotation whose columns are parent-space local X/Y/Z.
    """

    from pxr import Gf

    forward = _normalize(_axis_to_tuple(forward_axis))
    up_hint = _normalize(_axis_to_tuple(up_axis))
    if abs(_dot(forward, up_hint)) > 0.999:
        raise ValueError("--forward-axis and --up-axis cannot be parallel")

    local_z_in_parent = tuple(-x for x in forward)
    local_y_in_parent = _normalize(
        tuple(up_hint[i] - _dot(up_hint, local_z_in_parent) * local_z_in_parent[i] for i in range(3))
    )
    local_x_in_parent = _normalize(_cross(local_y_in_parent, local_z_in_parent))
    if image_roll_deg:
        theta = math.radians(image_roll_deg)
        c = math.cos(theta)
        s = math.sin(theta)
        rolled_x = tuple(c * local_x_in_parent[i] + s * local_y_in_parent[i] for i in range(3))
        rolled_y = tuple(-s * local_x_in_parent[i] + c * local_y_in_parent[i] for i in range(3))
        local_x_in_parent = _normalize(rolled_x)
        local_y_in_parent = _normalize(rolled_y)

    # Row-major matrix with local-to-parent basis vectors as columns.
    matrix = (
        (local_x_in_parent[0], local_y_in_parent[0], local_z_in_parent[0]),
        (local_x_in_parent[1], local_y_in_parent[1], local_z_in_parent[1]),
        (local_x_in_parent[2], local_y_in_parent[2], local_z_in_parent[2]),
    )
    qw, qx, qy, qz = _quat_from_matrix3(matrix)
    return Gf.Quatf(float(qw), float(qx), float(qy), float(qz))


def _find_robot_child_by_name(stage, robot_root: str, name: str):
    root = stage.GetPrimAtPath(robot_root)
    if not root.IsValid():
        raise RuntimeError(f"Robot root prim does not exist: {robot_root}")
    matches = [
        prim
        for prim in stage.Traverse()
        if prim.GetPath().pathString.startswith(robot_root + "/") and prim.GetName() == name
    ]
    if not matches:
        raise RuntimeError(f"Cannot find prim named {name!r} below {robot_root}")
    if len(matches) > 1:
        print(f"WARNING: found multiple prims named {name!r}; using {matches[0].GetPath()}")
    return matches[0]


def _clear_xform_ops(prim) -> None:
    from pxr import UsdGeom

    UsdGeom.Xformable(prim).ClearXformOpOrder()


def _author_camera(stage, parent_prim, pose: CameraPose, args: argparse.Namespace) -> None:
    from pxr import Gf, Sdf, UsdGeom

    mount_path = parent_prim.GetPath().AppendChild(pose.name)
    camera_path = mount_path.AppendChild(pose.camera_prim_name)
    legacy_camera_path = mount_path.AppendChild("Camera")

    for path in (legacy_camera_path, camera_path):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            stage.RemovePrim(path)

    mount = UsdGeom.Xform.Define(stage, mount_path)
    _clear_xform_ops(mount.GetPrim())
    mount_xf = UsdGeom.Xformable(mount.GetPrim())
    mount_xf.AddTranslateOp().Set(Gf.Vec3d(*pose.xyz_m))
    # Use rotateXYZ for human-editable calibration deltas.
    mount_xf.AddRotateXYZOp().Set(Gf.Vec3f(*pose.rpy_deg))

    camera = UsdGeom.Camera.Define(stage, camera_path)
    _clear_xform_ops(camera.GetPrim())
    cam_xf = UsdGeom.Xformable(camera.GetPrim())
    forward_axis = args.forward_axis or pose.forward_axis
    up_axis = args.up_axis or pose.up_axis
    cam_xf.AddOrientOp().Set(_camera_align_quat(forward_axis, up_axis, args.image_roll_deg))

    # USD stores aperture/focal length in tenths of a world unit by convention
    # for Camera; in Isaac Sim this is commonly treated as millimeters.  The
    # absolute aperture is less important than the aperture/focal ratio.
    horizontal_aperture_mm = 20.955
    focal_length_mm = horizontal_aperture_mm / (2.0 * math.tan(math.radians(args.horizontal_fov_deg) / 2.0))
    vertical_aperture_mm = horizontal_aperture_mm * (args.height / args.width)

    camera.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)
    camera.CreateHorizontalApertureAttr().Set(horizontal_aperture_mm)
    camera.CreateVerticalApertureAttr().Set(vertical_aperture_mm)
    camera.CreateFocalLengthAttr().Set(focal_length_mm)
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(args.near, args.far))

    prim = camera.GetPrim()
    prim.CreateAttribute("rm75:camera:model", Sdf.ValueTypeNames.String).Set("Intel RealSense D435-like")
    prim.CreateAttribute("rm75:camera:width", Sdf.ValueTypeNames.Int).Set(args.width)
    prim.CreateAttribute("rm75:camera:height", Sdf.ValueTypeNames.Int).Set(args.height)
    prim.CreateAttribute("rm75:camera:forwardAxis", Sdf.ValueTypeNames.String).Set(forward_axis)
    prim.CreateAttribute("rm75:camera:upAxis", Sdf.ValueTypeNames.String).Set(up_axis)
    prim.CreateAttribute("rm75:camera:depthAnnotator", Sdf.ValueTypeNames.String).Set("distance_to_image_plane")

    print(f"{pose.name}: {camera_path} parented to {parent_prim.GetPath()}")


def main() -> None:
    args = _parse_args()
    simulation_app = SimulationApp({"headless": True})
    try:
        import omni.usd

        if not omni.usd.get_context().open_stage(str(args.scene)):
            raise RuntimeError(f"Failed to open Stage: {args.scene}")

        # Give references/payloads a moment to compose before Traverse().
        for _ in range(8):
            simulation_app.update()

        stage = omni.usd.get_context().get_stage()
        for pose in D435_CAMERA_POSES:
            parent = _find_robot_child_by_name(stage, args.robot_root, pose.parent_link_name)
            _author_camera(stage, parent, pose, args)

        if args.dry_run:
            print("Dry-run requested; not saving USD edits.")
        else:
            stage.GetRootLayer().Save()
            print(f"Saved virtual D435 cameras into: {args.scene}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
