#!/usr/bin/env python3
"""RViz2 Interactive Marker front-end for RM75 cuRobo planning.

Run this with the system ROS 2 Humble Python (not Isaac Python):

    python3 scripts/rm75_rviz_interactive_markers.py

The node exposes normal ROS 2 topics/services for RViz and forwards requests to
Isaac/cuRobo through a localhost JSON socket served by
scripts/launch_ros2_curobo_scene.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import socket
import threading
from typing import Callable
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker, MarkerArray


ROOT = Path(__file__).resolve().parents[1]
BASE_FRAME = "base_link_underpan"
NAMESPACE = "rm75_goals"
BACKEND_HOST = os.environ.get("RM75_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("RM75_BACKEND_PORT", "50075"))
OBSTACLE_CONFIG = ROOT / "config" / "simple_obstacles.yml"
URDF_PATH = ROOT / "assets" / "urdf" / "rm75_dual_rmg24_sim.urdf"
SAFE_MODE = os.environ.get("RM75_SAFE_MODE", "G").strip().upper()
ENABLE_IK_PREVIEW = os.environ.get("RM75_ENABLE_IK_PREVIEW", "1") == "1"
SAFE_MODE_RANK = {name: index for index, name in enumerate("ABCDEFG")}
RESET_GOALS_ON_START = os.environ.get("RM75_RESET_GOALS_ON_START", "1") != "0"
IK_PREVIEW_HZ = float(os.environ.get("RM75_IK_PREVIEW_HZ", "8.0"))
IK_PREVIEW_TIMEOUT = float(os.environ.get("RM75_IK_PREVIEW_TIMEOUT", "0.75"))
LOCKED_JOINT_DEFAULTS = {
    "platform_joint": 0.5,
    "head_joint1": 0.0,
    "head_joint2": 0.0,
    "l_Joint_finger1": 0.0325,
    "r_Joint_finger1": 0.0325,
}


@dataclass(frozen=True)
class GoalSpec:
    side: str
    marker_name: str
    topic: str
    state_topic: str
    default_xyz: tuple[float, float, float]
    color: tuple[float, float, float, float]


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]


GOALS = {
    "left": GoalSpec("left", "left_goal", "/rm75/left_goal", "/rm75/left_goal_state", (0.35, 0.35, 0.85), (0.1, 0.45, 1.0, 0.85)),
    "right": GoalSpec("right", "right_goal", "/rm75/right_goal", "/rm75/right_goal_state", (0.35, -0.35, 0.85), (1.0, 0.45, 0.1, 0.85)),
}


def _identity_orientation() -> Quaternion:
    q = Quaternion()
    q.w = 1.0
    return q


def _make_pose(xyz: tuple[float, float, float]) -> Pose:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = xyz
    pose.orientation = _identity_orientation()
    return pose


def _copy_pose(pose: Pose) -> Pose:
    copied = Pose()
    copied.position.x = float(pose.position.x)
    copied.position.y = float(pose.position.y)
    copied.position.z = float(pose.position.z)
    copied.orientation.x = float(pose.orientation.x)
    copied.orientation.y = float(pose.orientation.y)
    copied.orientation.z = float(pose.orientation.z)
    copied.orientation.w = float(pose.orientation.w)
    return copied


def _pose_to_backend(pose: Pose) -> dict[str, list[float]]:
    return {
        "position": [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
        "quaternion_xyzw": [
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        ],
    }


def _backend_to_pose(data: dict) -> Pose:
    pose = Pose()
    position = data.get("position", [0.0, 0.0, 0.0])
    quat = data.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0])
    pose.position.x, pose.position.y, pose.position.z = [float(v) for v in position]
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = [float(v) for v in quat]
    return pose


def _parse_vec(text: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not text:
        return default
    values = [float(value) for value in text.split()]
    if len(values) != 3:
        return default
    return values[0], values[1], values[2]


def _translation_matrix(xyz: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def _rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    transform = np.eye(4)
    transform[:3, :3] = rz @ ry @ rx
    return transform


def _axis_rotation_matrix(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.eye(4)
    x, y, z = vector / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    rot = np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = rot
    return transform


def _axis_translation_matrix(axis: tuple[float, float, float], distance: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(axis, dtype=float) * float(distance)
    return transform


class URDFKinematicPreview:
    """Tiny URDF FK helper used only for visual-only IK preview markers.

    It deliberately does not publish TF or joint states.  We parse the URDF tree
    and draw link-origin points directly in ``base_link_underpan``.
    """

    def __init__(self, urdf_path: Path, root_link: str = BASE_FRAME) -> None:
        self.root_link = root_link
        self._children_by_parent: dict[str, list[JointSpec]] = {}
        self._path_cache: dict[str, list[JointSpec]] = {}
        self._load_urdf(urdf_path)

    def _load_urdf(self, urdf_path: Path) -> None:
        root = ET.parse(urdf_path).getroot()
        for joint in root.findall("joint"):
            parent_elem = joint.find("parent")
            child_elem = joint.find("child")
            if parent_elem is None or child_elem is None:
                continue
            origin = joint.find("origin")
            axis = joint.find("axis")
            spec = JointSpec(
                name=str(joint.attrib["name"]),
                joint_type=str(joint.attrib.get("type", "fixed")),
                parent=str(parent_elem.attrib["link"]),
                child=str(child_elem.attrib["link"]),
                xyz=_parse_vec(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
                rpy=_parse_vec(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
                axis=_parse_vec(axis.attrib.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0)),
            )
            self._children_by_parent.setdefault(spec.parent, []).append(spec)

    def chain_to(self, tip_link: str) -> list[JointSpec]:
        cached = self._path_cache.get(tip_link)
        if cached is not None:
            return cached

        def dfs(link: str, path: list[JointSpec]) -> list[JointSpec] | None:
            if link == tip_link:
                return path
            for joint in self._children_by_parent.get(link, []):
                found = dfs(joint.child, path + [joint])
                if found is not None:
                    return found
            return None

        path = dfs(self.root_link, [])
        if path is None:
            raise RuntimeError(f"URDF chain not found: {self.root_link} -> {tip_link}")
        self._path_cache[tip_link] = path
        return path

    def link_origins(self, tip_link: str, joints: dict[str, float]) -> list[tuple[str, tuple[float, float, float]]]:
        transform = np.eye(4)
        origins: list[tuple[str, tuple[float, float, float]]] = [(self.root_link, (0.0, 0.0, 0.0))]
        for joint in self.chain_to(tip_link):
            transform = transform @ _translation_matrix(joint.xyz) @ _rpy_matrix(joint.rpy)
            value = float(joints.get(joint.name, LOCKED_JOINT_DEFAULTS.get(joint.name, 0.0)))
            if joint.joint_type in ("revolute", "continuous"):
                transform = transform @ _axis_rotation_matrix(joint.axis, value)
            elif joint.joint_type == "prismatic":
                transform = transform @ _axis_translation_matrix(joint.axis, value)
            xyz = transform[:3, 3]
            origins.append((joint.child, (float(xyz[0]), float(xyz[1]), float(xyz[2]))))
        return origins


def _make_axis_control(name: str, orientation: tuple[float, float, float, float], interaction_mode: int) -> InteractiveMarkerControl:
    control = InteractiveMarkerControl()
    control.name = name
    control.orientation.w = orientation[0]
    control.orientation.x = orientation[1]
    control.orientation.y = orientation[2]
    control.orientation.z = orientation[3]
    control.interaction_mode = interaction_mode
    return control


def _make_visual_control(spec: GoalSpec) -> InteractiveMarkerControl:
    marker = Marker()
    marker.type = Marker.SPHERE
    marker.scale.x = 0.14
    marker.scale.y = 0.14
    marker.scale.z = 0.14
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = spec.color

    control = InteractiveMarkerControl()
    control.name = f"{spec.side}_visual"
    control.always_visible = True
    control.interaction_mode = InteractiveMarkerControl.MOVE_ROTATE_3D
    control.markers.append(marker)
    return control


def make_6dof_marker(spec: GoalSpec) -> InteractiveMarker:
    marker = InteractiveMarker()
    marker.header.frame_id = BASE_FRAME
    marker.name = spec.marker_name
    marker.description = f"RM75 {spec.side} TCP goal"
    marker.scale = 0.42
    marker.pose = _make_pose(spec.default_xyz)
    marker.controls.append(_make_visual_control(spec))
    marker.controls.extend(
        [
            _make_axis_control("rotate_x", (1.0, 1.0, 0.0, 0.0), InteractiveMarkerControl.ROTATE_AXIS),
            _make_axis_control("move_x", (1.0, 1.0, 0.0, 0.0), InteractiveMarkerControl.MOVE_AXIS),
            _make_axis_control("rotate_y", (1.0, 0.0, 1.0, 0.0), InteractiveMarkerControl.ROTATE_AXIS),
            _make_axis_control("move_y", (1.0, 0.0, 1.0, 0.0), InteractiveMarkerControl.MOVE_AXIS),
            _make_axis_control("rotate_z", (1.0, 0.0, 0.0, 1.0), InteractiveMarkerControl.ROTATE_AXIS),
            _make_axis_control("move_z", (1.0, 0.0, 0.0, 1.0), InteractiveMarkerControl.MOVE_AXIS),
        ]
    )
    return marker


class RM75InteractiveMarkers(Node):
    """RViz-side goal marker server, ROS services, and socket client."""

    def __init__(self) -> None:
        super().__init__("rm75_interactive_markers")
        self._safe_mode = SAFE_MODE if SAFE_MODE in SAFE_MODE_RANK else "G"
        self._safe_rank = SAFE_MODE_RANK[self._safe_mode]
        self._enable_backend = self._safe_rank >= SAFE_MODE_RANK["B"]
        self._enable_markers = self._safe_rank >= SAFE_MODE_RANK["C"]
        self._enable_ik_preview = self._safe_rank >= SAFE_MODE_RANK["D"] and ENABLE_IK_PREVIEW
        self._enable_planning = self._safe_rank >= SAFE_MODE_RANK["F"]
        self._enable_execution = self._safe_rank >= SAFE_MODE_RANK["G"]
        self.server = InteractiveMarkerServer(self, NAMESPACE)
        self.menu = MenuHandler()
        self.pose_publishers = {side: self.create_publisher(PoseStamped, spec.topic, 10) for side, spec in GOALS.items()}
        self.goal_state_publishers = {side: self.create_publisher(PoseStamped, spec.state_topic, 10) for side, spec in GOALS.items()}
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.status_pub = self.create_publisher(String, "/rm75/status", 10)
        self.trajectory_pub = self.create_publisher(JointTrajectory, "/rm75/planned_trajectory", 10)
        self.trajectory_marker_pub = self.create_publisher(MarkerArray, "/rm75/trajectory_markers", 10)
        self.obstacle_pub = self.create_publisher(MarkerArray, "/rm75/obstacles", 10)
        self.goal_marker_pub = self.create_publisher(MarkerArray, "/rm75/goal_markers", 10)
        self.ik_preview_pub = self.create_publisher(MarkerArray, "/rm75/ik_preview", 10)
        self._obstacle_markers = self._build_obstacle_markers()
        self._fk_preview = URDFKinematicPreview(URDF_PATH)
        self._last_pose_by_marker: dict[str, Pose] = {}
        self._latest_joint_values: dict[str, float] = dict(LOCKED_JOINT_DEFAULTS)
        self._dirty_goal_sides: set[str] = set()
        self._ik_preview_joints: dict[str, dict[str, float]] = {}
        self._ik_preview_pending: dict[str, Pose] = {}
        self._ik_preview_inflight = False
        self._ik_preview_lock = threading.Lock()
        self._last_ik_success: dict[str, bool] = {}
        self._last_backend_available = False
        self._last_status = ""
        self._updating_from_backend = False
        self._initial_goal_sync_pending = self._enable_backend

        self._build_menu()
        self._create_services()
        if self._enable_markers:
            for spec in GOALS.values():
                marker = make_6dof_marker(spec)
                self._last_pose_by_marker[spec.marker_name] = marker.pose
                self.server.insert(marker, feedback_callback=self._process_feedback)
                self.menu.apply(self.server, spec.marker_name)
                self._publish_pose(spec.side, marker.pose)
            self.server.applyChanges()
            self.create_timer(0.05, self._flush_dirty_goals)
        if self._enable_ik_preview:
            period = 1.0 / max(IK_PREVIEW_HZ, 0.1)
            self.create_timer(period, self._kick_ik_preview_worker)
        if self._enable_backend:
            self.create_timer(0.10, self._poll_backend_state)
        self.create_timer(1.0, self._publish_obstacles)
        self.create_timer(0.20, self._publish_goal_markers)
        self.create_timer(0.20, self._publish_ik_preview_markers)
        self._publish_obstacles()
        self._publish_goal_markers()
        self._publish_ik_preview_markers()
        self.get_logger().info(
            f"RViz frontend ready. SAFE_MODE={self._safe_mode} "
            f"backend={self._enable_backend} markers={self._enable_markers} "
            f"ik_preview={self._enable_ik_preview} planning={self._enable_planning} "
            f"execution={self._enable_execution}"
        )
        if self._enable_backend and self._enable_markers and RESET_GOALS_ON_START:
            self.get_logger().info("Resetting RViz goals from current Isaac TCP poses on startup")
            self._run_backend_async({"command": "reset_goals"})

    def _build_obstacle_markers(self) -> MarkerArray:
        marker_array = MarkerArray()
        if not OBSTACLE_CONFIG.exists():
            self.get_logger().warn(f"Obstacle config not found: {OBSTACLE_CONFIG}")
            return marker_array

        config = yaml.safe_load(OBSTACLE_CONFIG.read_text()) or {}
        cuboids = config.get("cuboid", {})
        for index, (name, cuboid) in enumerate(cuboids.items()):
            dims = [float(value) for value in cuboid["dims"]]
            # cuRobo config stores cuboid pose as [x, y, z, qw, qx, qy, qz].
            pose_values = [float(value) for value in cuboid["pose"]]

            marker = Marker()
            marker.header.frame_id = BASE_FRAME
            marker.ns = "rm75_obstacles"
            marker.id = index
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.text = str(name)
            marker.pose.position.x = pose_values[0]
            marker.pose.position.y = pose_values[1]
            marker.pose.position.z = pose_values[2]
            marker.pose.orientation.w = pose_values[3]
            marker.pose.orientation.x = pose_values[4]
            marker.pose.orientation.y = pose_values[5]
            marker.pose.orientation.z = pose_values[6]
            marker.scale.x, marker.scale.y, marker.scale.z = dims
            marker.color.r = 0.55
            marker.color.g = 0.55
            marker.color.b = 0.60
            marker.color.a = 0.45
            marker_array.markers.append(marker)
        return marker_array

    def _publish_obstacles(self) -> None:
        now = self.get_clock().now().to_msg()
        for marker in self._obstacle_markers.markers:
            marker.header.stamp = now
        self.obstacle_pub.publish(self._obstacle_markers)

    def _publish_goal_markers(self) -> None:
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        for index, spec in enumerate(GOALS.values()):
            pose = self._last_pose_by_marker.get(spec.marker_name, _make_pose(spec.default_xyz))

            sphere = Marker()
            sphere.header.stamp = now
            sphere.header.frame_id = BASE_FRAME
            sphere.ns = "rm75_goal_markers"
            sphere.id = index
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose = pose
            sphere.scale.x = 0.16
            sphere.scale.y = 0.16
            sphere.scale.z = 0.16
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = spec.color
            sphere.color.a = 0.55
            marker_array.markers.append(sphere)

            text = Marker()
            text.header.stamp = now
            text.header.frame_id = BASE_FRAME
            text.ns = "rm75_goal_labels"
            text.id = index
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = pose.position.x
            text.pose.position.y = pose.position.y
            text.pose.position.z = pose.position.z + 0.14
            text.pose.orientation.w = 1.0
            text.scale.z = 0.08
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 0.9
            text.text = spec.marker_name
            marker_array.markers.append(text)
        self.goal_marker_pub.publish(marker_array)

    def _publish_ik_preview_markers(self) -> None:
        marker_array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        if not self._enable_ik_preview:
            self.ik_preview_pub.publish(marker_array)
            return

        now = self.get_clock().now().to_msg()
        with self._ik_preview_lock:
            preview_snapshot = {side: dict(joints) for side, joints in self._ik_preview_joints.items()}
        for side_index, (side, joints) in enumerate(preview_snapshot.items()):
            if side not in GOALS:
                continue
            spec = GOALS[side]
            tip = "l_tool0" if side == "left" else "r_tool0"
            all_joints = dict(self._latest_joint_values)
            all_joints.update(joints)
            try:
                origins = self._fk_preview.link_origins(tip, all_joints)
            except Exception as exc:  # noqa: BLE001 - keep RViz node alive if URDF parsing/FK is wrong
                self.get_logger().warn(f"{side} IK preview FK failed: {exc}")
                continue

            # Keep the visual quiet and useful: draw only the moving arm section,
            # but the FK above still includes the base/platform transforms so the
            # coordinates remain in base_link_underpan.
            arm_prefix = "l_" if side == "left" else "r_"
            arm_points = [(name, xyz) for name, xyz in origins if name.startswith(arm_prefix)]
            if not arm_points:
                arm_points = origins

            base_id = side_index * 100
            line = Marker()
            line.header.stamp = now
            line.header.frame_id = BASE_FRAME
            line.ns = f"rm75_ik_preview_{side}"
            line.id = base_id
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.025
            line.color.r, line.color.g, line.color.b, _ = spec.color
            line.color.a = 0.42
            for _, xyz in arm_points:
                point = Point()
                point.x, point.y, point.z = xyz
                line.points.append(point)
            marker_array.markers.append(line)

            for idx, (link_name, xyz) in enumerate(arm_points):
                sphere = Marker()
                sphere.header.stamp = now
                sphere.header.frame_id = BASE_FRAME
                sphere.ns = f"rm75_ik_preview_{side}_joints"
                sphere.id = base_id + idx + 1
                sphere.type = Marker.SPHERE
                sphere.action = Marker.ADD
                sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = xyz
                sphere.pose.orientation.w = 1.0
                sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.055 if link_name != tip else 0.085
                sphere.color.r, sphere.color.g, sphere.color.b, _ = spec.color
                sphere.color.a = 0.35 if link_name != tip else 0.65
                marker_array.markers.append(sphere)

            if arm_points:
                label = Marker()
                label.header.stamp = now
                label.header.frame_id = BASE_FRAME
                label.ns = f"rm75_ik_preview_{side}_label"
                label.id = base_id + 90
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position.x, label.pose.position.y, label.pose.position.z = arm_points[-1][1]
                label.pose.position.z += 0.10
                label.pose.orientation.w = 1.0
                label.scale.z = 0.065
                label.color.r, label.color.g, label.color.b, label.color.a = 0.75, 1.0, 1.0, 0.85
                label.text = f"{side} IK preview"
                marker_array.markers.append(label)

        self.ik_preview_pub.publish(marker_array)

    def _build_menu(self) -> None:
        if self._enable_planning:
            self.menu.insert("Plan left", callback=lambda fb: self._run_backend_async({"command": "plan", "mode": "left"}))
            self.menu.insert("Plan right", callback=lambda fb: self._run_backend_async({"command": "plan", "mode": "right"}))
            self.menu.insert("Plan dual", callback=lambda fb: self._run_backend_async({"command": "plan", "mode": "dual"}))
        if self._enable_execution:
            self.menu.insert("Execute latest", callback=lambda fb: self._run_backend_async({"command": "execute"}))
        self.menu.insert("Reset goals from Isaac", callback=lambda fb: self._run_backend_async({"command": "reset_goals"}))

    def _create_services(self) -> None:
        self.create_service(Trigger, "/rm75/plan_left", lambda req, res: self._service_backend({"command": "plan", "mode": "left"}, res))
        self.create_service(Trigger, "/rm75/plan_right", lambda req, res: self._service_backend({"command": "plan", "mode": "right"}, res))
        self.create_service(Trigger, "/rm75/plan_dual", lambda req, res: self._service_backend({"command": "plan", "mode": "dual"}, res))
        self.create_service(Trigger, "/rm75/execute", lambda req, res: self._service_backend({"command": "execute"}, res))
        self.create_service(Trigger, "/rm75/reset_goals", lambda req, res: self._service_backend({"command": "reset_goals"}, res))

    def _stamp_pose(self, pose: Pose) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME
        msg.pose = pose
        return msg

    def _publish_pose(self, side: str, pose: Pose) -> None:
        self.pose_publishers[side].publish(self._stamp_pose(pose))

    def _process_feedback(self, feedback) -> None:
        if self._updating_from_backend:
            return
        if feedback.event_type == feedback.POSE_UPDATE:
            self._last_pose_by_marker[feedback.marker_name] = feedback.pose
            for side, spec in GOALS.items():
                if spec.marker_name == feedback.marker_name:
                    self._publish_pose(side, feedback.pose)
                    self._dirty_goal_sides.add(side)
                    break
        elif feedback.event_type == feedback.MOUSE_UP:
            self.get_logger().info(
                f"{feedback.marker_name}: xyz=({feedback.pose.position.x:.3f}, "
                f"{feedback.pose.position.y:.3f}, {feedback.pose.position.z:.3f})"
            )

    def _flush_dirty_goals(self) -> None:
        if not self._dirty_goal_sides:
            return
        dirty = list(self._dirty_goal_sides)
        self._dirty_goal_sides.clear()
        for side in dirty:
            spec = GOALS[side]
            pose = self._last_pose_by_marker[spec.marker_name]
            self._backend_request({"command": "set_goal", "side": side, "pose": _pose_to_backend(pose)}, timeout=0.10, quiet=True)
            self._request_ik_preview(side, pose)

    def _request_ik_preview(self, side: str, pose: Pose) -> None:
        if not (self._enable_backend and self._enable_markers and self._enable_ik_preview):
            return
        with self._ik_preview_lock:
            # Latest-value semantics: if the user drags faster than the backend
            # solves IK, keep only the newest target pose per arm.
            self._ik_preview_pending[side] = _copy_pose(pose)

    def _kick_ik_preview_worker(self) -> None:
        if not (self._enable_backend and self._enable_ik_preview):
            return
        with self._ik_preview_lock:
            if self._ik_preview_inflight or not self._ik_preview_pending:
                return
            batch = dict(self._ik_preview_pending)
            self._ik_preview_pending.clear()
            self._ik_preview_inflight = True
        threading.Thread(target=self._ik_preview_worker, args=(batch,), daemon=True).start()

    def _ik_preview_worker(self, batch: dict[str, Pose]) -> None:
        try:
            for side, pose in batch.items():
                response = self._backend_request(
                    {"command": "preview_ik", "side": side, "pose": _pose_to_backend(pose)},
                    timeout=IK_PREVIEW_TIMEOUT,
                    quiet=True,
                )
                self._handle_ik_preview_response(side, response)
        finally:
            with self._ik_preview_lock:
                self._ik_preview_inflight = False

    def _handle_ik_preview_response(self, side: str, response: dict | None) -> None:
        ok = bool(response and response.get("success") and response.get("joints"))
        with self._ik_preview_lock:
            previous = self._last_ik_success.get(side)
            self._last_ik_success[side] = ok
            if ok:
                self._ik_preview_joints[side] = {
                    str(name): float(value) for name, value in response.get("joints", {}).items()
                }
            else:
                self._ik_preview_joints.pop(side, None)
        if not ok and response is not None and previous is not False:
            message = str(response.get("message", f"{side} IK preview failed"))
            self._publish_status(message)

    def _backend_request(self, request: dict, timeout: float = 1.0, quiet: bool = False) -> dict | None:
        try:
            with socket.create_connection((BACKEND_HOST, BACKEND_PORT), timeout=timeout) as sock:
                sock.settimeout(timeout)
                sock.sendall((json.dumps(request) + "\n").encode("utf-8"))
                data = b""
                while not data.endswith(b"\n"):
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    data += chunk
            if not data:
                raise RuntimeError("empty backend response")
            if not self._last_backend_available:
                self.get_logger().info("Isaac/cuRobo backend connected")
            self._last_backend_available = True
            return json.loads(data.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - connection errors are user-facing status
            if self._last_backend_available or not quiet:
                self.get_logger().warn(f"Isaac/cuRobo backend unavailable: {exc}")
            self._last_backend_available = False
            return None

    def _run_backend_async(self, request: dict) -> None:
        command = request.get("command")
        if not self._enable_backend:
            self.get_logger().warn(f"backend disabled by SAFE_MODE={self._safe_mode}")
            return
        if command == "plan" and not self._enable_planning:
            self.get_logger().warn(f"planning disabled by SAFE_MODE={self._safe_mode}")
            return
        if command == "execute" and not self._enable_execution:
            self.get_logger().warn(f"execution disabled by SAFE_MODE={self._safe_mode}")
            return
        threading.Thread(target=lambda: self._handle_backend_response(self._backend_request(request, timeout=300.0)), daemon=True).start()

    def _service_backend(self, request: dict, response):
        command = request.get("command")
        if not self._enable_backend:
            response.success = False
            response.message = f"backend disabled by SAFE_MODE={self._safe_mode}"
            return response
        if command == "plan" and not self._enable_planning:
            response.success = False
            response.message = f"planning disabled by SAFE_MODE={self._safe_mode}"
            return response
        if command == "execute" and not self._enable_execution:
            response.success = False
            response.message = f"execution disabled by SAFE_MODE={self._safe_mode}"
            return response
        backend_response = self._backend_request(request, timeout=300.0)
        self._handle_backend_response(backend_response)
        response.success = bool(backend_response and backend_response.get("success"))
        response.message = "backend unavailable" if backend_response is None else str(backend_response.get("message", ""))
        return response

    def _handle_backend_response(self, response: dict | None) -> None:
        if response is None:
            return
        message = str(response.get("message", ""))
        if message:
            (self.get_logger().info if response.get("success") else self.get_logger().warn)(message)
            self._publish_status(message)
        if "goals" in response:
            self._apply_backend_goals(response["goals"])
        if "trajectory" in response:
            self._publish_trajectory(response["trajectory"])

    def _poll_backend_state(self) -> None:
        if not self._enable_backend:
            return
        response = self._backend_request({"command": "get_state"}, timeout=0.20, quiet=True)
        if not response or not response.get("success"):
            return
        if "joints" in response:
            self._publish_joint_state(response["joints"])
        if "goals" in response:
            update_markers = self._initial_goal_sync_pending and self._enable_markers
            self._apply_backend_goals(response["goals"], update_markers=update_markers)
            if update_markers:
                self._initial_goal_sync_pending = False
        message = str(response.get("message", ""))
        if message and message != self._last_status:
            self._last_status = message
            self._publish_status(message)

    def _apply_backend_goals(self, goals: dict, update_markers: bool = True) -> None:
        self._updating_from_backend = True
        try:
            for side, pose_data in goals.items():
                if side not in GOALS:
                    continue
                spec = GOALS[side]
                pose = _backend_to_pose(pose_data)
                self.goal_state_publishers[side].publish(self._stamp_pose(pose))
                if update_markers and self._enable_markers:
                    self._last_pose_by_marker[spec.marker_name] = pose
                    self.server.setPose(spec.marker_name, pose)
                if update_markers:
                    self._request_ik_preview(side, pose)
            if update_markers and self._enable_markers:
                self.server.applyChanges()
        finally:
            self._updating_from_backend = False

    def _publish_joint_state(self, joints: dict) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [str(name) for name in joints.get("names", [])]
        msg.position = [float(value) for value in joints.get("positions", [])]
        self._latest_joint_values.update(dict(zip(msg.name, msg.position)))
        self.joint_state_pub.publish(msg)

    def _publish_trajectory(self, trajectory: dict) -> None:
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [str(name) for name in trajectory.get("joint_names", [])]
        dt = float(trajectory.get("dt", 0.025))
        for index, positions in enumerate(trajectory.get("points", [])):
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in positions]
            t = index * dt
            point.time_from_start = Duration(sec=int(t), nanosec=int((t % 1.0) * 1e9))
            msg.points.append(point)
        self.trajectory_pub.publish(msg)

        # Show the planned TCP paths directly in RViz without a MoveIt
        # MotionPlanning display or a second robot-state publisher.
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        points = trajectory.get("points", [])
        names = list(trajectory.get("joint_names", []))
        for index, side in enumerate(("left", "right")):
            tip = "l_tool0" if side == "left" else "r_tool0"
            prefix = "l_" if side == "left" else "r_"
            line = Marker()
            line.header.stamp = self.get_clock().now().to_msg()
            line.header.frame_id = BASE_FRAME
            line.ns = "rm75_curobo_path"
            line.id = index
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.012
            line.color.r, line.color.g, line.color.b, line.color.a = GOALS[side].color
            for values in points:
                joints = dict(zip(names, values))
                try:
                    origins = self._fk_preview.link_origins(tip, joints)
                    xyz = next(point for name, point in origins if name == tip)
                except Exception:
                    continue
                point = Point()
                point.x, point.y, point.z = xyz
                line.points.append(point)
            if len(line.points) > 1:
                markers.markers.append(line)
        self.trajectory_marker_pub.publish(markers)

    def _publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = RM75InteractiveMarkers()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
