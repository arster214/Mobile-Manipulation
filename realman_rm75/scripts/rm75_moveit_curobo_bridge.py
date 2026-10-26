#!/usr/bin/env python3
"""MoveIt2 ``move_action`` server backed by the RM75 cuRobo socket.

The bridge deliberately treats MoveIt as a pose frontend.  It extracts the
end-effector pose constraints, sends those poses to Isaac/cuRobo, and returns
cuRobo's resulting joint trajectory for RViz visualization.
"""

from __future__ import annotations

import json
import os
import socket
import time

import rclpy
from builtin_interfaces.msg import Duration
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    DisplayTrajectory,
    MoveItErrorCodes,
    RobotState,
    RobotTrajectory,
)
from moveit_msgs.srv import GetPositionFK
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import InteractiveMarkerFeedback


HOST = os.environ.get("RM75_BACKEND_HOST", "127.0.0.1")
PORT = int(os.environ.get("RM75_BACKEND_PORT", "50075"))
MARKER_FEEDBACK_TOPICS = tuple(filter(None, os.environ.get(
    "RM75_MOVEIT_MARKER_FEEDBACK_TOPICS",
    "rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/feedback,"
    "/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/feedback,"
    "/robot_interaction_interactive_marker_topic/feedback",
).split(",")))
TIP_BY_LINK = {"l_tool0": "left", "l_link7": "left", "r_tool0": "right", "r_link7": "right"}
GROUP_MODE = {"l_arm": "left", "r_arm": "right", "dual_arm": "dual"}
ALL_JOINTS = [f"l_joint{i}" for i in range(1, 8)] + [f"r_joint{i}" for i in range(1, 8)]


def _duration(seconds: float) -> Duration:
    msg = Duration()
    msg.sec = int(seconds)
    msg.nanosec = int((seconds - msg.sec) * 1e9)
    return msg


def _error(code: int) -> MoveItErrorCodes:
    msg = MoveItErrorCodes()
    msg.val = code
    return msg


class RM75MoveItCuroboBridge(Node):
    def __init__(self) -> None:
        super().__init__("rm75_moveit_curobo_bridge")
        self._callback_group = ReentrantCallbackGroup()
        self._last_state: dict = {}
        self._marker_pose_by_side: dict = {}
        self._initial_goal_synced = False
        self._joint_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_cb, 10,
            callback_group=self._callback_group)
        self._display_pub = self.create_publisher(DisplayTrajectory, "/rm75/curobo_trajectory", 10)
        self._legacy_display_pub = self.create_publisher(DisplayTrajectory, "/display_planned_path", 10)
        self._goal_state_pub = self.create_publisher(
            RobotState, "/rviz/moveit/update_custom_goal_state", 1)
        self._fk_client = self.create_client(
            GetPositionFK, "compute_fk", callback_group=self._callback_group)
        self._marker_subscriptions = [
            self.create_subscription(
                InteractiveMarkerFeedback, topic, self._marker_feedback_cb, 10,
                callback_group=self._callback_group)
            for topic in MARKER_FEEDBACK_TOPICS
        ]
        self._action = ActionServer(
            self, MoveGroup, "move_action", execute_callback=self._execute,
            goal_callback=self._goal_cb, cancel_callback=self._cancel_cb,
            callback_group=self._callback_group)
        self._goal_sync_timer = self.create_timer(
            0.25, self._sync_initial_goal_state, callback_group=self._callback_group)
        self.get_logger().info(f"pose-only MoveIt action bridge ready; backend={HOST}:{PORT}")

    def _joint_cb(self, msg: JointState) -> None:
        self._last_state = {"names": list(msg.name), "positions": list(msg.position)}

    def _sync_initial_goal_state(self) -> None:
        if self._initial_goal_synced or not self._last_state:
            return
        if self._goal_state_pub.get_subscription_count() == 0:
            return
        self._goal_state_pub.publish(self._start_state())
        self._initial_goal_synced = True
        self.get_logger().info("initialized native MoveIt Goal State from current /joint_states")

    def _marker_feedback_cb(self, msg: InteractiveMarkerFeedback) -> None:
        name = msg.marker_name.lower()
        side = None
        if "left" in name or "l_link7" in name or "l_tool0" in name:
            side = "left"
        elif "right" in name or "r_link7" in name or "r_tool0" in name:
            side = "right"
        if side is None:
            return
        pose = msg.pose
        self._marker_pose_by_side[side] = (
            [pose.position.x, pose.position.y, pose.position.z],
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        )

    def _goal_cb(self, goal_request: MoveGroup.Goal) -> GoalResponse:
        group = goal_request.request.group_name
        if group not in GROUP_MODE:
            self.get_logger().warn(f"rejecting unsupported MoveIt group {group!r}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_cb(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _extract_poses(self, request, mode: str) -> dict[str, tuple[list[float], list[float]]]:
        constraints = request.goal_constraints[0] if request.goal_constraints else None
        if constraints is None:
            raise ValueError("MoveIt request has no goal_constraints")
        positions = {}
        orientations = {}
        for item in constraints.position_constraints:
            side = TIP_BY_LINK.get(item.link_name)
            if side is None or not item.constraint_region.primitives:
                continue
            primitive_pose = item.constraint_region.primitive_poses[0]
            # RViz normally sends a zero-radius sphere around the target.
            # The primitive pose is the target position in header.frame_id.
            positions[side] = ([primitive_pose.position.x, primitive_pose.position.y, primitive_pose.position.z], item.link_name)
        for item in constraints.orientation_constraints:
            side = TIP_BY_LINK.get(item.link_name)
            if side is not None:
                q = item.orientation
                orientations[side] = [q.x, q.y, q.z, q.w]
        if mode == "left":
            wanted = ("left",)
        elif mode == "right":
            wanted = ("right",)
        else:
            wanted = ("left", "right")

        if all(side in self._marker_pose_by_side for side in wanted):
            marker_result = {}
            for side in wanted:
                position, quat = self._marker_pose_by_side[side]
                frame = "l_tool0" if side == "left" else "r_tool0"
                marker_result[frame] = (self._link7_pose_to_tool0(position, quat), quat)
            self.get_logger().info("using native MoveIt Interactive Marker workspace pose as cuRobo goal")
            return marker_result

        result = {}
        for side in wanted:
            if side not in positions:
                return self._poses_from_goal_state_fk(constraints, wanted)
            quat = orientations.get(side, [0.0, 0.0, 0.0, 1.0])
            frame = "l_tool0" if side == "left" else "r_tool0"
            position, link_name = positions[side]
            if link_name.endswith("link7"):
                # MoveIt's original RM75 end-effector is link7. cuRobo's
                # planning frame is tool0, fixed 44.5 mm along link7 +Z.
                result[frame] = (self._link7_pose_to_tool0(position, quat), quat)
            else:
                result[frame] = (position, quat)
        return result

    @staticmethod
    def _link7_pose_to_tool0(position, quat) -> list[float]:
        x, y, z, w = quat
        tx, ty, tz = 0.0, 0.0, 0.0445
        return [
            position[0] + (1 - 2 * (y * y + z * z)) * tx + 2 * (x * y - z * w) * ty + 2 * (x * z + y * w) * tz,
            position[1] + 2 * (x * y + z * w) * tx + (1 - 2 * (x * x + z * z)) * ty + 2 * (y * z - x * w) * tz,
            position[2] + 2 * (x * z - y * w) * tx + 2 * (y * z + x * w) * ty + (1 - 2 * (x * x + y * y)) * tz,
        ]

    def _poses_from_goal_state_fk(self, constraints, wanted) -> dict[str, tuple[list[float], list[float]]]:
        """Recover X_goal from MoveIt's Goal State without using q_goal as a cuRobo goal."""
        if not constraints.joint_constraints:
            raise ValueError("MoveIt goal has neither pose nor joint constraints")
        current = dict(zip(
            self._last_state.get("names", []),
            self._last_state.get("positions", []),
        ))
        if not current:
            raise RuntimeError("no current /joint_states available for Goal State FK")
        for joint in constraints.joint_constraints:
            current[joint.joint_name] = float(joint.position)

        fk_request = GetPositionFK.Request()
        fk_request.header.frame_id = "base_link_underpan"
        fk_request.fk_link_names = [
            "l_tool0" if side == "left" else "r_tool0" for side in wanted
        ]
        fk_request.robot_state.joint_state.name = list(current)
        fk_request.robot_state.joint_state.position = [float(current[name]) for name in current]
        if not self._fk_client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError("MoveIt compute_fk service is unavailable")
        fk_response = self._fk_client.call(fk_request)
        if fk_response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f"MoveIt Goal State FK failed: code={fk_response.error_code.val}")

        result = {}
        for link_name, stamped in zip(fk_response.fk_link_names, fk_response.pose_stamped):
            pose = stamped.pose
            result[link_name] = (
                [pose.position.x, pose.position.y, pose.position.z],
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            )
        missing = [side for side in wanted if ("l_tool0" if side == "left" else "r_tool0") not in result]
        if missing:
            raise RuntimeError(f"MoveIt FK response missing tool poses for: {missing}")
        self.get_logger().info("converted MoveIt Goal State to workspace tool pose via MoveIt FK")
        return result

    @staticmethod
    def _request(command: dict, timeout: float = 360.0) -> dict:
        with socket.create_connection((HOST, PORT), timeout=2.0) as sock:
            sock.settimeout(timeout)
            sock.sendall((json.dumps(command) + "\n").encode("utf-8"))
            raw = b""
            while not raw.endswith(b"\n"):
                chunk = sock.recv(65536)
                if not chunk:
                    break
                raw += chunk
        if not raw:
            raise RuntimeError("cuRobo backend closed the socket")
        return json.loads(raw.decode("utf-8"))

    def _start_state(self) -> RobotState:
        state = RobotState()
        state.joint_state.name = list(self._last_state.get("names", []))
        state.joint_state.position = list(self._last_state.get("positions", []))
        return state

    def _trajectory(self, data: dict) -> RobotTrajectory:
        result = RobotTrajectory()
        traj = data.get("trajectory") or {}
        result.joint_trajectory.joint_names = list(traj.get("joint_names", ALL_JOINTS))
        dt = float(traj.get("dt", 0.02))
        for index, values in enumerate(traj.get("points", [])):
            point = JointTrajectoryPoint()
            point.positions = [float(value) for value in values]
            point.time_from_start = _duration(index * dt)
            result.joint_trajectory.points.append(point)
        return result

    async def _execute(self, goal_handle):
        started = time.monotonic()
        result = MoveGroup.Result()
        request = goal_handle.request.request
        mode = GROUP_MODE.get(request.group_name)
        self.get_logger().info(f"received MoveIt goal: group={request.group_name!r} mode={mode!r}")
        result.trajectory_start = self._start_state()
        if mode is None:
            result.error_code = _error(MoveItErrorCodes.INVALID_GROUP_NAME)
            goal_handle.abort()
            return result
        try:
            poses = self._extract_poses(request, mode)
            for frame, (position, quaternion) in poses.items():
                self.get_logger().info(
                    f"cuRobo pose goal {frame}: "
                    f"position={[round(float(v), 6) for v in position]} "
                    f"quaternion_xyzw={[round(float(v), 6) for v in quaternion]}"
                )
                side = "left" if frame.startswith("l_") else "right"
                response = self._request({"command": "set_goal", "side": side, "pose": {
                    "position": position, "quaternion_xyzw": quaternion}} , timeout=5.0)
                if not response.get("success"):
                    raise RuntimeError(response.get("message", "set_goal failed"))
            goal_handle.publish_feedback(MoveGroup.Feedback(state="cuRobo planning"))
            response = self._request({"command": "plan", "mode": mode})
            if not response.get("success"):
                raise RuntimeError(response.get("message", "cuRobo planning failed"))
            result.planned_trajectory = self._trajectory(response)
            result.planning_time = time.monotonic() - started
            display = DisplayTrajectory()
            display.trajectory_start = result.trajectory_start
            display.trajectory = [result.planned_trajectory]
            self._display_pub.publish(display)
            self._legacy_display_pub.publish(display)
            if goal_handle.is_cancel_requested:
                result.error_code = _error(MoveItErrorCodes.PREEMPTED)
                goal_handle.canceled()
            else:
                result.error_code = _error(MoveItErrorCodes.SUCCESS)
                goal_handle.succeed()
            return result
        except Exception as exc:  # report a normal MoveIt failure to RViz
            self.get_logger().error(f"MoveIt -> cuRobo planning failed: {exc}")
            result.error_code = _error(MoveItErrorCodes.PLANNING_FAILED)
            result.planning_time = time.monotonic() - started
            goal_handle.abort()
            return result


def main() -> None:
    rclpy.init()
    node = RM75MoveItCuroboBridge()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except ExternalShutdownException:
        pass
    finally:
        executor.shutdown()
        node._action.destroy()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
