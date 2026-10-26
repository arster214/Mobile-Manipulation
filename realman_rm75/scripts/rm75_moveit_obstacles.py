#!/usr/bin/env python3
"""Publish the canonical RM75 obstacle scene into MoveIt2 PlanningScene."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, ObjectColor, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from shape_msgs.msg import SolidPrimitive

from rm75_obstacles import DEFAULT_CONFIG, ObstacleScene, load_obstacle_scene


CONFIG_PATH = Path(os.environ.get("RM75_OBSTACLE_CONFIG", str(DEFAULT_CONFIG)))
OBJECT_PREFIX = "rm75_obstacle/"
BACKEND_HOST = os.environ.get("RM75_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("RM75_BACKEND_PORT", "50075"))


class RM75MoveItObstacles(Node):
    def __init__(self) -> None:
        super().__init__("rm75_moveit_obstacles")
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._scene_pub = self.create_publisher(PlanningScene, "obstacle_scene", qos)
        self._apply_client = self.create_client(ApplyPlanningScene, "apply_planning_scene")
        self._active_ids: set[str] = set()
        self._backend_verified = False
        self._last_verify_error = ""
        self._pending: tuple[ObstacleScene, PlanningScene] | None = None
        self._apply_future = None
        self._source = load_obstacle_scene(CONFIG_PATH)
        self._pending = (self._source, self._build_message(self._source))
        self._scene_pub.publish(self._pending[1])
        self._timer = self.create_timer(0.25, self._try_apply)

    @staticmethod
    def _object_id(name: str) -> str:
        return OBJECT_PREFIX + name

    def _build_message(self, source: ObstacleScene) -> PlanningScene:
        message = PlanningScene()
        message.name = f"rm75_obstacles_{source.fingerprint}"
        message.is_diff = True
        message.robot_state.is_diff = True

        new_ids = {self._object_id(item.name) for item in source.cuboids}
        for stale_id in sorted(self._active_ids - new_ids):
            stale = CollisionObject()
            stale.header.frame_id = source.frame_id
            stale.id = stale_id
            stale.operation = CollisionObject.REMOVE
            message.world.collision_objects.append(stale)

        for item in source.cuboids:
            collision = CollisionObject()
            collision.header.frame_id = source.frame_id
            collision.id = self._object_id(item.name)
            collision.operation = CollisionObject.ADD

            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = list(item.dims)
            collision.primitives.append(primitive)
            local_pose = Pose()
            local_pose.orientation.w = 1.0
            collision.primitive_poses.append(local_pose)

            pose = collision.pose
            pose.position.x, pose.position.y, pose.position.z = item.position
            qw, qx, qy, qz = item.quaternion_wxyz
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
            message.world.collision_objects.append(collision)

            color = ObjectColor()
            color.id = collision.id
            color.color.r, color.color.g, color.color.b, color.color.a = item.color_rgba
            message.object_colors.append(color)

        return message

    def _verify_backend_scene(self) -> bool:
        try:
            with socket.create_connection((BACKEND_HOST, BACKEND_PORT), timeout=0.3) as sock:
                sock.settimeout(0.5)
                sock.sendall(b'{"command":"get_state"}\n')
                response = json.loads(sock.makefile("r", encoding="utf-8").readline())
            remote = response.get("obstacle_scene") or {}
            fingerprint = remote.get("fingerprint")
            if fingerprint != self._source.fingerprint:
                raise RuntimeError(
                    f"backend obstacle fingerprint={fingerprint!r}, "
                    f"local={self._source.fingerprint!r}; restart the complete RM75 stack"
                )
            if remote.get("frame_id") != self._source.frame_id:
                raise RuntimeError(
                    f"backend obstacle frame={remote.get('frame_id')!r}, "
                    f"local={self._source.frame_id!r}"
                )
            self._backend_verified = True
            self.get_logger().info(
                f"verified Isaac/cuRobo obstacle fingerprint={self._source.fingerprint}"
            )
            return True
        except Exception as exc:
            error = str(exc)
            if error != self._last_verify_error:
                self.get_logger().error(f"obstacle scene handshake failed: {error}")
                self._last_verify_error = error
            return False

    def _try_apply(self) -> None:
        if self._pending is None or self._apply_future is not None:
            return
        if not self._backend_verified and not self._verify_backend_scene():
            return
        if not self._apply_client.service_is_ready():
            return
        source, message = self._pending
        request = ApplyPlanningScene.Request()
        request.scene = message
        self._apply_future = self._apply_client.call_async(request)
        self._apply_future.add_done_callback(
            lambda future, queued_source=source: self._apply_done(future, queued_source)
        )

    def _apply_done(self, future, source: ObstacleScene) -> None:
        self._apply_future = None
        try:
            response = future.result()
            if response is None or not response.success:
                self.get_logger().error("MoveIt rejected the canonical obstacle scene; retrying")
                return
        except Exception as exc:
            self.get_logger().error(f"apply_planning_scene failed: {exc!r}; retrying")
            return

        self._active_ids = {self._object_id(item.name) for item in source.cuboids}
        self._pending = None
        self._scene_pub.publish(self._build_message(source))
        self.get_logger().info(
            f"applied {len(source.cuboids)} MoveIt obstacles in {source.frame_id}; "
            f"fingerprint={source.fingerprint}"
        )


def main() -> None:
    rclpy.init()
    node = RM75MoveItObstacles()
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
