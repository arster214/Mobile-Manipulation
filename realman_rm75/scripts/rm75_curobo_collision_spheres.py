#!/usr/bin/env python3
"""Publish the live cuRobo collision-sphere model as RViz markers."""

from __future__ import annotations

import json
import os
import socket

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


BASE_FRAME = "base_link_underpan"


class RM75CuroboCollisionSpheres(Node):
    def __init__(self) -> None:
        super().__init__("rm75_curobo_collision_spheres")
        self.host = os.environ.get("RM75_BACKEND_HOST", "127.0.0.1")
        self.port = int(os.environ.get("RM75_BACKEND_PORT", "50075"))
        self.pub = self.create_publisher(
            MarkerArray, "/rm75/curobo_collision_spheres", 1
        )
        self.timer = self.create_timer(0.5, self._poll)
        self._warned = False
        self._reported_count = False
        self.get_logger().info(
            f"cuRobo collision-sphere display ready; backend={self.host}:{self.port}"
        )

    def _request(self) -> dict:
        with socket.create_connection((self.host, self.port), timeout=1.0) as sock:
            # The first request may initialize the dual-arm cuRobo planner and
            # capture CUDA graphs, so allow that one-time cost.
            sock.settimeout(30.0)
            sock.sendall(b'{"command":"get_collision_spheres"}\n')
            raw = b""
            while not raw.endswith(b"\n"):
                chunk = sock.recv(65536)
                if not chunk:
                    break
                raw += chunk
        if not raw:
            raise RuntimeError("backend closed collision-sphere request")
        return json.loads(raw.decode("utf-8"))

    @staticmethod
    def _color(link: str) -> tuple[float, float, float, float]:
        if link.startswith("l_"):
            return (0.10, 0.55, 1.00, 0.42)
        if link.startswith("r_"):
            return (1.00, 0.45, 0.08, 0.42)
        if link.startswith("head_") or link == "camera_link":
            return (0.72, 0.25, 0.95, 0.38)
        return (0.65, 0.68, 0.72, 0.32)

    def _poll(self) -> None:
        try:
            response = self._request()
            if not response.get("success"):
                raise RuntimeError(
                    response.get("message", "collision-sphere request failed")
                )
            frame_id = response.get("frame_id", BASE_FRAME)
            markers = MarkerArray()
            clear = Marker()
            clear.action = Marker.DELETEALL
            markers.markers.append(clear)
            for index, sphere in enumerate(response.get("spheres", [])):
                position = sphere["position"]
                radius = float(sphere["radius"])
                marker = Marker()
                marker.header.frame_id = frame_id
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "rm75_curobo_collision_spheres"
                marker.id = index
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                marker.pose.position.x = float(position[0])
                marker.pose.position.y = float(position[1])
                marker.pose.position.z = float(position[2])
                marker.scale.x = 2.0 * radius
                marker.scale.y = 2.0 * radius
                marker.scale.z = 2.0 * radius
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = (
                    self._color(str(sphere.get("link", "")))
                )
                markers.markers.append(marker)
            self.pub.publish(markers)
            if not self._reported_count:
                self.get_logger().info(
                    f"publishing {len(markers.markers) - 1} live cuRobo collision spheres "
                    f"in {frame_id}"
                )
                self._reported_count = True
            self._warned = False
        except Exception as exc:  # keep RViz display alive while backend starts/stops
            if not self._warned:
                self.get_logger().warning(
                    f"cuRobo collision spheres unavailable: {exc}"
                )
                self._warned = True


def main() -> None:
    rclpy.init()
    node = RM75CuroboCollisionSpheres()
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
