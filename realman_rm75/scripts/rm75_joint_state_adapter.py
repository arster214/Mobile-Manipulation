#!/usr/bin/env python3
"""Small ROS2-only adapter for the Isaac/cuRobo state socket.

This is intentionally separate from the old Interactive Marker frontend.  The
MoveIt2 workflow only needs the real joint state stream; it does not need the
legacy marker, IK-preview, obstacle-marker, or service machinery.
"""

from __future__ import annotations

import json
import os
import socket

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState


class RM75JointStateAdapter(Node):
    def __init__(self) -> None:
        super().__init__("rm75_joint_state_adapter")
        self.host = os.environ.get("RM75_BACKEND_HOST", "127.0.0.1")
        self.port = int(os.environ.get("RM75_BACKEND_PORT", "50075"))
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.timer = self.create_timer(0.05, self._poll)
        self._warned = False
        self.get_logger().info(f"joint-state adapter ready; backend={self.host}:{self.port}")

    def _poll(self) -> None:
        try:
            with socket.create_connection((self.host, self.port), timeout=0.15) as sock:
                sock.settimeout(0.25)
                sock.sendall(b'{"command":"get_state"}\n')
                raw = b""
                while not raw.endswith(b"\n"):
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    raw += chunk
            response = json.loads(raw.decode("utf-8"))
            joints = response.get("joints") or {}
            names = joints.get("names") or []
            positions = joints.get("positions") or []
            if not response.get("success") or len(names) != len(positions):
                raise RuntimeError(response.get("message", "invalid backend joint snapshot"))
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = [str(name) for name in names]
            msg.position = [float(value) for value in positions]
            self.pub.publish(msg)
            self._warned = False
        except Exception as exc:  # backend can be starting or shutting down
            if not self._warned:
                self.get_logger().warning(f"backend joint state unavailable: {exc}")
                self._warned = True


def main() -> None:
    rclpy.init()
    node = RM75JointStateAdapter()
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
