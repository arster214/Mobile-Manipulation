"""Stable local socket transport between ROS2/RViz and Isaac/cuRobo.

One JSON object per line over TCP. Network threads never touch USD, Physics,
Articulation, or cuRobo directly. They only parse messages, update thread-safe
latest-value goal buffers, or enqueue bounded main-thread commands.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import socketserver
import threading
import time
from typing import Any, Literal

import yaml

from rm75_planner import ALL_JOINTS, MODE_TOOL_FRAMES, RM75MotionPlanner
from rm75_obstacles import load_obstacle_scene


ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = ROOT / "config" / "rm75_dual.yml"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 50075
TOOL_FRAME_BY_SIDE = {"left": "l_tool0", "right": "r_tool0"}
CHAIN_LINKS = {
    "left": ("l_base_link1", "l_tool0"),
    "right": ("r_base_link1", "r_tool0"),
}
COMMAND_QUEUE_SIZE = 8
VALID_STATES = ("IDLE", "GOAL_READY", "PLANNING", "PLAN_READY", "EXECUTING", "RESETTING")


@dataclass
class StoredGoal:
    position: list[float]
    quaternion_xyzw: list[float]

    @property
    def curobo(self) -> tuple[list[float], list[float]]:
        x, y, z, w = self.quaternion_xyzw
        return self.position, [w, x, y, z]

    def as_json(self) -> dict[str, list[float]]:
        return {"position": self.position, "quaternion_xyzw": self.quaternion_xyzw}


class _JsonLineHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server: "_ThreadedServer" = self.server  # type: ignore[assignment]
        try:
            for raw in self.rfile:
                try:
                    request = json.loads(raw.decode("utf-8"))
                    response = server.backend.submit(request)
                except Exception as exc:  # noqa: BLE001 - serialize backend failure
                    response = {"success": False, "message": repr(exc)}
                self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))
                self.wfile.flush()
        except (ConnectionResetError, BrokenPipeError):
            return


class _ThreadedServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, backend: "RM75SocketGoalServer") -> None:
        self.backend = backend
        super().__init__(address, _JsonLineHandler)


class RM75SocketGoalServer:
    """Main-thread command dispatcher for Isaac and cuRobo."""

    def __init__(
        self,
        stage,
        robot_prim,
        planner: RM75MotionPlanner,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        enable_planning: bool = True,
        enable_execution: bool = True,
        enable_ik_preview: bool = False,
    ):
        from isaacsim.core.experimental.prims import Articulation

        self._stage = stage
        self._robot_prim = robot_prim
        self._planner = planner
        self._enable_planning = enable_planning
        self._enable_execution = enable_execution
        self._enable_ik_preview = enable_ik_preview
        self._obstacle_scene = load_obstacle_scene()
        robot_config = yaml.safe_load(ROBOT_CONFIG.read_text())
        self._locked_joint_positions = {
            str(name): float(value)
            for name, value in robot_config["kinematics"].get("lock_joints", {}).items()
        }
        self._articulation = Articulation(str(robot_prim.GetPath()))
        self._dof_names = list(self._articulation.dof_names)
        self._locked_dof_indices = [
            self._dof_names.index(name)
            for name in self._locked_joint_positions
            if name in self._dof_names
        ]
        self._locked_dof_positions = [
            self._locked_joint_positions[self._dof_names[index]]
            for index in self._locked_dof_indices
        ]
        self._reported_locked_hold_error = False
        self._last_locked_hold_time = 0.0
        self._locked_hold_interval = float(os.environ.get("RM75_LOCKED_HOLD_INTERVAL", "0.25"))
        self._enable_locked_target_hold = os.environ.get("RM75_LOCKED_TARGET_HOLD", "1") != "0"

        self._state = "IDLE"
        self._status = "socket backend starting"
        self._plan_request_id = 0
        self._plan_count = 0
        self._last_diag_time = time.monotonic()
        self._socket_rx_count = 0
        self._goal_update_count = 0
        self._joint_snapshot_count = 0

        self._goals: dict[str, StoredGoal] = {}
        self._ik_posers = {}
        self._goal_snapshot: dict[str, dict[str, list[float]]] = {}
        self._pending_goals: dict[str, StoredGoal] = {}
        self._last_joints: dict[str, Any] = {"names": self._dof_names.copy(), "positions": []}
        self._lock = threading.Lock()

        self._requests: queue.Queue[tuple[dict[str, Any], queue.Queue[dict[str, Any]]]] = queue.Queue(
            maxsize=COMMAND_QUEUE_SIZE
        )
        self._server = _ThreadedServer((host, port), self)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="rm75_socket_goal_server",
            daemon=True,
        )
        self._planner.set_status_callback(self._set_status)
        self.reset_goals_from_current_tcp()
        self._update_snapshots()
        self._thread.start()
        self._set_status(f"socket backend ready at {host}:{port}")

    def _set_state(self, state: str) -> None:
        if state not in VALID_STATES:
            raise ValueError(f"Invalid backend state: {state}")
        if state != self._state:
            print(f"RM75 socket backend: [STATE] {self._state} -> {state}")
            self._state = state

    def _set_status(self, text: str) -> None:
        with self._lock:
            self._status = text
        print(f"RM75 socket backend: {text}")

    def submit(self, request: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
        """Handle a network-thread request without touching Isaac objects."""
        with self._lock:
            self._socket_rx_count += 1

        command = request.get("command")
        if command == "set_goal":
            return self._submit_set_goal(request)
        if command == "get_state":
            return self._snapshot_state()

        response_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        try:
            self._requests.put_nowait((request, response_queue))
        except queue.Full:
            return {
                "success": False,
                "message": f"backend command queue full ({COMMAND_QUEUE_SIZE}); dropping {command!r}",
                "state": self._state,
            }
        try:
            return response_queue.get(timeout=timeout)
        except queue.Empty:
            return {"success": False, "message": "backend request timed out", "state": self._state}

    def _submit_set_goal(self, request: dict[str, Any]) -> dict[str, Any]:
        side = request.get("side")
        if side not in ("left", "right"):
            return {"success": False, "message": f"invalid side: {side!r}"}
        pose = request.get("pose") or {}
        try:
            position = [float(v) for v in pose.get("position", [])]
            quaternion = [float(v) for v in pose.get("quaternion_xyzw", [])]
        except (TypeError, ValueError):
            return {"success": False, "message": "pose contains non-numeric values"}
        if len(position) != 3 or len(quaternion) != 4:
            return {"success": False, "message": "pose must contain position[3] and quaternion_xyzw[4]"}

        goal = StoredGoal(position, quaternion)
        with self._lock:
            # latest-value semantics: newest goal overwrites older unconsumed goals.
            self._pending_goals[side] = goal
            self._goal_snapshot[side] = goal.as_json()
            self._goal_update_count += 1
        return {"success": True, "message": f"{side} goal accepted as latest value", "state": self._state}

    def update(self) -> None:
        """Run queued Isaac/cuRobo commands from the Isaac main/update thread."""
        self._hold_locked_joints()
        self._consume_pending_goals()
        self._update_execution_state()

        processed = 0
        while processed < COMMAND_QUEUE_SIZE:
            try:
                request, response_queue = self._requests.get_nowait()
            except queue.Empty:
                break
            try:
                response = self._handle_main_thread_request(request)
            except Exception as exc:  # noqa: BLE001 - serialize backend failure
                response = {"success": False, "message": repr(exc), "state": self._state}
            response_queue.put(response)
            processed += 1

        self._update_snapshots()
        self._maybe_print_diag()

    def _consume_pending_goals(self) -> None:
        with self._lock:
            pending = dict(self._pending_goals)
            self._pending_goals.clear()
        if not pending:
            return
        self._goals.update(pending)
        if self._state not in ("PLANNING", "EXECUTING", "RESETTING"):
            self._set_state("GOAL_READY")

    def _handle_main_thread_request(self, request: dict[str, Any]) -> dict[str, Any]:
        self._consume_pending_goals()
        command = request.get("command")
        if command == "plan":
            return self._handle_plan(request)
        if command == "execute":
            return self._handle_execute()
        if command == "preview_ik":
            return self._handle_preview_ik(request)
        if command == "get_collision_spheres":
            return self._handle_collision_spheres()
        if command == "reset_goals":
            return self._handle_reset_goals()
        return {"success": False, "message": f"unknown command: {command!r}", "state": self._state}

    def _handle_collision_spheres(self) -> dict[str, Any]:
        if not self._enable_planning:
            return {
                "success": False,
                "message": "collision spheres unavailable while planning is disabled by SAFE_MODE",
                "state": self._state,
            }
        snapshot = self._planner.collision_spheres_snapshot()
        snapshot.update({
            "success": True,
            "message": "cuRobo collision spheres",
            "state": self._state,
        })
        return snapshot

    def _handle_preview_ik(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self._enable_ik_preview:
            return {"success": False, "message": "IK preview disabled by SAFE_MODE", "state": self._state}
        if self._state in ("PLANNING", "EXECUTING", "RESETTING"):
            return {"success": False, "message": f"reject IK preview while state={self._state}", "state": self._state}

        side = request.get("side")
        if side not in ("left", "right"):
            return {"success": False, "message": f"invalid IK side: {side!r}", "state": self._state}

        pose_data = request.get("pose")
        if pose_data is not None:
            try:
                goal = StoredGoal(
                    [float(v) for v in pose_data.get("position", [])],
                    [float(v) for v in pose_data.get("quaternion_xyzw", [])],
                )
            except (TypeError, ValueError):
                return {"success": False, "message": "IK pose contains non-numeric values", "state": self._state}
            if len(goal.position) != 3 or len(goal.quaternion_xyzw) != 4:
                return {"success": False, "message": "IK pose must contain position[3] and quaternion_xyzw[4]", "state": self._state}
        else:
            self._consume_pending_goals()
            if side not in self._goals:
                self.reset_goals_from_current_tcp()
            goal = self._goals[side]

        poser = self._get_ik_poser(side)
        from usd.schema.isaac.robot_schema import math as poser_math

        qx, qy, qz, qw = goal.quaternion_xyzw
        target = poser_math.Transform(t=goal.position, q=[qw, qx, qy, qz])
        result = poser.solve_ik(target)
        joint_values = {}
        try:
            from pxr import Sdf

            joint_values = {Sdf.Path(path).name: float(value) for path, value in result.joints.items()}
        except Exception:
            joint_values = {str(path).split("/")[-1]: float(value) for path, value in result.joints.items()}

        if result.success:
            # Feed the next preview solve from the last preview solution only.
            # This does not modify the real articulation.
            seed_by_name = joint_values
            poser.set_seed(
                [seed_by_name.get(str(joint.prim_path).split("/")[-1], 0.0) for joint in poser.joints]
            )
        return {
            "success": bool(result.success),
            "message": f"{side} IK preview {'ok' if result.success else 'failed'}",
            "state": self._state,
            "side": side,
            "joints": joint_values,
        }

    def _handle_plan(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self._enable_planning:
            return {"success": False, "message": "planning disabled by SAFE_MODE", "state": self._state}
        if self._state in ("PLANNING", "EXECUTING", "RESETTING"):
            return {"success": False, "message": f"reject plan while state={self._state}", "state": self._state}

        mode = request.get("mode")
        if mode not in ("left", "right", "dual"):
            return {"success": False, "message": f"invalid mode: {mode!r}", "state": self._state}

        self._plan_request_id += 1
        request_id = self._plan_request_id
        print(f"RM75 socket backend: [PLAN_REQUEST] id={request_id} mode={mode}")
        self._set_state("PLANNING")
        started = time.perf_counter()
        try:
            pose_map = self._pose_map_for_mode(mode)
            print(f"RM75 socket backend: [PLAN_START] id={request_id} mode={mode}")
            plan = self._planner.plan_to_poses(mode, pose_map)
            duration = time.perf_counter() - started
            if plan is None:
                self._set_state("GOAL_READY")
                print(f"RM75 socket backend: [PLAN_END] id={request_id} success=false duration={duration:.3f}s")
                detail = self._planner.last_failure or "no trajectory returned"
                return {
                    "success": False,
                    "message": f"{mode} planning failed: {detail}",
                    "state": self._state,
                }

            self._plan_count += 1
            self._set_state("PLAN_READY")
            print(
                f"RM75 socket backend: [PLAN_END] id={request_id} success=true "
                f"duration={duration:.3f}s points={plan.positions.shape[0]}"
            )
            return {
                "success": True,
                "message": f"{mode} plan ready: {plan.duration:.2f}s, {plan.positions.shape[0]} points",
                "state": self._state,
                "trajectory": {
                    "joint_names": ALL_JOINTS.copy(),
                    "dt": plan.dt,
                    "points": plan.positions.tolist(),
                    "duration": plan.duration,
                },
            }
        except Exception:
            self._set_state("GOAL_READY")
            raise

    def _handle_execute(self) -> dict[str, Any]:
        if not self._enable_execution:
            return {"success": False, "message": "execution disabled by SAFE_MODE", "state": self._state}
        if self._state in ("PLANNING", "RESETTING"):
            return {"success": False, "message": f"reject execute while state={self._state}", "state": self._state}
        if self._state == "EXECUTING":
            return {"success": False, "message": "already executing", "state": self._state}
        if not self._planner.has_plan:
            return {"success": False, "message": "no executable trajectory", "state": self._state}

        ok = self._planner.execute()
        if ok:
            self._set_state("EXECUTING")
        return {
            "success": bool(ok),
            "message": "executing latest trajectory" if ok else "execute rejected by planner",
            "state": self._state,
        }

    def _handle_reset_goals(self) -> dict[str, Any]:
        if self._state in ("PLANNING", "RESETTING"):
            return {"success": False, "message": f"reject reset while state={self._state}", "state": self._state}
        self._set_state("RESETTING")
        self._planner.clear_plan()
        with self._lock:
            self._pending_goals.clear()
        self.reset_goals_from_current_tcp()
        self._update_snapshots()
        self._set_state("IDLE")
        return {
            "success": True,
            "message": "goals reset from current Isaac TCP poses; plan cleared",
            "state": self._state,
            "goals": self._goals_json(),
        }

    def _update_execution_state(self) -> None:
        if self._state == "EXECUTING" and not self._planner.is_executing:
            self._set_state("IDLE")

    def _pose_map_for_mode(self, mode: Literal["left", "right", "dual"]):
        required_sides = ("left", "right") if mode == "dual" else (mode,)
        missing = [side for side in required_sides if side not in self._goals]
        if missing:
            self.reset_goals_from_current_tcp()
        pose_map = {}
        for frame in MODE_TOOL_FRAMES[mode]:
            side = "left" if frame.startswith("l_") else "right"
            pose_map[frame] = self._goals[side].curobo
        return pose_map

    def reset_goals_from_current_tcp(self) -> None:
        for side, frame_name in TOOL_FRAME_BY_SIDE.items():
            self._goals[side] = self._read_tool_pose(frame_name)

    def _get_ik_poser(self, side: Literal["left", "right"]):
        poser = self._ik_posers.get(side)
        if poser is not None:
            return poser
        from isaacsim.robot.poser import robot_poser

        start_name, end_name = CHAIN_LINKS[side]
        start = self._find_robot_descendant(start_name)
        end = self._find_robot_descendant(end_name)
        if start is None or end is None:
            raise RuntimeError(f"IK preview chain is missing: {start_name} -> {end_name}")
        poser = robot_poser.RobotPoser(self._stage, self._robot_prim, start, end)
        if not poser.joints:
            raise RuntimeError(f"IK preview chain has no joints: {side}")
        seed = {}
        joint_by_name = self._actual_joint_dict()
        for joint in poser.joints:
            name = str(joint.prim_path).split("/")[-1]
            if name in joint_by_name:
                seed[joint.prim_path] = joint_by_name[name]
        if seed:
            poser.set_seed(
                [seed.get(joint.prim_path, 0.0) for joint in poser.joints]
            )
        self._ik_posers[side] = poser
        print(f"RM75 socket backend: initialized {side} IK preview poser")
        return poser

    def _goals_json(self) -> dict[str, dict[str, list[float]]]:
        return {side: goal.as_json() for side, goal in self._goals.items()}

    def _joints_json(self) -> dict[str, Any]:
        positions = self._articulation.get_dof_positions().numpy().reshape(-1)
        for index, value in zip(self._locked_dof_indices, self._locked_dof_positions):
            positions[index] = value
        return {"names": self._dof_names.copy(), "positions": [float(value) for value in positions]}

    def _actual_joint_dict(self) -> dict[str, float]:
        data = self._joints_json()
        return {name: float(value) for name, value in zip(data["names"], data["positions"])}

    def _hold_locked_joints(self) -> None:
        """Keep cuRobo-locked passive Isaac joints from polluting q_actual.

        The head/lift/gripper joints are not part of the 14-DOF planner. In
        Isaac they can still appear as articulation DOFs, so if physics makes a
        passive head joint wobble RViz would visualize that wobble. Publishing
        is always clamped in ``_joints_json``. This optional hold only refreshes
        position targets at low rate; it deliberately does not teleport joints
        every frame.
        """
        if not self._locked_dof_indices or not self._enable_locked_target_hold:
            return
        now = time.monotonic()
        if now - self._last_locked_hold_time < self._locked_hold_interval:
            return
        self._last_locked_hold_time = now
        try:
            import numpy as np

            values = np.asarray([self._locked_dof_positions], dtype=np.float32)
            self._articulation.set_dof_position_targets(values, dof_indices=self._locked_dof_indices)
        except Exception as exc:  # noqa: BLE001 - report once through status path
            if not self._reported_locked_hold_error:
                self._reported_locked_hold_error = True
                self._set_status(f"locked joint hold failed: {exc!r}")

    def _update_snapshots(self) -> None:
        joints = self._joints_json()
        goals = self._goals_json()
        with self._lock:
            self._last_joints = joints
            self._goal_snapshot = goals
            self._joint_snapshot_count += 1

    def _snapshot_state(self) -> dict[str, Any]:
        with self._lock:
            status = self._status
            goals = json.loads(json.dumps(self._goal_snapshot))
            joints = json.loads(json.dumps(self._last_joints))
            socket_rx_count = self._socket_rx_count
            goal_update_count = self._goal_update_count
            joint_snapshot_count = self._joint_snapshot_count
        return {
            "success": True,
            "message": status,
            "state": self._state,
            "goals": goals,
            "joints": joints,
            "obstacle_scene": {
                "frame_id": self._obstacle_scene.frame_id,
                "count": len(self._obstacle_scene.cuboids),
                "fingerprint": self._obstacle_scene.fingerprint,
            },
            "diag": {
                "state": self._state,
                "queue_size": self._requests.qsize(),
                "plan_count": self._plan_count,
                "socket_rx_count": socket_rx_count,
                "goal_update_count": goal_update_count,
                "joint_snapshot_count": joint_snapshot_count,
                "threads": threading.active_count(),
            },
        }

    def _maybe_print_diag(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_diag_time
        if elapsed < 1.0:
            return
        with self._lock:
            socket_rx = self._socket_rx_count
            goal_updates = self._goal_update_count
            joint_snapshots = self._joint_snapshot_count
            self._socket_rx_count = 0
            self._goal_update_count = 0
            self._joint_snapshot_count = 0
        rss_gb = None
        try:
            import resource

            rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
        except Exception:
            pass
        rss_text = f" rss={rss_gb:.2f}GB" if rss_gb is not None else ""
        print(
            "RM75 socket backend: [DIAG] "
            f"state={self._state} socket_rx_rate={socket_rx / elapsed:.1f}/s "
            f"goal_update_rate={goal_updates / elapsed:.1f}/s "
            f"joint_snapshot_rate={joint_snapshots / elapsed:.1f}/s "
            f"plan_count={self._plan_count} queue_size={self._requests.qsize()} "
            f"threads={threading.active_count()}{rss_text}"
        )
        self._last_diag_time = now

    def _read_tool_pose(self, frame_name: str) -> StoredGoal:
        from pxr import Usd, UsdGeom

        prim = self._find_robot_descendant(frame_name)
        if prim is None:
            raise RuntimeError(f"Cannot find tool frame in Isaac stage: {frame_name}")
        world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = world.ExtractTranslation()
        quat = world.ExtractRotation().GetQuat()
        imag = quat.GetImaginary()
        return StoredGoal(
            position=[float(translation[0]), float(translation[1]), float(translation[2])],
            quaternion_xyzw=[float(imag[0]), float(imag[1]), float(imag[2]), float(quat.GetReal())],
        )

    def _find_robot_descendant(self, name: str):
        from pxr import Usd

        for prim in Usd.PrimRange(self._robot_prim):
            if prim.GetName() == name:
                return prim
        return None

    def destroy(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1.0)
