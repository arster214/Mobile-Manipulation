#!/usr/bin/env python3
"""Interactive cuRobo motion planning for the RM75 dual-arm scene."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

import numpy as np
import yaml

from rm75_obstacles import load_obstacle_scene


ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = ROOT / "config" / "rm75_dual.yml"

LEFT_JOINTS = [f"l_joint{i}" for i in range(1, 8)]
RIGHT_JOINTS = [f"r_joint{i}" for i in range(1, 8)]
ALL_JOINTS = LEFT_JOINTS + RIGHT_JOINTS
VALID_MODES = ("left", "right", "dual")
MODE_TOOL_FRAMES = {
    "left": ["l_tool0"],
    "right": ["r_tool0"],
    "dual": ["l_tool0", "r_tool0"],
}
TOOL_POSE_NAMES = {
    "l_tool0": "left_arm_poser",
    "r_tool0": "right_arm_poser",
}


def make_mode_robot_config(
    base_config: dict,
    mode: str,
    locked_joint_positions: dict[str, float] | None = None,
) -> dict:
    """Create a cuRobo robot config for one arm or both arms.

    Single-arm configs remove the inactive arm from cspace and add its joints
    to ``lock_joints``. The inactive arm remains in the kinematic/collision
    model at the supplied joint positions.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown planning mode: {mode}")

    config = deepcopy(base_config)
    kinematics = config["kinematics"]
    cspace = kinematics["cspace"]
    original_names = list(cspace["joint_names"])

    if mode == "dual":
        kinematics["tool_frames"] = MODE_TOOL_FRAMES[mode].copy()
        return config

    active_names = LEFT_JOINTS if mode == "left" else RIGHT_JOINTS
    inactive_names = RIGHT_JOINTS if mode == "left" else LEFT_JOINTS
    active_indices = [original_names.index(name) for name in active_names]

    for key, value in list(cspace.items()):
        if isinstance(value, list) and len(value) == len(original_names):
            cspace[key] = [value[index] for index in active_indices]
    cspace["joint_names"] = active_names.copy()
    kinematics["tool_frames"] = MODE_TOOL_FRAMES[mode].copy()

    locked = locked_joint_positions or {}
    for name in inactive_names:
        kinematics["lock_joints"][name] = float(locked.get(name, 0.0))
    return config


@dataclass
class PlannedTrajectory:
    mode: str
    positions: np.ndarray
    dt: float
    planning_time: float

    @property
    def duration(self) -> float:
        return max(0, self.positions.shape[0] - 1) * self.dt


class RM75MotionPlanner:
    """Connect Robot Poser targets to cuRobo planning and Isaac execution."""

    def __init__(self, stage, robot_prim, *, build_window: bool = True) -> None:
        import omni.timeline
        from isaacsim.core.experimental.prims import Articulation

        self._stage = stage
        self._robot_prim = robot_prim
        self._robot_config = yaml.safe_load(ROBOT_CONFIG.read_text())
        self._world_config = load_obstacle_scene().as_curobo_dict()
        self._articulation = Articulation(str(robot_prim.GetPath()))
        self._timeline = omni.timeline.get_timeline_interface()
        self._planners = {}
        # Single-arm planners embed the inactive arm as locked kinematics.
        # Keep the lock snapshot so a reused planner can be invalidated when
        # the measured inactive-arm state changes.
        self._planner_lock_snapshots: dict[str, dict[str, float]] = {}
        self._plan: PlannedTrajectory | None = None
        self._last_failure = ""
        self._executing = False
        self._execution_elapsed = 0.0
        self._execution_last_update = 0.0
        self._last_execution_index = -1
        self._status_callback: Callable[[str], None] | None = None
        self._window = None

        self._dof_names = list(self._articulation.dof_names)
        missing = [name for name in ALL_JOINTS if name not in self._dof_names]
        if missing:
            raise RuntimeError(f"Articulation is missing planning joints: {missing}")
        self._planning_dof_indices = [self._dof_names.index(name) for name in ALL_JOINTS]

        if build_window:
            self._build_window()

    def _build_window(self) -> None:
        import omni.ui as ui

        self._window = ui.Window(
            "RM75 Motion Planner",
            width=360,
            height=170,
            visible=True,
            position_x=20,
            position_y=120,
        )
        with self._window.frame:
            with ui.VStack(spacing=8):
                with ui.HStack(height=34, spacing=6):
                    for mode in VALID_MODES:
                        ui.Button(mode, clicked_fn=lambda selected=mode: self.plan(selected))
                with ui.HStack(height=34, spacing=6):
                    ui.Button("Execute", clicked_fn=self.execute)
                    ui.Button("Stop", clicked_fn=self.stop)
                self._status_label = ui.Label("Ready", word_wrap=True, height=54)
        self._status_callback = self._set_status_label
        self._window.visible = True

    def _set_status_label(self, message: str) -> None:
        if self._window is not None:
            self._status_label.text = message

    def _set_status(self, message: str) -> None:
        print(f"RM75 planner: {message}")
        if self._status_callback is not None:
            self._status_callback(message)

    def set_status_callback(self, callback: Callable[[str], None] | None) -> None:
        """Install a status callback, primarily for headless integration tests."""
        self._status_callback = callback

    def _read_joint_positions(self) -> np.ndarray:
        positions = self._articulation.get_dof_positions(
            dof_indices=self._planning_dof_indices
        ).numpy()
        return np.asarray(positions, dtype=np.float32).reshape(-1)

    def _joint_position_dict(self, positions: np.ndarray) -> dict[str, float]:
        return {name: float(positions[index]) for index, name in enumerate(ALL_JOINTS)}

    def _create_planner(self, mode: str, full_positions: np.ndarray):
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        locked = self._joint_position_dict(full_positions)
        robot_config = make_mode_robot_config(self._robot_config, mode, locked)
        planner_config = MotionPlannerCfg.create(
            robot=robot_config,
            scene_model=self._world_config,
            num_ik_seeds=32,
            num_trajopt_seeds=4,
            position_tolerance=0.005,
            orientation_tolerance=0.05,
            optimizer_collision_activation_distance=0.01,
            interpolation_dt=0.025,
            interpolation_buffer_size=1000,
            use_cuda_graph=True,
        )
        return MotionPlanner(planner_config)

    def _single_arm_lock_snapshot(
        self, mode: str, full_positions: np.ndarray
    ) -> dict[str, float]:
        if mode == "left":
            names = RIGHT_JOINTS
        elif mode == "right":
            names = LEFT_JOINTS
        else:
            return {}
        by_name = self._joint_position_dict(full_positions)
        return {name: by_name[name] for name in names}

    def _lock_snapshot_changed(
        self, previous: dict[str, float] | None, current: dict[str, float]
    ) -> bool:
        if previous is None or previous.keys() != current.keys():
            return True
        # Tiny physics noise should not force CUDA graph/model rebuilds, while
        # a real movement of the locked arm must never use stale kinematics.
        return any(abs(previous[name] - current[name]) > 1.0e-4 for name in current)

    def _destroy_planner(self, mode: str) -> None:
        planner = self._planners.pop(mode, None)
        self._planner_lock_snapshots.pop(mode, None)
        if planner is not None:
            destroy = getattr(planner, "destroy", None)
            if destroy is not None:
                destroy()

    def _get_planner(self, mode: str, full_positions: np.ndarray):
        planner = self._planners.get(mode)
        lock_snapshot = self._single_arm_lock_snapshot(mode, full_positions)
        if planner is None:
            self._set_status(f"Initializing {mode} planner...")
            planner = self._create_planner(mode, full_positions)
            self._planners[mode] = planner
            self._planner_lock_snapshots[mode] = lock_snapshot
        elif mode != "dual" and self._lock_snapshot_changed(
            self._planner_lock_snapshots.get(mode), lock_snapshot
        ):
            self._set_status(f"Refreshing {mode} planner for current locked-arm state...")
            self._destroy_planner(mode)
            planner = self._create_planner(mode, full_positions)
            self._planners[mode] = planner
            self._planner_lock_snapshots[mode] = lock_snapshot
        return planner

    @staticmethod
    def _validate_pose_map(
        mode: str,
        pose_by_frame: dict[str, tuple[list[float], list[float]]],
    ) -> None:
        for frame in MODE_TOOL_FRAMES[mode]:
            if frame not in pose_by_frame:
                raise RuntimeError(f"Missing external target pose for {frame}")
            position, quaternion = pose_by_frame[frame]
            if len(position) != 3 or len(quaternion) != 4:
                raise ValueError(f"Invalid pose shape for {frame}")
            values = np.asarray([*position, *quaternion], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite target pose for {frame}")
            norm = float(np.linalg.norm(np.asarray(quaternion, dtype=np.float64)))
            if norm < 1.0e-8:
                raise ValueError(f"Zero-norm target quaternion for {frame}")

    def _get_goal(self, mode: str, planner):
        import torch
        from curobo.types import GoalToolPose
        from isaacsim.robot.poser import robot_poser

        positions = []
        quaternions = []
        for frame in MODE_TOOL_FRAMES[mode]:
            named_pose = robot_poser.get_named_pose(
                self._stage, self._robot_prim, TOOL_POSE_NAMES[frame]
            )
            if (
                named_pose is None
                or named_pose.target_position is None
                or named_pose.target_orientation is None
            ):
                raise RuntimeError(f"Robot Poser target is unavailable: {frame}")
            positions.append(named_pose.target_position)
            quaternions.append(named_pose.target_orientation)

        device = planner.device_cfg.device
        dtype = planner.device_cfg.dtype
        position_tensor = torch.as_tensor(positions, device=device, dtype=dtype)
        quaternion_tensor = torch.as_tensor(quaternions, device=device, dtype=dtype)
        quaternion_tensor = quaternion_tensor / torch.linalg.vector_norm(
            quaternion_tensor, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        return GoalToolPose(
            tool_frames=MODE_TOOL_FRAMES[mode].copy(),
            position=position_tensor.view(1, 1, len(positions), 1, 3),
            quaternion=quaternion_tensor.view(1, 1, len(quaternions), 1, 4),
        )

    def _get_goal_from_pose_map(
        self,
        mode: str,
        planner,
        pose_by_frame: dict[str, tuple[list[float], list[float]]],
    ):
        """Build a cuRobo goal from explicit tool poses.

        ``pose_by_frame`` maps cuRobo tool frame names to
        ``([x, y, z], [w, x, y, z])`` in the cuRobo base frame.
        """
        import torch
        from curobo.types import GoalToolPose

        positions = []
        quaternions = []
        for frame in MODE_TOOL_FRAMES[mode]:
            if frame not in pose_by_frame:
                raise RuntimeError(f"Missing external target pose for {frame}")
            position, quaternion_wxyz = pose_by_frame[frame]
            positions.append(position)
            quaternions.append(quaternion_wxyz)

        device = planner.device_cfg.device
        dtype = planner.device_cfg.dtype
        position_tensor = torch.as_tensor(positions, device=device, dtype=dtype)
        quaternion_tensor = torch.as_tensor(quaternions, device=device, dtype=dtype)
        quaternion_tensor = quaternion_tensor / torch.linalg.vector_norm(
            quaternion_tensor, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        return GoalToolPose(
            tool_frames=MODE_TOOL_FRAMES[mode].copy(),
            position=position_tensor.view(1, 1, len(positions), 1, 3),
            quaternion=quaternion_tensor.view(1, 1, len(quaternions), 1, 4),
        )

    @staticmethod
    def _tensor_summary(value) -> str:
        """Return a compact scalar summary without flooding the backend log."""
        if value is None:
            return "n/a"
        try:
            flat = value.detach().reshape(-1)
            if flat.numel() == 0:
                return "empty"
            if flat.dtype == __import__("torch").bool:
                return f"{int(flat.count_nonzero().item())}/{flat.numel()} true"
            return f"min={float(flat.min().item()):.6g}, max={float(flat.max().item()):.6g}"
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return str(value)

    def _diagnose_pose_failure(self, planner, goal, current_state) -> str:
        """Classify a failed pose plan without changing the planning target."""
        try:
            ik_result = planner.ik_solver.solve_pose(
                goal,
                return_seeds=planner.trajopt_solver.config.num_seeds,
                current_state=current_state,
            )
            success_count = int(ik_result.success.count_nonzero().item())
            seed_count = int(ik_result.success.numel())
            details = (
                f"cuRobo IK {success_count}/{seed_count}; "
                f"position_error({self._tensor_summary(ik_result.position_error)}); "
                f"rotation_error({self._tensor_summary(ik_result.rotation_error)}); "
                f"feasible({self._tensor_summary(ik_result.feasible)})"
            )
            if success_count == 0:
                feasible = ik_result.feasible
                if feasible is not None and int(feasible.count_nonzero().item()) == 0:
                    reason = "all returned IK candidates violate collision or joint constraints"
                else:
                    reason = "IK candidates did not jointly satisfy pose tolerances and constraints"
                return reason + "; " + details
            return "IK succeeded but trajectory optimization failed; " + details
        except Exception as exc:
            return f"cuRobo failure diagnostics unavailable: {exc!r}"

    @staticmethod
    def _trajectory_positions(trajectory) -> np.ndarray:
        positions = trajectory.position.detach().cpu().numpy()
        while positions.ndim > 2:
            positions = positions[0]
        if positions.ndim != 2:
            raise RuntimeError(f"Unexpected trajectory shape: {positions.shape}")
        return np.asarray(positions, dtype=np.float32)

    def _expand_trajectory(
        self,
        active_positions: np.ndarray,
        active_joint_names: list[str],
        start_positions: np.ndarray,
    ) -> np.ndarray:
        full = np.repeat(start_positions.reshape(1, -1), active_positions.shape[0], axis=0)
        for source_index, name in enumerate(active_joint_names):
            full[:, ALL_JOINTS.index(name)] = active_positions[:, source_index]
        return full

    def plan(self, mode: str) -> PlannedTrajectory | None:
        """Plan from the measured articulation state to Robot Poser targets."""
        import torch
        from curobo.types import JointState

        if mode not in VALID_MODES:
            raise ValueError(f"Unknown planning mode: {mode}")

        self.stop(update_status=False)
        self._plan = None
        try:
            start_positions = self._read_joint_positions()
            planner = self._get_planner(mode, start_positions)

            start_by_name = self._joint_position_dict(start_positions)
            active_start = [start_by_name[name] for name in planner.joint_names]
            current_state = JointState.from_position(
                torch.as_tensor(
                    active_start,
                    device=planner.device_cfg.device,
                    dtype=planner.device_cfg.dtype,
                ).view(1, -1),
                joint_names=planner.joint_names,
            )
            goal = self._get_goal(mode, planner)
            current_tool_poses = planner.compute_kinematics(current_state).tool_poses.as_goal()
            position_delta = float(
                torch.max(torch.abs(current_tool_poses.position - goal.position)).item()
            )
            quaternion_alignment = torch.abs(
                torch.sum(
                    current_tool_poses.quaternion * goal.quaternion,
                    dim=-1,
                )
            ).clamp(max=1.0)
            orientation_delta = float(
                torch.max(2.0 * torch.acos(quaternion_alignment)).item()
            )
            print(
                f"RM75 planner: {mode} target delta from current FK: "
                f"position={position_delta:.6f} m, orientation={orientation_delta:.6f} rad"
            )
            self._set_status(f"Planning {mode}...")
            started = time.perf_counter()
            result = planner.plan_pose(goal, current_state, max_attempts=5)
            elapsed = time.perf_counter() - started

            if result is None or not bool(torch.any(result.success).item()):
                if result is not None:
                    details = []
                    for name in ("position_error", "rotation_error", "feasible", "cspace_error"):
                        value = getattr(result, name, None)
                        if value is not None:
                            try:
                                details.append(f"{name}={value.detach().cpu().numpy().tolist()}")
                            except AttributeError:
                                details.append(f"{name}={value}")
                    if details:
                        print("RM75 planner failure details: " + ", ".join(details))
                self._set_status(f"{mode} planning failed")
                return None

            trajectory = result.get_interpolated_plan()
            active_positions = self._trajectory_positions(trajectory)
            full_positions = self._expand_trajectory(
                active_positions, list(planner.joint_names), start_positions
            )
            self._plan = PlannedTrajectory(
                mode=mode,
                positions=full_positions,
                dt=float(planner.trajopt_solver.config.interpolation_dt),
                planning_time=elapsed,
            )
            self._set_status(
                f"{mode} ready | {self._plan.duration:.2f}s | "
                f"{full_positions.shape[0]} points | planned in {elapsed:.2f}s"
            )
            return self._plan
        except Exception as exc:
            self._set_status(f"{mode} error: {exc}")
            print(f"RM75 planner exception ({mode}): {exc!r}")
            return None

    def plan_to_poses(
        self,
        mode: str,
        pose_by_frame: dict[str, tuple[list[float], list[float]]],
    ) -> PlannedTrajectory | None:
        """Plan from the measured articulation state to explicit external poses.

        This is the ROS2/RViz entry point. Poses are expressed in the cuRobo
        base frame (``base_link_underpan``) and quaternions use cuRobo order
        ``[w, x, y, z]``.
        """
        import torch
        from curobo.types import JointState

        if mode not in VALID_MODES:
            raise ValueError(f"Unknown planning mode: {mode}")
        self._validate_pose_map(mode, pose_by_frame)

        self.stop(update_status=False)
        self._plan = None
        self._last_failure = ""
        try:
            start_positions = self._read_joint_positions()
            planner = self._get_planner(mode, start_positions)

            start_by_name = self._joint_position_dict(start_positions)
            active_start = [start_by_name[name] for name in planner.joint_names]
            current_state = JointState.from_position(
                torch.as_tensor(
                    active_start,
                    device=planner.device_cfg.device,
                    dtype=planner.device_cfg.dtype,
                ).view(1, -1),
                joint_names=planner.joint_names,
            )
            goal = self._get_goal_from_pose_map(mode, planner, pose_by_frame)
            print(
                f"RM75 planner: {mode} q_start "
                + ", ".join(
                    f"{name}={value:.6f}"
                    for name, value in zip(planner.joint_names, active_start)
                )
            )
            for frame, (position, quaternion_wxyz) in pose_by_frame.items():
                print(
                    f"RM75 planner: {mode} X_goal[{frame}] "
                    f"position={[round(float(v), 6) for v in position]} "
                    f"quaternion_wxyz={[round(float(v), 6) for v in quaternion_wxyz]}"
                )
            current_tool_poses = planner.compute_kinematics(current_state).tool_poses.as_goal()
            position_delta = float(
                torch.max(torch.abs(current_tool_poses.position - goal.position)).item()
            )
            quaternion_alignment = torch.abs(
                torch.sum(
                    current_tool_poses.quaternion * goal.quaternion,
                    dim=-1,
                )
            ).clamp(max=1.0)
            orientation_delta = float(
                torch.max(2.0 * torch.acos(quaternion_alignment)).item()
            )
            print(
                f"RM75 planner: {mode} external target delta from current FK: "
                f"position={position_delta:.6f} m, orientation={orientation_delta:.6f} rad"
            )
            current_positions = current_tool_poses.position.detach().reshape(-1, 3)
            current_quaternions = current_tool_poses.quaternion.detach().reshape(-1, 4)
            for index, frame in enumerate(MODE_TOOL_FRAMES[mode]):
                print(
                    f"RM75 planner: {mode} FK(q_start)[{frame}] "
                    f"position={[round(float(v), 6) for v in current_positions[index]]} "
                    f"quaternion_wxyz={[round(float(v), 6) for v in current_quaternions[index]]}"
                )
            self._set_status(f"Planning {mode}...")
            started = time.perf_counter()
            result = planner.plan_pose(goal, current_state, max_attempts=5)
            elapsed = time.perf_counter() - started

            if result is None or not bool(torch.any(result.success).item()):
                self._last_failure = self._diagnose_pose_failure(
                    planner, goal, current_state
                )
                if result is not None:
                    details = []
                    for name in ("position_error", "rotation_error", "feasible", "cspace_error"):
                        value = getattr(result, name, None)
                        if value is not None:
                            try:
                                details.append(f"{name}={value.detach().cpu().numpy().tolist()}")
                            except AttributeError:
                                details.append(f"{name}={value}")
                    if details:
                        print("RM75 planner failure details: " + ", ".join(details))
                self._set_status(f"{mode} planning failed: {self._last_failure}")
                return None

            trajectory = result.get_interpolated_plan()
            active_positions = self._trajectory_positions(trajectory)
            full_positions = self._expand_trajectory(
                active_positions, list(planner.joint_names), start_positions
            )
            self._plan = PlannedTrajectory(
                mode=mode,
                positions=full_positions,
                dt=float(planner.trajopt_solver.config.interpolation_dt),
                planning_time=elapsed,
            )
            self._set_status(
                f"{mode} ready | {self._plan.duration:.2f}s | "
                f"{full_positions.shape[0]} points | planned in {elapsed:.2f}s"
            )
            return self._plan
        except Exception as exc:
            self._last_failure = repr(exc)
            self._set_status(f"{mode} error: {exc}")
            print(f"RM75 planner external exception ({mode}): {exc!r}")
            return None

    def collision_spheres_snapshot(self) -> dict:
        """Return current world-frame collision spheres from cuRobo.

        This is a read-only visualization path. It uses the same dual-arm
        kinematics and collision-sphere model as planning.
        """
        import torch
        from curobo.types import JointState

        start_positions = self._read_joint_positions()
        planner = self._get_planner("dual", start_positions)
        start_by_name = self._joint_position_dict(start_positions)
        active_start = [start_by_name[name] for name in planner.joint_names]
        current_state = JointState.from_position(
            torch.as_tensor(
                active_start,
                device=planner.device_cfg.device,
                dtype=planner.device_cfg.dtype,
            ).view(1, -1),
            joint_names=planner.joint_names,
        )
        kinematics_state = planner.compute_kinematics(current_state)
        spheres = kinematics_state.robot_spheres
        if spheres is None:
            raise RuntimeError("cuRobo kinematics did not return collision spheres")

        kin_config = planner.kinematics.config.kinematics_config
        link_by_index = {
            int(index): name
            for name, index in kin_config.link_name_to_idx_map.items()
        }
        sphere_link_indices = (
            kin_config.link_sphere_idx_map.detach().cpu().reshape(-1).tolist()
        )
        sphere_values = spheres.detach().cpu().reshape(-1, 4).tolist()
        result = []
        for index, values in enumerate(sphere_values):
            x, y, z, radius = [float(value) for value in values]
            if radius <= 0.0:
                continue
            link_index = int(sphere_link_indices[index])
            result.append({
                "link": link_by_index.get(link_index, f"link_index_{link_index}"),
                "position": [x, y, z],
                "radius": radius,
            })
        return {
            "frame_id": "base_link_underpan",
            "joint_names": list(planner.joint_names),
            "joint_positions": [float(value) for value in active_start],
            "spheres": result,
        }

    def execute(self) -> bool:
        """Execute the latest trajectory using articulation position targets."""
        if self._plan is None:
            self._set_status("No trajectory to execute")
            return False

        current = self._read_joint_positions()
        start_error = float(np.max(np.abs(current - self._plan.positions[0])))
        if start_error > 0.10:
            self._set_status(f"Robot moved {start_error:.3f} rad; plan again")
            return False

        self._execution_elapsed = 0.0
        self._execution_last_update = time.perf_counter()
        self._last_execution_index = -1
        self._executing = True
        self._set_status(f"Executing {self._plan.mode}...")
        return True

    def stop(self, *, update_status: bool = True) -> None:
        was_executing = self._executing
        self._executing = False
        self._last_execution_index = -1
        if update_status and was_executing:
            self._set_status("Execution stopped")

    def clear_plan(self) -> None:
        """Stop execution and discard the latest planned trajectory."""
        self.stop(update_status=False)
        self._plan = None
        self._set_status("Plan cleared")

    @property
    def is_executing(self) -> bool:
        return self._executing

    @property
    def has_plan(self) -> bool:
        return self._plan is not None

    @property
    def last_failure(self) -> str:
        return self._last_failure

    def update(self) -> None:
        """Advance trajectory execution; call once per SimulationApp update."""
        if not self._executing or self._plan is None:
            return

        now = time.perf_counter()
        if not self._timeline.is_playing():
            self._execution_last_update = now
            return
        self._execution_elapsed += now - self._execution_last_update
        self._execution_last_update = now
        elapsed = self._execution_elapsed
        index = min(int(elapsed / self._plan.dt), self._plan.positions.shape[0] - 1)
        if index != self._last_execution_index:
            self._articulation.set_dof_position_targets(
                self._plan.positions[index].reshape(1, -1),
                dof_indices=self._planning_dof_indices,
            )
            self._last_execution_index = index

        if elapsed >= self._plan.duration:
            self._executing = False
            self._set_status(f"{self._plan.mode} execution complete")

    def destroy(self) -> None:
        self.stop(update_status=False)
        for planner in self._planners.values():
            planner.destroy()
        self._planners.clear()
        if self._window is not None:
            self._window.destroy()
            self._window = None
