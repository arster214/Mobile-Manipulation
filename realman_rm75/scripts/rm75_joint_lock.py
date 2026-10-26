"""Runtime mechanical-lock state machine for the RM75 arm joints.

The imported USD contains the real joint angle and effort limits, but it does
not contain the physical lock that the hardware uses at rest.  This controller
models that lock with a narrow temporary joint limit instead of relying on a
large viscous damping value.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time

import numpy as np


ARM_JOINT_NAMES = [f"l_joint{i}" for i in range(1, 8)] + [f"r_joint{i}" for i in range(1, 8)]


@dataclass
class _JointLimit:
    name: str
    prim: object
    lower_deg: float
    upper_deg: float
    effort: float


class RM75ArmJointLock:
    """Lock, unlock, move, settle, and re-lock the two RM75 arms."""

    LOCKED = "LOCKED"
    UNLOCKING = "UNLOCKING"
    MOVING = "MOVING"
    STOPPING = "STOPPING"

    def __init__(
        self,
        stage,
        robot_prim,
        articulation,
        *,
        stiffness: float,
        damping: float,
        max_force_cap: float,
    ) -> None:
        from pxr import UsdPhysics

        self._stage = stage
        self._articulation = articulation
        self._dof_names = list(articulation.dof_names)
        self._dof_indices = [self._dof_names.index(name) for name in ARM_JOINT_NAMES]
        self._stiffness = float(stiffness)
        self._damping = float(damping)
        self._max_force_cap = float(max_force_cap)
        self._lock_epsilon = float(os.environ.get("RM75_LOCK_EPSILON", "0.002"))
        self._unlock_duration = float(os.environ.get("RM75_UNLOCK_DURATION", "0.15"))
        self._settle_error = float(os.environ.get("RM75_SETTLE_ERROR", "0.003"))
        self._settle_velocity = float(os.environ.get("RM75_SETTLE_VELOCITY", "0.03"))
        self._settle_duration = float(os.environ.get("RM75_SETTLE_DURATION", "0.12"))
        self._target_change_threshold = float(os.environ.get("RM75_TARGET_CHANGE_THRESHOLD", "0.00001"))
        # The cuRobo trajectory is already time-parameterized. Do not add a
        # second velocity limiter by default; users can opt in for noisy goals.
        self._target_slew_rate = float(os.environ.get("RM75_ARM_TARGET_SLEW_RATE", "0.0"))
        # Newton can combine tensor actuation forces with an authored joint
        # drive. Keep the experimental direct-effort path opt-in; the default
        # position-drive path is the stable runtime behavior for this scene.
        self._use_direct_effort = os.environ.get("RM75_ARM_DIRECT_EFFORT", "0") == "1"
        self._torque_slew_rate = float(os.environ.get("RM75_ARM_TORQUE_SLEW_RATE", "800.0"))
        self._critical_damping_ratio = float(os.environ.get("RM75_ARM_DAMPING_RATIO", "1.0"))
        # Used only by the opt-in direct-effort controller. The normal
        # position-drive path uses the configured drive damping below.
        self._kd_max = float(os.environ.get("RM75_ARM_KD_MAX", "60.0"))
        self._kd_reported = False
        self._last_update = time.perf_counter()
        self._settle_since: float | None = None
        self._unlock_started = 0.0
        self._state = self.LOCKED
        self._was_executing = False
        self._lock_positions = np.zeros(len(ARM_JOINT_NAMES), dtype=np.float32)
        self._pending_targets = self._lock_positions.copy()
        self._command_targets = self._lock_positions.copy()
        self._applied_targets = self._lock_positions.copy()
        self._last_efforts = np.zeros(len(ARM_JOINT_NAMES), dtype=np.float32)
        self._limits: list[_JointLimit] = []

        for name in ARM_JOINT_NAMES:
            matches = [
                prim
                for prim in stage.Traverse()
                if prim.GetName() == name and prim.IsA(UsdPhysics.RevoluteJoint)
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one revolute joint named {name}, found {len(matches)}")
            prim = matches[0]
            lower = prim.GetAttribute("physics:lowerLimit").Get()
            upper = prim.GetAttribute("physics:upperLimit").Get()
            effort = prim.GetAttribute("urdf:limit:effort").Get()
            if lower is None or upper is None or effort is None:
                raise RuntimeError(f"Joint {name} is missing USD limit/effort metadata")
            self._limits.append(_JointLimit(name, prim, float(lower), float(upper), float(effort)))

        print(
            "RM75 arm lock controller: "
            f"joints={len(self._limits)}, stiffness={self._stiffness}, damping={self._damping}, "
            f"lock_epsilon={self._lock_epsilon} rad, unlock={self._unlock_duration}s, "
            f"control={'direct_effort' if self._use_direct_effort else 'position_drive'}, "
            f"effort_limits={[limit.effort for limit in self._limits]}"
        )

    @property
    def state(self) -> str:
        return self._state

    def _positions(self) -> np.ndarray:
        return np.asarray(
            self._articulation.get_dof_positions(dof_indices=self._dof_indices).numpy(),
            dtype=np.float32,
        ).reshape(-1)

    def _velocities(self) -> np.ndarray:
        return np.asarray(
            self._articulation.get_dof_velocities(dof_indices=self._dof_indices).numpy(),
            dtype=np.float32,
        ).reshape(-1)

    def _targets(self) -> np.ndarray:
        return np.asarray(
            self._articulation.get_dof_position_targets(dof_indices=self._dof_indices).numpy(),
            dtype=np.float32,
        ).reshape(-1)

    def _set_targets(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        self._articulation.set_dof_position_targets(
            values.reshape(1, -1),
            dof_indices=self._dof_indices,
        )
        self._applied_targets = values.copy()

    def _capture_external_target(self, requested: np.ndarray) -> bool:
        """Capture a target written by cuRobo/socket code since our last write."""
        changed = bool(np.max(np.abs(requested - self._applied_targets)) > self._target_change_threshold)
        if changed:
            self._command_targets = requested.copy()
        return changed

    def _slew_target(self, command: np.ndarray, dt: float) -> np.ndarray:
        if self._target_slew_rate <= 0.0 or dt <= 0.0:
            return command.copy()
        max_step = self._target_slew_rate * dt
        delta = np.clip(command - self._applied_targets, -max_step, max_step)
        return self._applied_targets + delta

    def _set_efforts(self, target: np.ndarray, current: np.ndarray, velocities: np.ndarray, dt: float) -> None:
        """Apply gravity-compensated PD effort with physical slew and limits."""
        gravity = np.asarray(
            self._articulation.get_dof_gravity_compensation_forces(
                dof_indices=self._dof_indices
            ).numpy(),
            dtype=np.float32,
        ).reshape(-1)
        effort_limits = np.asarray(
            [min(limit.effort, self._max_force_cap) if self._max_force_cap > 0.0 else limit.effort
             for limit in self._limits],
            dtype=np.float32,
        )
        damping = np.full(len(self._dof_indices), self._damping, dtype=np.float32)
        try:
            mass_matrix = np.asarray(self._articulation.get_mass_matrices().numpy(), dtype=np.float32)[0]
            effective_inertia = np.maximum(np.diag(mass_matrix)[self._dof_indices], 1e-6)
            damping = self._critical_damping_ratio * 2.0 * np.sqrt(self._stiffness * effective_inertia)
            if self._kd_max > 0.0:
                damping = np.minimum(damping, self._kd_max)
            if not self._kd_reported:
                print(
                    "RM75 arm lock: moving Kd from critical damping estimate, "
                    f"min={float(np.min(damping)):.3f}, max={float(np.max(damping)):.3f}"
                )
                self._kd_reported = True
        except Exception as exc:  # noqa: BLE001 - retain a usable fallback if a backend omits mass matrices
            if not self._kd_reported:
                print(f"RM75 arm lock: mass-matrix Kd unavailable ({exc!r}); using damping={self._damping}")
                self._kd_reported = True
        effort = gravity + self._stiffness * (target - current) - damping * velocities
        effort = np.clip(effort, -effort_limits, effort_limits)
        if self._torque_slew_rate > 0.0 and dt > 0.0:
            max_step = self._torque_slew_rate * dt
            effort = self._last_efforts + np.clip(effort - self._last_efforts, -max_step, max_step)
        self._articulation.set_dof_efforts(effort.reshape(1, -1), dof_indices=self._dof_indices)
        self._last_efforts = effort.copy()

    def _set_drive_gains(self, fraction: float) -> None:
        from pxr import UsdPhysics

        fraction = max(0.0, min(1.0, float(fraction)))
        for index, limit in enumerate(self._limits):
            drive = UsdPhysics.DriveAPI.Get(limit.prim, "angular")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(limit.prim, "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(self._stiffness * fraction)
            drive.CreateDampingAttr(self._damping * fraction)
            # The cap is an upper bound; the USD/URDF effort is the real limit.
            effort_limit = limit.effort
            if self._max_force_cap > 0.0:
                effort_limit = min(effort_limit, self._max_force_cap)
            drive.CreateMaxForceAttr(effort_limit * fraction if fraction > 0.0 else 0.001)

    def _restore_limits(self) -> None:
        for limit in self._limits:
            limit.prim.GetAttribute("physics:lowerLimit").Set(limit.lower_deg)
            limit.prim.GetAttribute("physics:upperLimit").Set(limit.upper_deg)

    def _lock_at(self, positions: np.ndarray) -> None:
        self._lock_positions = np.asarray(positions, dtype=np.float32).copy()
        epsilon_deg = math.degrees(self._lock_epsilon)
        for index, limit in enumerate(self._limits):
            center_deg = math.degrees(float(self._lock_positions[index]))
            lower = max(limit.lower_deg, center_deg - epsilon_deg)
            upper = min(limit.upper_deg, center_deg + epsilon_deg)
            if lower >= upper:
                raise RuntimeError(f"Cannot narrow limit for {limit.name}: {lower} >= {upper}")
            limit.prim.GetAttribute("physics:lowerLimit").Set(lower)
            limit.prim.GetAttribute("physics:upperLimit").Set(upper)
        self._set_drive_gains(0.0)
        self._set_targets(self._lock_positions)
        if self._use_direct_effort:
            self._articulation.set_dof_efforts(
                np.zeros((1, len(self._dof_indices)), dtype=np.float32),
                dof_indices=self._dof_indices,
            )
        self._last_efforts.fill(0.0)
        self._command_targets = self._lock_positions.copy()
        self._state = self.LOCKED
        self._settle_since = None
        print(
            "RM75 arm lock: LOCKED at "
            f"max_abs_q={float(np.max(np.abs(self._lock_positions))):.6f} rad"
        )

    def lock_current(self) -> None:
        """Lock the current arm pose after startup synchronization."""
        self._lock_at(self._positions())

    def _begin_unlock(self, requested_targets: np.ndarray) -> None:
        current = self._positions()
        self._restore_limits()
        self._pending_targets = np.asarray(requested_targets, dtype=np.float32).copy()
        self._command_targets = self._pending_targets.copy()
        self._unlock_hold_targets = current.copy()
        self._set_targets(current)
        if self._use_direct_effort:
            self._articulation.set_dof_efforts(
                np.zeros((1, len(self._dof_indices)), dtype=np.float32),
                dof_indices=self._dof_indices,
            )
        self._last_efforts.fill(0.0)
        self._unlock_started = time.perf_counter()
        self._state = self.UNLOCKING
        self._settle_since = None
        print("RM75 arm lock: UNLOCKING, holding current q while gains ramp")

    def update(self, executing: bool) -> None:
        """Advance the lock state machine once per simulation frame."""
        now = time.perf_counter()
        current = self._positions()
        requested = self._targets()
        velocities = self._velocities()
        dt = max(0.0, min(now - self._last_update, 0.1))
        self._last_update = now
        target_changed = self._capture_external_target(requested)

        if self._state == self.LOCKED:
            if executing or target_changed:
                self._begin_unlock(self._command_targets if target_changed else current)
            return

        if self._state == self.UNLOCKING:
            # A planner may publish the first trajectory samples during the
            # ramp. Remember them, but keep the physical target at current q
            # until the gains have reached their normal value.
            if target_changed:
                self._pending_targets = self._command_targets.copy()
                self._set_targets(self._unlock_hold_targets)
            fraction = (now - self._unlock_started) / max(self._unlock_duration, 1e-6)
            self._set_drive_gains(fraction)
            if fraction >= 1.0:
                self._set_drive_gains(1.0)
                self._set_targets(self._pending_targets)
                # Direct effort mode owns the complete command. The default
                # position-drive mode keeps the restored drive active.
                self._set_drive_gains(0.0 if self._use_direct_effort else 1.0)
                self._state = self.MOVING
                self._was_executing = executing
                print("RM75 arm lock: MOVING, normal gains and original limits restored")
            return

        if self._state == self.MOVING:
            if self._use_direct_effort:
                self._set_efforts(self._command_targets, current, velocities, dt)
            elif np.max(np.abs(self._command_targets - self._applied_targets)) > self._target_change_threshold:
                self._set_targets(self._slew_target(self._command_targets, dt))
            if not executing and self._was_executing:
                self._state = self.STOPPING
                self._settle_since = None
                self._pending_targets = self._command_targets.copy()
                print("RM75 arm lock: STOPPING, waiting for q/v settle")
            self._was_executing = executing
            return

        if self._state == self.STOPPING:
            if target_changed or executing:
                self._begin_unlock(self._command_targets if target_changed else current)
                return
            if self._use_direct_effort:
                self._set_efforts(self._pending_targets, current, velocities, dt)
            elif np.max(np.abs(self._pending_targets - self._applied_targets)) > self._target_change_threshold:
                self._set_targets(self._slew_target(self._pending_targets, dt))
            error = float(np.max(np.abs(current - self._pending_targets)))
            speed = float(np.max(np.abs(velocities)))
            if error <= self._settle_error and speed <= self._settle_velocity:
                if self._settle_since is None:
                    self._settle_since = now
                elif now - self._settle_since >= self._settle_duration:
                    self._lock_at(current)
            else:
                self._settle_since = None

    def destroy(self) -> None:
        """Restore original limits before the simulation process exits."""
        self._restore_limits()
