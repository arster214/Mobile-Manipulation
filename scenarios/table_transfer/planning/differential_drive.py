"""Differential-drive integration; URDF forward is local -Y.

Base controls and upper-body joint samples share one time axis. These helpers
validate base kinematics only; full-body collision/limits remain planner checks.
"""
from __future__ import annotations
import numpy as np


def integrate(base_start: np.ndarray, controls: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Integrate piecewise constant (forward velocity, yaw rate) exactly."""
    dt = np.diff(times)
    if controls.shape != (len(dt), 2) or np.any(dt <= 0):
        raise ValueError('Controls must match a strictly increasing time axis')
    poses = np.empty((len(times), 3))
    poses[0] = base_start
    for i, ((v, omega), step) in enumerate(zip(controls, dt)):
        yaw_delta = omega * step
        yaw_mid = poses[i, 2] + yaw_delta / 2
        length = v * step * np.sinc(yaw_delta / (2*np.pi))
        poses[i+1] = poses[i] + np.array([length*np.sin(yaw_mid), -length*np.cos(yaw_mid), yaw_delta])
    return poses


def validate_base_motion(poses: np.ndarray, controls: np.ndarray, times: np.ndarray,
                         max_linear_speed: float = 1.5) -> dict:
    """Check sampled base motion against its controls, including no side slip."""
    if poses.shape != (len(times), 3):
        raise ValueError('Base pose samples and time axis must have matching lengths')
    if not (np.isfinite(poses).all() and np.isfinite(controls).all() and np.isfinite(times).all()):
        raise ValueError('Base trajectory must be finite')
    predicted = integrate(poses[0], controls, times)
    error = np.max(np.abs(predicted-poses), axis=0)
    return {
        'nonholonomic_valid': bool(np.all(error <= 1e-6)),
        'linear_speed_valid': bool(np.all(np.abs(controls[:, 0]) <= max_linear_speed)),
        'max_pose_error': error.tolist(),
        'angular_and_acceleration_limits_checked': False,
    }
