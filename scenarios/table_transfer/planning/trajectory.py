"""Synchronized differential-drive and upper-body trajectory candidates.

These are sampling-planner candidates, never execution-ready before full-body
validation. Base motion is parameterized by forward speed and yaw rate.
"""
from __future__ import annotations
import numpy as np
from .differential_drive import integrate
from .interfaces import PlannedTrajectory, RobotState
from .model import UPPER_BODY_JOINTS, state_vector


def smoothstep(s: np.ndarray) -> np.ndarray:
    return s**3 * (10 + s * (-15 + 6*s))


def wrapped(angle: float | np.ndarray) -> float | np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def candidate(start: RobotState, goal: np.ndarray, limits: dict, settings: dict,
              environment_version: int, target_version: int,
              route: str = 'direct', upper_timing: str = 'together', reverse: bool = False,
              backoff_m: tuple[float, float] = (0., 0.)) -> PlannedTrajectory:
    """Generate stop-to-stop synchronized motion, with three route seeds."""
    q0 = state_vector(start, ['base_x', 'base_y', 'base_yaw'] + UPPER_BODY_JOINTS)
    anchors = [q0[:2]]
    directions = []
    def waypoint(xy, backwards):
        anchors.append(np.asarray(xy))
        directions.append(backwards)
    before, after = backoff_m
    if before:
        forward = np.array([np.sin(q0[2]), -np.cos(q0[2])])
        waypoint(q0[:2] - before * forward, True)
    begin = anchors[-1]
    forward_goal = np.array([np.sin(goal[2]), -np.cos(goal[2])])
    finish = goal[:2] + after * forward_goal
    if route == 'x_then_y':
        waypoint([finish[0], begin[1]], reverse)
    elif route == 'y_then_x':
        waypoint([begin[0], finish[1]], reverse)
    waypoint(finish, reverse)
    if after:
        waypoint(goal[:2], True)
    dt = settings['interpolation_dt_s']
    yaw = q0[2]
    controls = []
    def segment(amount: float, turning: bool) -> None:
        if abs(amount) < 1e-9:
            return
        speed = settings['simulation_base_yaw_rate_rad_s'] if turning else settings['base_max_linear_velocity_m_s']
        accel = settings['simulation_base_angular_acceleration_rad_s2'] if turning else settings['simulation_base_acceleration_m_s2']
        duration = max(1.875*abs(amount)/speed, np.sqrt(5.774*abs(amount)/accel), dt)
        count = max(2, int(np.ceil(duration/dt)))
        velocity = np.diff(smoothstep(np.linspace(0, 1, count+1))) * amount / dt
        controls.extend([[0., v] if turning else [v, 0.] for v in velocity])
    for a, b, backwards in zip(anchors[:-1], anchors[1:], directions):
        direction = b-a
        if np.linalg.norm(direction) < 1e-9:
            continue
        desired = np.arctan2(direction[1], direction[0]) + np.pi/2
        if backwards:
            desired += np.pi
        delta = wrapped(desired-yaw)
        segment(delta, True)
        yaw += delta
        segment(float(np.linalg.norm(direction)) * (-1 if backwards else 1), False)
    segment(wrapped(goal[2]-yaw), True)
    if not controls:
        controls = [[0., 0.], [0., 0.]]
    controls = np.array(controls)
    times = np.arange(len(controls)+1)*dt
    # Stretch the entire common clock to honor body velocity/acceleration/jerk limits.
    difference = np.abs(goal[3:]-q0[3:])
    minimum = max(float(np.max(1.875*difference/limits['velocity'])),
                  float(np.max(np.sqrt(5.774*difference/limits['acceleration']))),
                  float(np.max(np.cbrt(60*difference/limits['jerk']))))
    timing_fraction = .75 if upper_timing != 'together' else 1.
    scale = max(1., minimum/(times[-1]*timing_fraction))
    times *= scale
    controls /= scale
    subdivisions = max(1, int(np.ceil(scale)))
    if subdivisions > 1:
        controls = np.repeat(controls, subdivisions, axis=0)
        times = np.linspace(0., times[-1], len(controls)+1)
    base = integrate(q0[:3], controls, times)
    s = times/times[-1]
    if upper_timing == 'early':
        s = np.clip(s/.75, 0, 1)
    elif upper_timing == 'late':
        s = np.clip((s-.25)/.75, 0, 1)
    positions = q0[3:] + smoothstep(s)[:,None]*(goal[3:]-q0[3:])
    velocities = np.gradient(positions, times, axis=0, edge_order=2)
    accelerations = np.gradient(velocities, times, axis=0, edge_order=2)
    return PlannedTrajectory(UPPER_BODY_JOINTS.copy(), positions, velocities, accelerations,
                             times, start, environment_version, target_version,
                             base_poses=base, base_controls=controls)
