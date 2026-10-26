"""Module C: cuRobo IK/collision kernels and synchronized candidate planning.

The mobile path search is a small sampling baseline (three route seeds), not a
claim that cuRobo's built-in joint spline enforces differential-drive dynamics.
"""
from __future__ import annotations
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.types import JointState, GoalToolPose, Pose
from .interfaces import RobotState, TaskTarget, EnvironmentState, PlannedTrajectory, Status
from .model import ACTIVE_JOINTS, UPPER_BODY_JOINTS, build_config, state_vector
from .trajectory import candidate, wrapped
from .differential_drive import validate_base_motion
from .selection import side_allowed, singularity_metrics, path_cost


class CuroboPlanner:
    """A planner instance is bound to one measured lock state and environment."""

    def __init__(self, robot_source: Path, state: RobotState, environment: EnvironmentState,
                 settings: dict, runtime_directory: Path, arm_only: bool = False, arm: str = 'r', reference_state=None):
        self.robot_source = robot_source
        self.runtime_directory = runtime_directory
        self.extra_validation = None
        self.search_stats = {}
        self.arm = arm
        self.reference_state = state if reference_state is None else reference_state
        self.settings = settings
        self.arm_only = arm_only
        self.environment = environment
        self.state = state
        self.robot = build_config(robot_source, state, runtime_directory, settings, arm_only, arm, self.reference_state)
        self.names = self.robot['kinematics']['cspace']['joint_names']
        config = InverseKinematicsCfg.create(
            robot=self.robot, scene_model=environment.obstacles, self_collision_check=True,
            num_seeds=settings['num_ik_seeds'], seed_solver_num_seeds=settings['num_ik_seeds'], max_batch_size=1, position_tolerance=settings['position_tolerance_m'],
            orientation_tolerance=settings['orientation_tolerance_rad'], use_cuda_graph=settings.get('use_cuda_graph', True),
            optimizer_collision_activation_distance=settings['collision_activation_distance_m'],
        )
        self.ik = InverseKinematics(config)
        self.checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
            robot_config=self.robot, scene_model=environment.obstacles,
            collision_activation_distance=0., self_collision_activation_distance=0.,
        ))
        self.checker.kinematics.compute_jacobian = True
        internal_names = self.checker.kinematics.config.kinematics_config.joint_names
        self.to_internal = [self.names.index(name) for name in internal_names]
        self.from_internal = [internal_names.index(name) for name in self.names]
        urdf = ET.parse(self.robot['kinematics']['urdf_path']).getroot()
        cs = self.robot['kinematics']['cspace']
        self.rng = np.random.default_rng(0)
        self.bounds = np.array([[float(urdf.find(f"joint[@name='{n}']/limit").get(key))
                                 for n in self.names] for key in ('lower', 'upper')])
        self.limits = {
            'velocity': np.array([float(urdf.find(f"joint[@name='{n}']/limit").get('velocity')) for n in self.names]),
            'acceleration': np.array(cs['max_acceleration']),
            'jerk': np.array(cs['max_jerk']),
        }

    def grasp_solver(self, scope):
        """Use measured locks for the IK tier; connection validation stays full-body."""
        settings = dict(self.settings, grasp_ik_scope=scope)
        return CuroboPlanner(self.robot_source, self.state, self.environment, settings,
                             self.runtime_directory / scope, arm=self.arm,
                             reference_state=self.reference_state)

    @staticmethod
    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(values), device='cuda', dtype=torch.float32)

    def ik_candidates(self, target: TaskTarget, seed: np.ndarray) -> list[np.ndarray]:
        goal = GoalToolPose.from_poses({target.tool_frame: Pose(
            position=self.tensor(target.target_pose[:3][None]),
            quaternion=self.tensor(target.target_pose[3:][None]),
        )}, num_goalset=1)
        state = JointState.from_position(self.tensor(seed[None]), joint_names=self.names)
        bank = self.rng.uniform(self.bounds[0], self.bounds[1],
                                (self.settings['num_ik_seeds'], len(self.names)))
        # Local seeds preserve posture; hard URDF bounds enforce the 0.8-rad envelope.
        bank = np.clip(seed + self.rng.normal(0., .2, bank.shape), self.bounds[0], self.bounds[1])
        if 'base_x' in self.names:
            # Seed robot placements on the requested side of the object, near the task direction.
            bank[::2, 3:] = self.rng.uniform(self.bounds[0,3:], self.bounds[1,3:], bank[::2,3:].shape)
            other = 'l' if self.arm == 'r' else 'r'
            for i,name in enumerate(self.names):
                if name.startswith(f'{other}_joint'):
                    bank[:,i] = seed[i]
            direction = np.array(target.task_constraints['approach_direction'][:2])
            forward = np.arctan2(direction[1], direction[0])
            width = np.deg2rad(self.settings['grasp_side_half_angle_deg'])
            yaw = forward+np.pi/2+self.rng.uniform(-width,width,len(bank))
            side_angle = self.rng.uniform(0,width,len(bank))*(1 if self.arm=='l' else -1)
            bearing = yaw-np.pi/2+side_angle
            distance = self.rng.uniform(.4,.9,len(bank))
            bank[:,0] = target.object_pose[0]-distance*np.cos(bearing)
            bank[:,1] = target.object_pose[1]-distance*np.sin(bearing)
            bank[:,2] = wrapped(yaw)
            bank = np.clip(bank,self.bounds[0],self.bounds[1])
        bank[0] = seed
        seeds = self.tensor(bank[:, self.to_internal][None])
        result = self.ik.solve_pose(goal, current_state=state, seed_config=seeds,
                                    return_seeds=self.settings['num_ik_seeds'])
        indices = [result.js_solution.joint_names.index(name) for name in self.names]
        solutions = result.js_solution.position[..., indices].detach().cpu().numpy().reshape(-1, len(self.names))
        success = result.success.detach().cpu().numpy().reshape(-1)
        return [q for q, valid in zip(solutions, success) if valid]

    def collision_free(self, positions: np.ndarray) -> bool:
        q = self.tensor(positions[:, self.to_internal])[None]
        return bool(self.checker.validate(q).all().item())

    def posture_metrics(self, positions: np.ndarray) -> dict:
        state = self.checker.get_kinematics(self.tensor(positions[:, self.to_internal])[None])
        internal = self.checker.kinematics.config.kinematics_config.joint_names
        indices = [internal.index(f'{self.arm}_joint{i}') for i in range(1,8)]
        jacobian = state.tool_jacobians[0,:,0].detach().cpu().numpy()[..., indices]
        minimum, condition = singularity_metrics(jacobian)
        arm_indices = [i for i,n in enumerate(self.names) if n.startswith(('l_joint','r_joint'))]
        reference = state_vector(self.reference_state, [self.names[i] for i in arm_indices])
        delta = np.abs(positions[:,arm_indices]-reference).max()
        return {'min_singular_value':float(minimum.min()), 'max_condition_number':float(condition.max()),
                'max_arm_joint_change_rad':float(delta),
                'passed':bool(minimum.min() >= self.settings['jacobian_min_singular_value']
                              and condition.max() <= self.settings['jacobian_max_condition_number']
                              and delta <= self.settings['arm_joint_max_change_rad']+1e-7)}

    def tcp_pose(self, positions):
        pose = self.checker.get_kinematics(self.tensor(positions[:, self.to_internal])[None]).tool_poses
        return np.concatenate([pose.position[0,:,0].cpu().numpy(),pose.quaternion[0,:,0].cpu().numpy()],axis=1)

    def validate(self, plan: PlannedTrajectory) -> bool:
        """Check all command samples and midpoint samples, including kinematics."""
        settings = self.settings
        base = validate_base_motion(plan.base_poses, plan.base_controls, plan.times,
                                    settings['base_max_linear_velocity_m_s'])
        base_ok = base['nonholonomic_valid'] and base['linear_speed_valid']
        plan.base_validation_status = Status.PASSED if base_ok else Status.FAILED
        if self.arm_only:
            q = plan.positions
            limits = self.limits
        else:
            q = np.column_stack([plan.base_poses[:,:2], wrapped(plan.base_poses[:,2]), plan.positions])
            limits = {key: value[3:] for key, value in self.limits.items()}
        # Midpoint base poses follow differential-drive integration, not XY interpolation.
        from .differential_drive import integrate
        mid_base = np.array([integrate(plan.base_poses[i], plan.base_controls[i:i+1],
                            np.array([0., (plan.times[i+1]-plan.times[i])/2]))[-1]
                            for i in range(len(plan.times)-1)])
        mid_body = (plan.positions[:-1]+plan.positions[1:])/2
        middle = mid_body if self.arm_only else np.column_stack([mid_base[:,:2], wrapped(mid_base[:,2]), mid_body])
        motion_ok = (np.all(np.abs(plan.velocities) <= limits['velocity']+1e-6)
                     and np.all(np.abs(plan.accelerations) <= limits['acceleration']+1e-6))
        collision_ok = self.collision_free(np.vstack([q, middle]))
        posture = self.posture_metrics(np.vstack([q, middle]))
        endpoint_posture = self.posture_metrics(q[-1:])
        posture_ok = posture['passed'] if self.arm_only else (endpoint_posture['passed'] and posture['max_arm_joint_change_rad'] <= self.settings['arm_joint_max_change_rad']+1e-7)
        plan.constraint_metrics = {'path': posture, 'endpoint': endpoint_posture,
            'singularity_scope': 'all approach samples' if self.arm_only else 'IK endpoint; moving path reported separately'}
        extra = self.extra_validation(np.vstack([q,middle])) if self.extra_validation is not None else {'passed':True}
        plan.object_metrics = extra
        valid = bool(base_ok and motion_ok and collision_ok and posture_ok and extra['passed'])
        plan.validation_status = Status.PASSED if valid else Status.FAILED
        plan.reason = '' if valid else f'base={base_ok}, joint_motion={motion_ok}, collisions={collision_ok}, posture={posture}, carried_object={extra}'
        return valid

    def plan_mobile(self, target: TaskTarget, goal_configs=None) -> PlannedTrajectory:
        if self.arm_only:
            raise ValueError('Mobile planning needs the full 18-DoF model')
        if not target.valid or not self.environment.valid:
            raise ValueError('Planning needs a valid target and environment')
        if np.any(np.abs(self.state.base_twist) > 1e-6) or np.any(np.abs(self.state.joint_velocities) > 1e-6):
            raise ValueError('This stop-to-stop baseline requires a measured stop before replanning')
        started = time.perf_counter()
        q0 = state_vector(self.state, self.names)
        if not self.collision_free(q0[None]):
            failed = candidate(self.state, q0, {k:v[3:] for k,v in self.limits.items()},
                               self.settings, self.environment.version, target.version)
            failed.planning_status = failed.validation_status = Status.FAILED
            failed.reason = 'Measured starting state violates collision or joint limits'
            failed.planning_time = time.perf_counter()-started
            return failed
        last = None
        found_ik = False
        options = []
        self.search_stats = {'ik_success':0, 'side_pass':0, 'posture_pass':0, 'path_candidates':0}
        batches = [goal_configs] if goal_configs is not None else (
            self.ik_candidates(target, q0) for _ in range(self.settings['max_attempts']))
        for solutions in batches:
            for q in solutions:
                self.search_stats['ik_success'] += 1
                if not side_allowed(q[:3], target.object_pose, self.arm, self.settings):
                    continue
                self.search_stats['side_pass'] += 1
                if not self.posture_metrics(q[None])['passed']:
                    continue
                self.search_stats['posture_pass'] += 1
                found_ik = True
                for route in ('direct', 'x_then_y', 'y_then_x'):
                    for timing in ('together', 'early', 'late'):
                        for reverse in (False, True):
                            limits = {key: value[3:] for key, value in self.limits.items()}
                            plan = candidate(self.state, q, limits, self.settings, self.environment.version,
                                             target.version, route, timing, reverse)
                            plan.ik_status = Status.PASSED
                            plan.planning_status = Status.PASSED
                            score, terms = path_cost(plan, self.arm, self.settings)
                            options.append((score, terms, plan))
        # Cost ordering covers all searched IK solutions/routes, not first-generated feasibility.
        self.search_stats['path_candidates'] = len(options)
        options.sort(key=lambda item:item[0])
        for score, terms, plan in options:
            if self.validate(plan):
                plan.planning_time = time.perf_counter()-started
                plan.cost = score
                plan.cost_terms = terms
                return plan
            last = plan
        if last is None:
            last = candidate(self.state, q0, {key:value[3:] for key,value in self.limits.items()},
                             self.settings, self.environment.version, target.version)
            last.ik_status = Status.PASSED if found_ik else Status.FAILED
        last.planning_status = Status.FAILED
        last.validation_status = Status.FAILED
        last.reason = last.reason or 'No IK/path satisfying arm-side, joint-change and singularity constraints'
        last.planning_time = time.perf_counter()-started
        return last

    def plan_approach(self, target: TaskTarget, tcp_start_pose: np.ndarray) -> PlannedTrajectory:
        """Straight Cartesian approach; base, lift and inactive arm stay measured-locked."""
        from dataclasses import replace
        if not self.arm_only:
            raise ValueError('Approach must use a fresh selected-arm-only model')
        q0 = state_vector(self.state, self.names)
        qs = [q0]
        for fraction in np.linspace(0., 1., self.settings['approach_samples'])[1:]:
            pose = tcp_start_pose.copy()
            pose[:3] += fraction*(target.target_pose[:3]-tcp_start_pose[:3])
            choices = [q for q in self.ik_candidates(replace(target, target_pose=pose), qs[-1])
                       if self.posture_metrics(q[None])['passed']]
            if not choices:
                error = RuntimeError(f'{self.arm}-arm {target.target_type} IK failed at {fraction:.1%}; reject this grasp candidate')
                error.partial_positions = np.array(qs)
                error.failed_target_pose = pose.copy()
                error.failed_fraction = float(fraction)
                raise error
            qs.append(min(choices, key=lambda q: np.linalg.norm(q-qs[-1])))
        positions = np.array(qs)
        # Uniform clock is stretched to respect joint speed/acceleration limits.
        dt = self.settings['interpolation_dt_s']
        velocity = np.gradient(positions, dt, axis=0)
        acceleration = np.gradient(velocity, dt, axis=0)
        scale = max(1., float(np.max(np.abs(velocity)/self.limits['velocity'])),
                    float(np.sqrt(np.max(np.abs(acceleration)/self.limits['acceleration']))))
        times = np.arange(len(qs))*dt*scale
        base = np.tile(state_vector(self.state, ['base_x','base_y','base_yaw']), (len(qs), 1))
        plan = PlannedTrajectory(self.names.copy(), positions, velocity/scale, acceleration/scale**2,
                                 times, self.state, self.environment.version, target.version,
                                 ik_status=Status.PASSED, planning_status=Status.PASSED,
                                 base_poses=base, base_controls=np.zeros((len(qs)-1,2)))
        self.validate(plan)
        return plan
