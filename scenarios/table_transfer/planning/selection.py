"""Arm-side feasibility and explicit, dimensionless whole-path scoring."""
import numpy as np
from .trajectory import wrapped


def heading_error(base_pose, object_pose):
    return wrapped(np.arctan2(object_pose[1]-base_pose[1], object_pose[0]-base_pose[0])
                   -base_pose[2]+np.pi/2)


def side_allowed(base_pose, object_pose, arm, settings):
    angle = heading_error(base_pose, object_pose) * (1 if arm == 'l' else -1)
    return bool(-1e-8 <= angle <= np.deg2rad(settings['grasp_side_half_angle_deg'])+1e-8)


def singularity_metrics(jacobians):
    # Geometric TCP Jacobian: linear m/rad + angular rad/rad, no arbitrary scaling.
    values = np.linalg.svd(jacobians, compute_uv=False)
    minimum = values[..., -1]
    condition = np.divide(values[..., 0], minimum, out=np.full_like(minimum, np.inf), where=minimum>0)
    return minimum, condition


def path_cost(plan, arm, settings):
    travel = np.abs(np.diff(plan.positions, axis=0)).sum(axis=0)
    by_name = dict(zip(plan.joint_names, travel))
    limit = settings['arm_joint_max_change_rad']
    def arm_cost(side):
        return float(np.sqrt(np.mean([(by_name.get(f'{side}_joint{i}', 0.)/limit)**2 for i in range(1,8)])))
    dt = np.diff(plan.times)
    terms = {
        'base_distance_m': float(np.sum(np.abs(plan.base_controls[:,0])*dt)),
        'base_rotation_rad': float(np.sum(np.abs(plan.base_controls[:,1])*dt)),
        'lift_fraction': float(by_name.get('platform_joint',0.)/settings['lift_max_m']),
        'active_arm_fraction': arm_cost(arm),
        'inactive_arm_fraction': arm_cost('l' if arm=='r' else 'r'),
        'duration_s': float(plan.times[-1]),
    }
    return sum(settings['path_cost_weights'][name]*value for name,value in terms.items()), terms
