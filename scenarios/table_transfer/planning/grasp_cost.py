"""Add measured-base preference to the mobile grasp IK optimization objective."""
from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg


class GraspCostManager(RobotCostManager):
    def compute_costs(self, state, cost_collection=None, goal=None, **kwargs):
        costs = super().compute_costs(state, cost_collection, goal, **kwargs)
        # cuRobo V2's stock manager reports start distance as a convergence
        # metric only; it does not add it to the optimization objective.
        if self.has_cost('start_cspace_dist') and goal is not None and goal.current_js is not None:
            cost = self.get_cost('start_cspace_dist')
            if cost.enabled:
                value = cost.forward(state.joint_state.position, goal.current_js.position,
                                     goal.idxs_current_js.view(-1))
                costs.add(value.unsqueeze(-1), 'grasp_start_distance')
        return costs


class GraspCostManagerCfg(RobotCostManagerCfg):
    @staticmethod
    def create(*args, **kwargs):
        base = RobotCostManagerCfg.create(*args, **kwargs)
        config = GraspCostManagerCfg(**vars(base))
        config.class_type = GraspCostManager
        return config
