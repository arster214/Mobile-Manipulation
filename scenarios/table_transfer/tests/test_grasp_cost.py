"""Verify the added GPU objective penalizes base motion without pinning the arm."""
from pathlib import Path
from types import SimpleNamespace
import sys,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning.grasp_cost import GraspCostManager
from curobo._src.cost.cost_cspace_dist import CSpaceDistCost
from curobo._src.cost.cost_cspace_dist_cfg import CSpaceDistCostCfg

class CostTests(unittest.TestCase):
    def test_base_only_gradient_and_rollout_shape(self):
        cfg=CSpaceDistCostCfg(weight=1.)
        cfg.update_dof(4)
        cfg.update_terminal_dof_weight([10.,10.,10.,0.])
        manager=GraspCostManager();manager.register_cost('start_cspace_dist',CSpaceDistCost(cfg))
        q=torch.tensor([[[.1,.1,.1,.6]]],device='cuda',requires_grad=True)
        state=SimpleNamespace(joint_state=q)
        # RobotState's joint_state exposes both shape and position.
        state.joint_state=SimpleNamespace(shape=q.shape,position=q)
        goal=SimpleNamespace(current_js=SimpleNamespace(position=torch.zeros((1,4),device='cuda')),
                             idxs_current_js=torch.zeros(1,device='cuda',dtype=torch.int32))
        costs=manager.compute_costs(state,goal=goal)
        value=costs.values[0]
        self.assertEqual(tuple(value.shape),(1,1,1))
        value.sum().backward()
        self.assertTrue(torch.all(q.grad[0,0,:3]>0).item())
        self.assertEqual(q.grad[0,0,3].item(),0.)
if __name__=='__main__':unittest.main()
