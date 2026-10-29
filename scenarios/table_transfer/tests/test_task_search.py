"""Failure recovery: reject an unusable suffix and continue the same IK tier."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planning.arm_comparison import _plan_grasp_tier
from planning.interfaces import Status
from planning.placement import plan_placement
from planning.vision_bridge import approach_angles


class SearchTests(unittest.TestCase):
    def test_placement_failure_retries_grasp_branch_before_escalating(self):
        target = NS(tool_frame='r_grasp_tcp', target_type='pregrasp', target_pose=np.zeros(7),
                    task_constraints={'grasp_pose':np.zeros(7)}, object_pose=np.zeros(7), version=1)
        def replaced(obj, **changes):
            return NS(**(vars(obj) | changes))
        plan = NS(validation_status=Status.PASSED, positions=np.zeros((2,1)), joint_names=['r_joint1'])
        planner = NS(names=['r_joint1'], rng=None,
                     ik_candidates=lambda *_: [np.array([.1]), np.array([.2])],
                     posture_metrics=lambda _: {'passed':True}, tcp_pose=lambda _: np.zeros((1,7)),
                     plan_approach=lambda *_: plan, plan_mobile=lambda *a, **k: plan,
                     validate=lambda _: True)
        planner.grasp_solver = lambda _: planner
        seen = []
        def suffix(item):
            seen.append(item)
            if len(seen)==1:
                raise RuntimeError('no feasible placement')
            return 'full solution'
        with patch.multiple('planning.arm_comparison', replace=replaced,
                            contact_environment=lambda *a:None, state_vector=lambda *a:np.zeros(1),
                            state_at=lambda *a:None, side_allowed=lambda *a:True,
                            reverse_retreat=lambda *a:plan, path_cost=lambda *a:(1.,{})):
            result = _plan_grasp_tier(None, None, target, lambda *a:planner,
                                      {'max_attempts':1}, 'arm_only', continuation=suffix)
        self.assertEqual(len(seen), 2)
        self.assertTrue(result.complete)
        self.assertEqual(result.task_suffix, 'full solution')
        self.assertEqual(result.search_stats['task_rejections'], 1)
        self.assertEqual(result.search_stats['motion_scope'], 'arm_only')

    def test_withdrawal_failure_retries_placement(self):
        plan = NS(validation_status=Status.PASSED, positions=np.zeros((2,1)), reason='')
        planner = NS(arm='r', names=['r_joint1'],
                     ik_candidates=lambda *_:[np.array([.1]),np.array([.2])],
                     posture_metrics=lambda _: {'passed':True},
                     tcp_pose=lambda _:np.array([[0,0,1,1,0,0,0]]),
                     plan_approach=lambda *_:plan, plan_mobile=lambda *a, **k:plan, validate=lambda _:True)
        planner.grasp_solver=lambda *a,**k:planner
        calls=[]
        def withdrawal(*args):
            calls.append(args)
            if len(calls)==1:raise RuntimeError('withdraw collision')
        with patch.multiple('planning.placement',state_vector=lambda *a:np.zeros(1),
                            state_at=lambda *a:None,reverse_retreat=lambda *a:plan):
            result=plan_placement(None,None,[0,0,0,1,0,0,0],NS(tcp_from_object=np.eye(4)),
                                  lambda *a:planner,{'max_attempts':1},.1,withdrawal)
        self.assertEqual(len(calls),2)
        self.assertEqual(result[-1]['scope'],'arm_only')
        self.assertEqual(result[-1]['trials'][0]['ik_solutions'],2)

    def test_cartesian_ik_retries_a_failed_waypoint(self):
        from planning.curobo_backend import CuroboPlanner
        from planning.interfaces import TaskTarget
        calls = []
        def ik(*args):
            calls.append(args)
            return [] if len(calls)==1 else [np.array([.1])]
        planner = NS(arm_only=True, arm='r', state=None, names=['r_joint1'],
                     settings={'max_attempts':2, 'approach_samples':3, 'interpolation_dt_s':.025},
                     ik_candidates=ik, posture_mask=lambda qs: np.ones(len(qs), dtype=bool),
                     limits={'velocity':np.ones(1), 'acceleration':np.ones(1)},
                     environment=NS(version=1), validate=lambda plan:True)
        target = TaskTarget(np.array([0,0,1,1,0,0,0.]), np.zeros(7), 0.)
        with patch('planning.curobo_backend.state_vector', side_effect=lambda state,names:np.zeros(len(names))):
            plan = CuroboPlanner.plan_approach(planner, target, target.target_pose.copy())
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(plan.positions), 3)

    def test_more_samples_preserve_existing_coverage(self):
        small=approach_angles(9);large=approach_angles(17)
        self.assertEqual(small,large[:9])
        self.assertEqual(len(set(large)),17)
        self.assertTrue(all(0<=angle<90 for angle in large))

if __name__=='__main__':unittest.main()
