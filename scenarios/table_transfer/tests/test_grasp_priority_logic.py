"""Verify tier locks and that failed connection cannot suppress escalation."""
from pathlib import Path
import json,sys,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning.model import build_config
from planning.interfaces import RobotState
from planning.arm_comparison import compare_arms
ROOT=Path(__file__).resolve().parents[1]

class PriorityTests(unittest.TestCase):
    def test_ik_active_joints_and_measured_locks(self):
        saved=json.loads((ROOT/'tests/artifacts/grasp_first/result.json').read_text())['start_state']
        state=RobotState(saved['joint_names'],np.array(saved['positions']),np.zeros(len(saved['positions'])),0.,np.array(saved['base_pose']))
        settings=json.loads((ROOT/'planning/config.json').read_text())
        source=Path(json.loads((ROOT/'scene_config.json').read_text())['robot']['curobo_config'])
        arm=[f'r_joint{i}' for i in range(1,8)]
        for scope,expected in [('arm_only',arm),('arm_lift',['platform_joint']+arm),('mobile',['base_x','base_y','base_yaw','platform_joint']+arm)]:
            with tempfile.TemporaryDirectory() as directory:
                k=build_config(source,state,directory,dict(settings,grasp_ik_scope=scope))['kinematics']
            self.assertEqual(k['cspace']['joint_names'],expected)
            for name,value in zip(state.joint_names,state.joint_positions):
                if name.startswith('l_joint') or name=='platform_joint' and scope=='arm_only':
                    self.assertEqual(k['lock_joints'][name],value)
            if scope!='mobile':
                self.assertIn('base_x',k['lock_joints'])
                self.assertIn('base_yaw',k['lock_joints'])

    def test_all_heights_before_escalating_and_stop_at_success(self):
        calls=[]
        targets={side:[SimpleNamespace(task_constraints={'candidate_rank':rank}) for rank in range(2)] for side in ('l','r')}
        def attempt(state,world,target,factory,settings,scope,planners=None):
            calls.append(scope)
            # Earlier tier can have a grasp IK but a failed connection: it is incomplete.
            return SimpleNamespace(complete=scope=='arm_lift',search_stats={},approach_reason='connection failed' if scope=='arm_only' else '',pregrasp=object(),approach=object())
        with patch('planning.arm_comparison._plan_grasp_tier',side_effect=attempt),patch('planning.arm_comparison.path_cost',return_value=(1.,{})):
            result,_,selected=compare_arms(None,None,targets,None,{})
        self.assertEqual(calls,['arm_only']*4+['arm_lift']*4)
        self.assertIsNotNone(selected)
        self.assertEqual(len(result[selected].search_stats['priority_trials']),2)

if __name__=='__main__':unittest.main()
