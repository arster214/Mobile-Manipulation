"""Short reverse adjustments stay feasible; long carry reversals cost more."""
from pathlib import Path
import sys,json,unittest
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning.selection import path_cost
ROOT=Path(__file__).resolve().parents[1]
class CarryPenaltyTests(unittest.TestCase):
 def score(self,phase,velocity,duration):
  settings=json.loads((ROOT/'planning/config.json').read_text())
  plan=SimpleNamespace(positions=np.zeros((2,7)),joint_names=[f'r_joint{i}' for i in range(1,8)],base_controls=np.array([[velocity,0.]]),times=np.array([0.,duration]),motion_phase=phase)
  return path_cost(plan,'r',settings)
 def test_approach_has_no_added_penalty(self):
  self.assertEqual(self.score('approach',-.2,2),self.score('approach',.2,2))
 def test_short_reverse_remains_finite(self):
  reverse,_=self.score('carry',-.2,1);forward,_=self.score('carry',.2,1)
  self.assertAlmostEqual(reverse-forward,.4)
 def test_long_reverse_is_quadratically_penalized(self):
  reverse,terms=self.score('carry',-.2,4);forward,_=self.score('carry',.2,4)
  self.assertAlmostEqual(reverse-forward,5.805)
  self.assertAlmostEqual(terms['carry_reverse_excess_time_squared_s2'],8.41)
 def test_forward_has_zero_reverse_terms(self):
  _,terms=self.score('carry',.2,4)
  self.assertEqual(terms['carry_reverse_distance_m'],0.)
  self.assertEqual(terms['carry_reverse_excess_time_squared_s2'],0.)
 def test_separate_short_adjustments_do_not_accumulate_time_penalty(self):
  settings=json.loads((ROOT/'planning/config.json').read_text())
  plan=SimpleNamespace(positions=np.zeros((4,7)),joint_names=[f'r_joint{i}' for i in range(1,8)],base_controls=np.array([[-.2,0.],[0.,.5],[-.2,0.]]),times=np.array([0.,1.,2.,3.]),motion_phase='carry')
  _,terms=path_cost(plan,'r',settings)
  self.assertAlmostEqual(terms['carry_reverse_distance_m'],.4)
  self.assertEqual(terms['carry_reverse_excess_time_squared_s2'],0.)
 def test_two_short_adjustments_preserve_endpoint_and_distance(self):
  from planning.interfaces import RobotState
  from planning.model import UPPER_BODY_JOINTS,state_vector
  from planning.trajectory import candidate,wrapped
  from planning.differential_drive import validate_base_motion
  settings=json.loads((ROOT/'planning/config.json').read_text())
  state=RobotState(UPPER_BODY_JOINTS,np.zeros(len(UPPER_BODY_JOINTS)),np.zeros(len(UPPER_BODY_JOINTS)),0.,np.array([0.,0.,0.,1.,0.,0.,0.]))
  goal=state_vector(state,['base_x','base_y','base_yaw']+UPPER_BODY_JOINTS)
  goal[:3]=[1.,.5,.8]
  limits={k:np.ones(len(UPPER_BODY_JOINTS)) for k in ['velocity','acceleration','jerk']}
  plan=candidate(state,goal,limits,settings,1,1,backoff_m=(.1,.2))
  np.testing.assert_allclose(plan.base_poses[-1,:2],goal[:2],atol=1e-8)
  self.assertAlmostEqual(float(wrapped(plan.base_poses[-1,2]-goal[2])),0.)
  distance=float(np.sum(np.maximum(-plan.base_controls[:,0],0)*np.diff(plan.times)))
  self.assertAlmostEqual(distance,.3)
  self.assertTrue(validate_base_motion(plan.base_poses,plan.base_controls,plan.times)['nonholonomic_valid'])
if __name__=='__main__':unittest.main()
