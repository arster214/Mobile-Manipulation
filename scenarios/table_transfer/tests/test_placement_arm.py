"""Placement targets must follow the arm selected during grasp search."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning.placement import plan_placement

class PlacementArmTest(unittest.TestCase):
 def test_selected_arm_target(self):
  for side in ('l','r'):
   targets=[]; preferences=[]
   class Solver:
    arm=side
    names=[]
    def grasp_solver(self,scope,prefer_base_motion=True):
     preferences.append(prefer_base_motion);return self
    def ik_candidates(self,target,seed):targets.append(target.tool_frame);return []
   attachment=SimpleNamespace(tcp_from_object=np.eye(4))
   with patch('planning.placement.state_vector',return_value=np.array([])):
    with self.assertRaisesRegex(RuntimeError,'No complete placement'):
     plan_placement(None,None,[0,0,0,1,0,0,0],attachment,lambda *_:Solver(),{'max_attempts':1},.1)
   self.assertEqual(targets,[f'{side}_grasp_tcp']*3)
   self.assertEqual(preferences,[False]*3)
if __name__=='__main__':unittest.main()
