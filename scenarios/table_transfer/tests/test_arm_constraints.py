"""Regression tests for signed sectors, singularity gates and reverse routing."""
import sys, unittest,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning.selection import side_allowed,singularity_metrics,path_cost
from planning.trajectory import candidate
from planning.interfaces import RobotState
from planning.model import ACTIVE_JOINTS,UPPER_BODY_JOINTS
ROOT=Path(__file__).resolve().parents[1]

class ArmConstraints(unittest.TestCase):
    def setUp(self):
        self.settings=json.loads((ROOT/'planning/config.json').read_text())

    def test_signed_side_sectors(self):
        # URDF yaw pi/2 points along world +X; world +Y is left.
        base=np.array([0,0,np.pi/2])
        for angle,left,right in [(10,True,False),(-10,False,True),(21,False,False),(-21,False,False),(0,True,True)]:
            object_pose=np.r_[np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)),0]
            self.assertEqual(side_allowed(base,object_pose,'l',self.settings),left)
            self.assertEqual(side_allowed(base,object_pose,'r',self.settings),right)

    def test_singular_value_and_condition_are_independent(self):
        matrices=np.zeros((3,6,7))
        matrices[0,:6,:6]=np.diag([1,1,1,1,1,.02])
        matrices[1,:6,:6]=np.diag([1,1,1,1,1,.006])
        matrices[2,:6,:6]=np.diag([.1,.1,.1,.1,.1,.004])
        minimum,condition=singularity_metrics(matrices)
        np.testing.assert_allclose(minimum,[.02,.006,.004])
        self.assertLess(condition[0],100)
        self.assertGreater(condition[1],100)
        self.assertLess(condition[2],100)

    def test_reverse_avoids_turning_to_move_backward(self):
        state=RobotState(UPPER_BODY_JOINTS,np.zeros(15),np.zeros(15),0,np.array([0,0,0,1,0,0,0]))
        goal=np.zeros(18);goal[1]=.1
        limits={key:np.ones(15) for key in ('velocity','acceleration','jerk')}
        forward=candidate(state,goal,limits,self.settings,1,1)
        reverse=candidate(state,goal,limits,self.settings,1,1,reverse=True)
        self.assertLess(path_cost(reverse,'r',self.settings)[0],path_cost(forward,'r',self.settings)[0])
        np.testing.assert_allclose(reverse.base_poses[-1],goal[:3],atol=1e-8)
        self.assertEqual(path_cost(reverse,'r',self.settings)[1]['base_rotation_rad'],0)

if __name__=='__main__':unittest.main()
