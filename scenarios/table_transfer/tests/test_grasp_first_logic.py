"""Regression checks for phase-local object exclusion and reversing retreat."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning.interfaces import EnvironmentState,TaskTarget,RobotState,PlannedTrajectory,Status
from planning.arm_comparison import contact_environment,reverse_retreat

class GraspFirstTests(unittest.TestCase):
    def test_exclusion_is_local_and_keeps_table(self):
        world=EnvironmentState({'cuboid':{'Cup':{'dims':[.06,.06,.12]},'Table':{'dims':[1.4,.6,.75]}}},0.)
        target=TaskTarget(np.zeros(7),np.zeros(7),0.)
        contact=contact_environment(world,target)
        self.assertIn('Cup',world.obstacles['cuboid'])
        self.assertNotIn('Cup',contact.obstacles['cuboid'])
        self.assertIn('Table',contact.obstacles['cuboid'])
        self.assertEqual(contact.version,world.version)

    def test_reverse_retreat_preserves_locks_and_time(self):
        state=RobotState(['r_joint1'],np.array([.2]),np.zeros(1),0.,np.array([0,0,0,1,0,0,0]))
        retreat=PlannedTrajectory(['r_joint1'],np.array([[.5],[.3],[.2]]),np.array([[-.2],[-.1],[0.]]),
            np.ones((3,1)),np.array([0.,.3,1.]),state,1,1,validation_status=Status.PASSED,
            base_poses=np.tile([.5,.7,1.],(3,1)),base_controls=np.zeros((2,2)))
        approach=reverse_retreat(retreat,state)
        np.testing.assert_allclose(approach.positions[:,0],[.2,.3,.5])
        np.testing.assert_allclose(approach.times,[0,.7,1])
        np.testing.assert_allclose(approach.velocities[:,0],[0,.1,.2])
        np.testing.assert_allclose(approach.base_controls,0)
        np.testing.assert_allclose(approach.base_poses,retreat.base_poses)
        self.assertEqual(approach.validation_status,Status.NOT_RUN)
        self.assertEqual(retreat.validation_status,Status.PASSED)

if __name__=='__main__':unittest.main()
