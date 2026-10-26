"""Tests for whole-body state ordering and differential-drive semantics."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planning.differential_drive import integrate, validate_base_motion
from planning.model import ACTIVE_JOINTS
from planning.interfaces import RobotState

class SynchronizedMotionTests(unittest.TestCase):
    def test_both_arms_lift_and_base_are_active(self):
        self.assertEqual(len(ACTIVE_JOINTS), 18)
        self.assertIn('l_joint7', ACTIVE_JOINTS)
        self.assertIn('r_joint7', ACTIVE_JOINTS)
        self.assertIn('platform_joint', ACTIVE_JOINTS)

    def test_joint_mapping_uses_names(self):
        state = RobotState(['r_joint1','platform_joint'], np.array([.2,.5]), np.array([.1,.0]), 0., np.array([0,0,0,1,0,0,0]))
        q, v = state.ordered(['platform_joint','r_joint1'])
        np.testing.assert_allclose(q, [.5,.2])
        np.testing.assert_allclose(v, [0,.1])

    def test_negative_y_is_forward(self):
        poses = integrate(np.zeros(3), np.array([[1.,0.]]), np.array([0.,1.]))
        np.testing.assert_allclose(poses[-1], [0,-1,0], atol=1e-12)

    def test_in_place_turn_and_arc(self):
        poses = integrate(np.zeros(3), np.array([[0.,np.pi/2],[1.,np.pi/2]]), np.array([0.,1.,2.]))
        np.testing.assert_allclose(poses[1], [0,0,np.pi/2], atol=1e-12)
        np.testing.assert_allclose(poses[2], [2/np.pi,2/np.pi,np.pi], atol=1e-12)

    def test_lateral_interpolation_is_rejected(self):
        poses = np.array([[0.,0.,0.],[1.,0.,0.]])
        check = validate_base_motion(poses, np.array([[1.,0.]]), np.array([0.,1.]))
        self.assertFalse(check['nonholonomic_valid'])

    def test_speed_limit_is_checked(self):
        u=np.array([[1.6,0.]])
        t=np.array([0.,1.]);poses=integrate(np.zeros(3),u,t)
        self.assertFalse(validate_base_motion(poses,u,t)['linear_speed_valid'])

if __name__ == '__main__':
    unittest.main()
