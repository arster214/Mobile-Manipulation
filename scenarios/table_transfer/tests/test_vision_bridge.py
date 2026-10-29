import unittest,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from planning.vision_bridge import world_observation,BoxPerception,rest_box_on_table
from planning.perception import ObjectObservation

class BridgeTests(unittest.TestCase):
    def test_calibrated_pose_composition(self):
        camera=np.eye(4);camera[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix();camera[:3,3]=[1,2,3]
        obj=np.eye(4);obj[:3,3]=[.1,.2,.3]
        rec=dict(world_extrinsics_valid=True,T_object_to_camera=obj,units='metres',camera_frame='optical x-right y-down z-forward',timestamp=2.,confidence=.8)
        obs=world_observation(rec,camera,3)
        np.testing.assert_allclose(obs.pose[:3],[.8,2.1,3.3]);self.assertEqual(obs.version,3)
        self.assertEqual(obs.timestamp,2.);self.assertEqual(obs.confidence,.8)
    def test_cube_symmetry_keeps_task_direction(self):
        policy=json.loads((ROOT/'grasp_policy.json').read_text())
        table=json.loads((ROOT/'scene_config.json').read_text())['tables'][0]
        model=BoxPerception({'cuboid':{}},[.04]*3)
        for rot in Rotation.create_group('O'):
            obs=ObjectObservation(np.r_[[-1.05,1.32,.77],rot.as_quat(scalar_first=True)],0.,1)
            canonical=rest_box_on_table(obs,[.04]*3,.75)
            self.assertLess(Rotation.from_quat(canonical.pose[3:],scalar_first=True).magnitude(),1e-8)
            target=model.targets(obs,table,policy)[0]
            np.testing.assert_allclose(target.task_constraints['approach_direction'],[0,1,0],atol=1e-8)
            np.testing.assert_allclose(target.target_pose[:3],[-1.05,1.17,.77])
            box=model.environment(obs).obstacles['cuboid']['ManipulatedCube']
            self.assertEqual(box['dims'],[.04]*3)
    def test_pitched_pregrasp_is_above_grasp(self):
        policy=json.loads((ROOT/'grasp_policy.json').read_text());policy['box_approach_pitch_deg']=[30]
        table=json.loads((ROOT/'scene_config.json').read_text())['tables'][0]
        obs=ObjectObservation(np.array([-1.05,1.32,.77,1,0,0,0]),0.,1)
        target=BoxPerception({'cuboid':{}},[.04]*3).targets(obs,table,policy)[0]
        self.assertAlmostEqual(target.target_pose[2],.845)
        self.assertLess(target.target_pose[1],obs.pose[1])
    def test_known_support_preserves_xy_and_removes_penetration(self):
        rotation=Rotation.from_euler('xyz',[.01,-.02,.3])
        obs=ObjectObservation(np.r_[[1.2,1.3,.769],rotation.as_quat(scalar_first=True)],0.,1)
        result=rest_box_on_table(obs,[.04]*3,.75)
        np.testing.assert_allclose(result.pose[:3],[1.2,1.3,.77])
        matrix=Rotation.from_quat(result.pose[3:],scalar_first=True).as_matrix()
        self.assertAlmostEqual(result.pose[2]-abs(matrix[2])@np.array([.02]*3),.75)
        np.testing.assert_allclose(obs.pose[:3],[1.2,1.3,.769])

    def test_missing_world_calibration_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'calibration'):
            world_observation({'world_extrinsics_valid':False},np.eye(4),1)

if __name__=='__main__':unittest.main()
