"""Offline tests for module boundaries, approach locks and snapshot invalidation."""
import sys,json,unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning.interfaces import RobotState,TaskTarget,EnvironmentState,Status,RunState
from planning.model import ACTIVE_JOINTS,UPPER_BODY_JOINTS,build_config,state_vector
from planning.trajectory import candidate
from planning.execution import TrajectoryExecutor
from planning.replanning import ReplanningMonitor
from planning.perception import compose_pose
from planning.local_control import create_local_controller
ROOT=Path(__file__).resolve().parents[1]

class FakeController:
    def __init__(self,state):self.state=state;self.commands=[];self.held=False
    def read_state(self):return self.state
    def send(self,command):self.commands.append(command)
    def hold(self):self.held=True

class ModuleTests(unittest.TestCase):
    def setUp(self):
        self.settings=json.loads((ROOT/'planning/config.json').read_text())
        scene=json.loads((ROOT/'scene_config.json').read_text())
        self.source=Path(scene['robot']['curobo_config'])
        import yaml
        k=yaml.safe_load(self.source.read_text())['kinematics']
        initial=json.loads(Path(scene['robot']['initial_posture']).read_text())
        values=dict(zip(initial['joint_names'],initial['joint_positions_rad']));values.update(k['lock_joints'])
        self.state=RobotState(list(values),np.array(list(values.values())),np.zeros(len(values)),1.,np.array([-1.3,.5,.2473455,0,0,0,1]))

    def test_measured_approach_locks(self):
        state=self.state
        state.joint_positions[state.joint_names.index('platform_joint')]=.42
        config=build_config(self.source,state,ROOT/'tests/artifacts/model',self.settings,arm_only=True)['kinematics']
        self.assertEqual(config['cspace']['joint_names'],[f'r_joint{i}' for i in range(1,8)])
        self.assertEqual(config['lock_joints']['platform_joint'],.42)
        self.assertEqual(config['lock_joints']['base_x'],-1.3)
        self.assertIn('l_joint7',config['lock_joints'])

    def test_left_approach_bounds_keep_original_reference(self):
        import xml.etree.ElementTree as ET
        reference=replace(self.state,joint_positions=self.state.joint_positions.copy())
        measured=replace(self.state,joint_positions=self.state.joint_positions.copy())
        index=measured.joint_names.index('l_joint3')
        measured.joint_positions[index]+=.4
        config=build_config(self.source,measured,ROOT/'tests/artifacts/left_model',self.settings,
                            arm_only=True,arm='l',reference_state=reference)['kinematics']
        self.assertEqual(config['cspace']['joint_names'],[f'l_joint{i}' for i in range(1,8)])
        self.assertIn('r_joint7',config['lock_joints'])
        self.assertNotIn('l_joint3',config['lock_joints'])
        limit=ET.parse(config['urdf_path']).getroot().find("joint[@name='l_joint3']/limit")
        self.assertLessEqual(float(limit.get('upper')),reference.joint_positions[index]+.8+1e-9)
        self.assertGreaterEqual(float(limit.get('lower')),reference.joint_positions[index]-.8-1e-9)

    def make_plan(self):
        goal=state_vector(self.state,ACTIVE_JOINTS);goal[1]+=.5;goal[3]+=.02;goal[4:]+=.05
        limits={'velocity':np.ones(15),'acceleration':np.ones(15),'jerk':np.ones(15)*10}
        plan=candidate(self.state,goal,limits,self.settings,1,1)
        plan.validation_status=plan.base_validation_status=Status.PASSED
        return plan

    def test_upper_body_moves_during_base_translation(self):
        p=self.make_plan()
        moving=np.abs(p.base_controls[:,0])>1e-6
        self.assertTrue(np.any(moving & (np.max(np.abs(p.velocities[:-1]),axis=1)>1e-6)))

    def test_environment_update_requests_measured_state(self):
        c=FakeController(self.state);e=TrajectoryExecutor(c,lambda s:False);p=self.make_plan();e.start(p,1.,1,1)
        target=TaskTarget(np.array([0,0,0,1,0,0,0]),np.array([0,0,0,1,0,0,0]),1.)
        req=ReplanningMonitor(e).update(target,EnvironmentState({},2.,2),2.)
        self.assertTrue(c.held);self.assertIs(req.current_state,c.state);self.assertEqual(e.state,RunState.REPLANNING)
        before=len(c.commands);e.tick(3.);self.assertEqual(len(c.commands),before)

    def test_invalid_or_stale_plan_never_dispatches(self):
        c=FakeController(self.state);e=TrajectoryExecutor(c,lambda s:True);p=self.make_plan()
        p.validation_status=Status.FAILED
        with self.assertRaises(ValueError):e.start(p,1.,1,1)
        p.validation_status=Status.PASSED
        with self.assertRaises(ValueError):e.start(p,1.,2,1)
        self.assertEqual(c.commands,[])

    def test_approach_failure_requests_pregrasp_without_unlocking(self):
        from planning.pipeline import ManipulationPipeline
        from types import SimpleNamespace
        controller=FakeController(self.state)
        executor=TrajectoryExecutor(controller,lambda state:True)
        calls=[]
        def factory(state,environment,arm_only,**kwargs):
            calls.append((state,arm_only))
            def fail(target,pose):
                raise RuntimeError('No arm-only IK')
            return SimpleNamespace(plan_approach=fail)
        pipeline=ManipulationPipeline(executor,factory,lambda state:np.array([0,0,0,1,0,0,0]))
        pipeline.target=TaskTarget(np.array([0,0,0,1,0,0,0]),np.array([0,0,0,1,0,0,0]),1.,
                                  task_constraints={'grasp_pose':[0,0,0,1,0,0,0]})
        pipeline.environment=EnvironmentState({'cuboid':{'Cup':{}}},1.)
        pipeline.phase='pregrasp_reached'
        pipeline.reference_state=self.state
        with self.assertRaises(RuntimeError):pipeline.plan_approach(2.)
        self.assertEqual(calls,[(self.state,True)])
        self.assertEqual(pipeline.phase,'needs_pregrasp')
        self.assertEqual(executor.state,RunState.REPLANNING)
        self.assertTrue(controller.held)
        self.assertEqual(controller.commands,[])

    def test_pose_composition_and_pink_disabled(self):
        parent=np.array([1,2,3,1,0,0,0]);child=np.array([.1,0,0,1,0,0,0])
        np.testing.assert_allclose(compose_pose(parent,child),[1.1,2,3,1,0,0,0])
        self.assertIsNone(create_local_controller())
        with self.assertRaises(NotImplementedError):create_local_controller(True)

if __name__=='__main__':unittest.main()
