"""Explicitly assisted Isaac adapter for a recorded, validated RM75 trajectory.

Object dynamics and release are PhysX. Robot/TCP tracking is kinematic replay;
closing creates a fixed joint, NOT frictional finger grasp. This adapter rejects
new layouts instead of pretending a cached trajectory is a general planner.
"""
from pathlib import Path
from functools import partial
import json
import hashlib
import time
from dataclasses import asdict
import numpy as np
import yaml
import yourdfpy
from scipy.spatial.transform import Rotation, Slerp
from .contracts import Pose, Surface, Snapshot, Fault
from .geometry import matrix


class IsaacAssistedContext:
    mode = 'isaac_physx_object__kinematic_robot__fixed_joint_grasp'

    def __init__(self, baseline, config, fault=None, object_xy=None, plan_cache=None):
        import omni.usd
        from pxr import UsdGeom, UsdPhysics, Gf
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleRigidPrim
        self.Gf, self.UsdPhysics = Gf, UsdPhysics
        self.folder, self.config, self.fault = Path(baseline), config, fault
        self.metadata = {'robot_tracking':'kinematic_reference', 'grasp_mechanism':'fixed_joint_assistance'}
        self.worker, self.plan_counter, self.reposition = None, 0, False
        self.report = json.loads((self.folder/'result.json').read_text())
        self.plan_cache = Path(plan_cache) if plan_cache else self.folder
        if plan_cache:
            self.report.update(json.loads((self.plan_cache/'result.json').read_text()))
        if not self.report['complete']:
            raise ValueError('Recorded plan must have all six validated stages')
        self.scene = json.loads((self.folder/'scene.json').read_text())
        self.world_data = yaml.safe_load((self.folder/'world.yml').read_text())
        omni.usd.get_context().new_stage()
        self.stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1)
        UsdGeom.SetStageUpAxis(self.stage, 'Z')
        for name, box in self.world_data['cuboid'].items():
            p = UsdGeom.Cube.Define(self.stage, '/World/Environment/'+name)
            p.CreateSizeAttr(1)
            transform = UsdGeom.Xformable(p)
            transform.AddTranslateOp().Set(Gf.Vec3d(*box['pose'][:3]))
            q = box['pose'][3:]
            transform.AddOrientOp().Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
            transform.AddScaleOp().Set(Gf.Vec3f(*box['dims']))
            UsdPhysics.CollisionAPI.Apply(p.GetPrim())
        obj = UsdGeom.Cube.Define(self.stage, '/World/Object')
        obj.CreateSizeAttr(.04)
        UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
        UsdPhysics.MassAPI.Apply(obj.GetPrim()).CreateMassAttr(.05)
        initial_object = self.report['object_pose'].copy()
        if object_xy is not None:initial_object[:2] = object_xy
        self.object = SingleRigidPrim('/World/Object', name='object', position=np.array(initial_object[:3]),
                                      orientation=np.array(self.report['object_pose'][3:]))
        tool = UsdGeom.Xform.Define(self.stage, '/World/Tool')
        UsdPhysics.RigidBodyAPI.Apply(tool.GetPrim()).CreateKinematicEnabledAttr(True)
        self.tool = SingleRigidPrim('/World/Tool', name='tool')
        self.world = World(physics_dt=config['execution']['tick_s'], rendering_dt=1/60, stage_units_in_meters=1)
        self.world.scene.add(self.object)
        self.world.reset()
        self.tool.initialize()
        self.elapsed, self.owner, self.action = 0., None, None
        self.attached, self.opening = False, .065
        self.paths = self._kinematic_paths()
        self.tcp = self.paths['pregrasp']['tcp'][0].copy()
        self.tool.set_world_pose(self.tcp[:3], self.tcp[3:])
        self.robot_state = self._robot_snapshot('initial')
        for _ in range(30):
            self._step_world()

    def _kinematic_paths(self):
        k = yaml.safe_load(Path(self.scene['robot']['curobo_config']).read_text())['kinematics']
        urdf = Path(k['urdf_path'])
        self.metadata['robot_model_sha256'] = hashlib.sha256(Path(self.scene['robot']['curobo_config']).read_bytes()+urdf.read_bytes()).hexdigest()
        self.metadata['world_sha256'] = hashlib.sha256((self.folder/'world.yml').read_bytes()).hexdigest()
        robot = yourdfpy.URDF.load(urdf, load_meshes=False, build_scene_graph=True,
                                   filename_handler=partial(yourdfpy.filename_handler_magic,dir=urdf.parent))
        initial = json.loads(Path(self.scene['robot']['initial_posture']).read_text())
        joints = dict(zip(initial['joint_names'], initial['joint_positions_rad']))
        joints.update(k['lock_joints'])
        root = Path(__file__).resolve().parents[1]
        z = json.loads((root/'scene_manifest.json').read_text())['robot_root_z_m']
        self.current_joints = joints.copy()
        self.current_base = np.array(self.report['initial_base'])
        self.base_z = z
        result = {}
        for name in ('pregrasp','approach','lift','carry','place','withdraw'):
            p = dict(np.load(self.plan_cache/(name+'.npz')))
            poses = []
            for q,b in zip(p['positions'],p['base_poses']):
                joints.update(zip(p['joint_names'],q)); robot.update_cfg(joints)
                base = np.eye(4); base[:3,:3] = Rotation.from_euler('z',b[2]).as_matrix();base[:3,3]=[*b[:2],z]
                t = base@robot.get_transform(self.report['selected_arm']+'_grasp_tcp')
                poses.append(np.r_[t[:3,3],Rotation.from_matrix(t[:3,:3]).as_quat(scalar_first=True)])
            p['tcp'] = np.array(poses)
            p['slerp'] = Slerp(p['times'], Rotation.from_quat(p['tcp'][:,3:], scalar_first=True))
            result[name] = p
        return result

    def _robot_snapshot(self, phase):
        return {'controller_mode':'kinematic_reference', 'stage':phase,
                'joint_names':list(self.current_joints), 'joint_positions':list(self.current_joints.values()),
                'base_pose':[*self.current_base[:2], self.base_z,
                             *Rotation.from_euler('z',self.current_base[2]).as_quat(scalar_first=True)],
                'timestamp':self.elapsed}

    def _plan(self, mode, goal, geometry):
        from .planner_process import PlannerProcess
        root = Path(__file__).resolve().parents[1]
        snapshot = self.observe(goal)
        table = snapshot.surfaces[goal.source_id]
        payload = {'mode':mode,'goal':asdict(goal),'geometry':geometry,
                   'state':self._robot_snapshot('planning'),
                   'object_pose':[*snapshot.object_pose.xyz,*snapshot.object_pose.wxyz],
                   'world':self.world_data,'robot_source':self.scene['robot']['curobo_config'],
                   'arm':self.report['selected_arm'],
                   'source_table':{'name':table.identity,'center_xy_m':list(table.pose.xyz[:2]),
                                   'size_m':[*table.size_xy,table.pose.xyz[2]]},
                   'settings':json.loads((root/'planning/config.json').read_text()),
                   'policy':json.loads((root/'grasp_policy.json').read_text())}
        self.plan_counter += 1
        self.worker = PlannerProcess(root/'tests/artifacts/pnp/plans'/self.owner/str(self.plan_counter), payload)

    def _matches_start(self, name):
        path = self.paths[name]
        q = np.array([self.current_joints[str(n)] for n in path['joint_names']])
        base_error = self.current_base-path['base_poses'][0]
        base_error[2] = (base_error[2]+np.pi)%(2*np.pi)-np.pi
        return np.allclose(q,path['positions'][0],atol=1e-6,rtol=0) and np.all(abs(base_error)<=1e-6)

    def acquire(self, run_id):
        if self.owner is not None:return False
        self.owner = run_id
        return True

    def release(self, run_id):
        if self.owner != run_id:raise RuntimeError('Invalid controller lease')
        self.owner = None

    def time(self):return self.elapsed

    def _step_world(self):
        self.world.step(render=False)
        self.elapsed += self.config['execution']['tick_s']

    def observe(self, goal):
        xyz, quat = self.object.get_world_pose()
        tcp_xyz, tcp_quat = self.tool.get_world_pose()
        def pose(x, q=(1.,0.,0.,0.)):
            return Pose(tuple(float(v) for v in x), tuple(float(v) for v in q), timestamp=self.elapsed,
                        valid=self.fault!='invalid_observation', source='physx_truth_feedback')
        tables = self.scene['tables']
        surfaces = {'source':Surface('source',tuple(tables[0]['size_m'][:2]),pose([*tables[0]['center_xy_m'],tables[0]['size_m'][2]])),
                    'tray':Surface('tray',tuple(self.scene['tray']['inner_free_size_m']),
                                   pose([*self.scene['tray']['center_xy_m'], self.scene['placement_center_m'][2]-.02]))}
        return Snapshot(self.scene['object']['name'], pose(xyz,quat),pose(tcp_xyz,tcp_quat),surfaces,
                        tuple(float(x) for x in self.object.get_linear_velocity()),
                        tuple(float(x) for x in self.object.get_angular_velocity()),
                        {'opening_m':self.opening,'action_complete':self.action is None,
                         'released':not self.attached,'mechanism':'explicit fixed-joint assistance'},self._robot_snapshot(self.robot_state['stage']))

    def start(self, stage, goal, geometry):
        if self.owner is None:raise RuntimeError('No execution lease')
        if self.action is not None:raise RuntimeError('Previous command still active')
        self.action, self.action_start = stage, self.elapsed
        if stage=='plan_pick':
            snapshot = self.observe(goal)
            expected = np.array(self.report['object_pose'])
            object_error = Rotation.from_quat(snapshot.object_pose.wxyz,scalar_first=True).inv()*Rotation.from_quat(expected[3:],scalar_first=True)
            if (object_error.magnitude()>.05 or goal.arm not in ('auto', self.report['selected_arm']) or goal.source_id!='source' or goal.destination_id!='tray'
                    or np.linalg.norm(np.array(snapshot.object_pose.xyz)-expected[:3])>.005
                    or np.linalg.norm(np.array(geometry['object_goal'])[:3,3]-self.report['object_goal_pose'][:3])>.005
                    or Rotation.from_matrix(Rotation.from_quat(self.report['object_goal_pose'][3:],scalar_first=True).as_matrix().T@np.array(geometry['object_goal'])[:3,:3]).magnitude()>.05):
                self._plan('pick', goal, geometry)
                return
            if not self._matches_start('pregrasp') or np.linalg.norm(np.array(snapshot.tcp_pose.xyz)-self.paths['pregrasp']['tcp'][0,:3])>.005:
                self._plan('pick', goal, geometry)
                return
        if stage=='plan_transport':
            current = self.observe(goal).tcp_pose
            if not self._matches_start('carry') or np.linalg.norm(np.array(current.xyz)-self.paths['carry']['tcp'][0,:3])>.005:
                self._plan('held', goal, geometry)
                return
        if stage=='preplace' and self.reposition:
            self.action = stage = 'carry'
            self.reposition = False
        if stage=='plan_place':
            planned = self.paths['place']['tcp'][-1]
            desired = np.array(geometry['tcp_place_goal'])
            rotation = Rotation.from_quat(planned[3:],scalar_first=True).as_matrix()
            if (np.linalg.norm(planned[:3]-desired[:3,3])>.005
                    or Rotation.from_matrix(rotation.T@desired[:3,:3]).magnitude()>.05):
                self._plan('held', goal, geometry)
                return
        if stage in self.paths:
            measured = np.array(self.observe(goal).tcp_pose.xyz)
            if not self._matches_start(stage) or np.linalg.norm(measured-self.paths[stage]['tcp'][0,:3])>.005:
                raise Fault('requires_replanning', 'Measured TCP differs from segment start; no teleport permitted')
        if stage=='close' and self.fault!='grasp_not_established':
            snapshot = self.observe(goal)
            if np.linalg.norm(np.array(snapshot.object_pose.xyz)-snapshot.tcp_pose.xyz)>.005:
                raise Fault('grasp_not_established','TCP is not at observed cube centre')
            relative = np.linalg.inv(matrix(snapshot.tcp_pose))@matrix(snapshot.object_pose)
            from pxr import Sdf, Usd
            joint_stage = Usd.Stage.CreateInMemory()
            joint = self.UsdPhysics.FixedJoint.Define(joint_stage,'/GraspConstraint')
            joint.CreateBody0Rel().SetTargets(['/World/Tool'])
            joint.CreateBody1Rel().SetTargets(['/World/Object'])
            joint.CreateLocalPos0Attr().Set(self.Gf.Vec3f(*relative[:3,3]))
            q = Rotation.from_matrix(relative[:3,:3]).as_quat(scalar_first=True)
            joint.CreateLocalRot0Attr().Set(self.Gf.Quatf(q[0],self.Gf.Vec3f(*q[1:])))
            joint.CreateLocalPos1Attr().Set(self.Gf.Vec3f(0,0,0))
            joint.CreateLocalRot1Attr().Set(self.Gf.Quatf(1,self.Gf.Vec3f(0,0,0)))
            Sdf.CopySpec(joint_stage.GetRootLayer(), '/GraspConstraint', self.stage.GetRootLayer(), '/World/GraspConstraint')
            self.attached, self.opening = True, .04
        if stage in ('open','release'):
            if stage=='release' and self.fault=='release_failed':return
            if self.attached:self.stage.RemovePrim('/World/GraspConstraint')
            self.attached, self.opening = False, geometry['opening_m']

    def step(self):
        if self.worker is not None:
            if not self.worker.poll():
                self._step_world()
                time.sleep(.01)
                return False
            report = json.loads((self.worker.directory/'result.json').read_text())
            for name in report['stages']:
                p = dict(np.load(self.worker.directory/(name+'.npz')))
                p['tcp'] = p['tcp_poses']
                p['slerp'] = Slerp(p['times'],Rotation.from_quat(p['tcp'][:,3:],scalar_first=True))
                self.paths[name] = p
            self.report.update(report)
            self.reposition = self.action=='plan_place'
            self.worker = None
            self.action = None
            return True
        action = self.action
        if action in self.paths:
            path = self.paths[action]
            elapsed = min(self.elapsed-self.action_start, float(path['times'][-1]))
            xyz = np.array([np.interp(elapsed,path['times'],path['tcp'][:,j]) for j in range(3)])
            quat = path['slerp'](elapsed).as_quat(scalar_first=True)
            self.tool.set_world_pose(xyz,quat)
            self.tcp = np.r_[xyz,quat]
            self.current_base = np.array([float(np.interp(elapsed,path['times'],path['base_poses'][:,i])) for i in range(3)])
            self.current_joints.update({str(name):float(np.interp(elapsed,path['times'],path['positions'][:,i]))
                                        for i,name in enumerate(path['joint_names'])})
            self.robot_state = self._robot_snapshot(action)
            done = elapsed >= path['times'][-1]
        else:
            done = True
        if self.fault=='object_lost' and action=='carry' and self.attached:
            self.stage.RemovePrim('/World/GraspConstraint');self.attached=False
        self._step_world()
        if done:self.action=None
        return done

    def hold(self):
        if self.worker is not None:
            self.worker.cancel()
            self.worker = None
        self.action = None  # Kinematic TCP stays; the physical joint remains if held.
        return True

    def close(self):
        from isaacsim.core.api import World
        self.world.stop()
        World.clear_instance()
