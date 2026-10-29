"""Small synchronous action API with cooperative cancellation and evidence gates."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import threading
import time
import uuid
import numpy as np
from scipy.spatial.transform import Rotation
from .contracts import Fault
from .geometry import derive, matrix, on_surface


class PickPlaceSkill:
    def __init__(self, context, log_directory, config=None):
        self.context = context
        self.config = config or json.loads(Path(__file__).with_name('config.json').read_text())
        self.logs = Path(log_directory)
        self.lock = threading.Lock()
        self.cancelled = threading.Event()
        self.completed = {}
        self.held_transform = None
        self.held_object = None
        self.stage = 'idle'

    def cancel(self):
        self.cancelled.set()

    def _checkpoint(self):
        if self.cancelled.is_set():
            raise Fault('cancelled', 'Cancellation requested')
        if time.monotonic() >= self.deadline:
            raise Fault('timeout', 'Total wall-clock deadline exceeded')

    def _observe(self, goal):
        state = self.context.observe(goal)
        if state.object_id != goal.object_id:
            raise Fault('invalid_observation', 'Observation object identity mismatch')
        poses = [state.object_pose, state.tcp_pose]
        for identity in (goal.source_id, goal.destination_id):
            if identity not in state.surfaces:
                raise Fault('invalid_observation', 'Missing support '+identity)
            poses.append(state.surfaces[identity].pose)
        for pose in poses:
            matrix(pose)
            age = self.context.time()-pose.timestamp
            if age < -1e-6 or age > self.config['execution']['observation_max_age_s']:
                raise Fault('stale_observation', f'Observation age {age:.6f}s')
        self.last = state
        return state

    def _emit(self, stage, **details):
        self.stage = stage
        event = dict(stage=stage, elapsed_s=time.monotonic()-self.started, **details)
        if self.last is not None:
            event['snapshot'] = asdict(self.last)
        self.events.write(json.dumps(event)+'\n')
        self.events.flush()
        self.feedback({k: v for k, v in event.items() if k != 'snapshot'})

    def _relative(self, snapshot):
        return np.linalg.inv(matrix(snapshot.tcp_pose))@matrix(snapshot.object_pose)

    def _same_grasp(self, snapshot):
        if self.held_transform is None:
            return False
        error = np.linalg.inv(self.held_transform)@self._relative(snapshot)
        cfg = self.config['execution']
        return (np.linalg.norm(error[:3, 3]) <= cfg['position_tolerance_m']
                and Rotation.from_matrix(error[:3, :3]).magnitude() <= cfg['rotation_tolerance_rad'])

    def _classify(self, snapshot):
        if snapshot.gripper.get('released') is True:
            self.held_transform = self.held_object = None
            return 'empty'
        if self.held_object == snapshot.object_id and self._same_grasp(snapshot):
            return 'holding_target'
        return 'unknown'

    def _action(self, stage, goal, geometry, monitor_hold=False):
        self._checkpoint()
        self._emit(stage)
        self._checkpoint()
        self.context.start(stage, goal, geometry)
        start = time.monotonic()
        while True:
            self._checkpoint()
            done = self.context.step()
            snapshot = self._observe(goal)
            if monitor_hold and (snapshot.gripper.get('released') is True or not self._same_grasp(snapshot)):
                self.held_transform = self.held_object = None
                raise Fault('object_lost', 'Measured object no longer follows held transform')
            if done:
                return snapshot
            if not stage.startswith('plan_') and time.monotonic()-start > self.config['execution']['stage_timeout_s']:
                raise Fault('execution_timeout', f'{stage} did not report measured completion')

    def _verify_hold(self, goal):
        self._emit('confirm_grasp')
        cfg = self.config['execution']
        initial = self._observe(goal)
        relative = self._relative(initial)
        began = self.context.time()
        while self.context.time()-began < cfg['verification_window_s']:
            self._checkpoint()
            self.context.step()
            snapshot = self._observe(goal)
            _, _, bottom = on_surface(snapshot.object_pose, goal.dimensions,
                                      snapshot.surfaces[goal.source_id], 0, cfg['position_tolerance_m'])
            error = np.linalg.inv(relative)@self._relative(snapshot)
            if (bottom < goal.dimensions[2]/2 or snapshot.gripper.get('released') is True
                    or np.linalg.norm(error[:3, 3]) > cfg['position_tolerance_m']
                    or Rotation.from_matrix(error[:3, :3]).magnitude() > cfg['rotation_tolerance_rad']):
                self.held_transform = self.held_object = None
                raise Fault('grasp_not_established', 'Object not lifted and retained relative to measured TCP')
        self.held_transform, self.held_object = relative, goal.object_id

    def pick(self, goal, geometry):
        """Reusable under execute's controller lease/deadline; not a second owner."""
        for stage in ('plan_pick', 'open', 'pregrasp', 'approach', 'close', 'lift'):
            self._action(stage, goal, geometry)
        self._verify_hold(goal)

    def transport(self, goal, geometry):
        geometry['tcp_from_object'] = self.held_transform.tolist()
        self._action('plan_transport', goal, geometry, monitor_hold=True)
        self._action('transport_posture', goal, geometry, monitor_hold=True)
        self._action('carry', goal, geometry, monitor_hold=True)

    def place(self, goal, geometry):
        self._emit('update_destination')
        snapshot = self._observe(goal)
        geometry.update(derive(goal, snapshot, self.config))
        # Always use the measured grasp transform, not a nominal cube-centre offset.
        geometry['tcp_from_object'] = self.held_transform.tolist()
        geometry['tcp_place_goal'] = (np.array(geometry['object_goal'])@np.linalg.inv(self.held_transform)).tolist()
        self._action('plan_place', goal, geometry, monitor_hold=True)
        self._action('preplace', goal, geometry, monitor_hold=True)
        self._action('place', goal, geometry, monitor_hold=True)
        self._action('release', goal, geometry)
        self._action('withdraw', goal, geometry)
        self._verify_placement(goal, geometry)

    def _placement_matches(self, snapshot, geometry):
        error = np.linalg.inv(np.array(geometry['object_goal']))@matrix(snapshot.object_pose)
        delta = Rotation.from_matrix(error[:3, :3])
        equivalent = min((delta*Rotation.from_euler('z', k*np.pi/2)).magnitude() for k in range(4))
        cfg = self.config['execution']
        return (np.linalg.norm(error[:3, 3]) <= cfg['position_tolerance_m']
                and equivalent <= cfg['rotation_tolerance_rad'])

    def _verify_placement(self, goal, geometry):
        self._emit('confirm_placement')
        cfg = self.config['execution']
        began = self.context.time()
        previous = self._observe(goal)
        while self.context.time()-began < cfg['verification_window_s']:
            self._checkpoint()
            self.context.step()
            snapshot = self._observe(goal)
            inside, supported, _ = on_surface(snapshot.object_pose, goal.dimensions,
                snapshot.surfaces[goal.destination_id], cfg['edge_margin_m'], cfg['position_tolerance_m'])
            stable = (np.linalg.norm(snapshot.linear_velocity) <= cfg['stable_linear_speed_m_s']
                      and np.linalg.norm(snapshot.angular_velocity) <= cfg['stable_angular_speed_rad_s']
                      and np.linalg.norm(np.array(snapshot.object_pose.xyz)-previous.object_pose.xyz)
                          <= cfg['position_tolerance_m'])
            if snapshot.gripper.get('released') is not True:
                raise Fault('result_uncertain', 'Release not independently confirmed')
            if not (inside and supported and stable):
                raise Fault('placement_not_satisfied', 'Footprint/support/stability verification failed after withdrawal')
            if not self._placement_matches(snapshot, geometry):
                raise Fault('placement_not_satisfied', 'Object pose outside requested placement tolerance')
            previous = snapshot
        self.held_transform = self.held_object = None

    def execute(self, goal, feedback=lambda event: None):
        if not self.lock.acquire(blocking=False):
            return self._admission_failure('busy')
        try:
            return self._execute(goal, feedback)
        finally:
            self.lock.release()

    @staticmethod
    def _admission_failure(reason):
        return dict(status='failed', reason=reason, holding='unknown', final_stage='admission',
                    retries=0, elapsed_s=0., run_id=None, log=None, last_state=None,
                    first_attempt_success=False, recovered_success=False)

    def _execute(self, goal, feedback):
        fingerprint = hashlib.sha256(json.dumps(asdict(goal), sort_keys=True).encode()).hexdigest()
        leased = False
        stop_confirmed = False
        self.started = time.monotonic()
        self.deadline = self.started+goal.timeout_s
        self.cancelled.clear()
        self.feedback, self.last = feedback, None
        run_id = uuid.uuid4().hex
        retries = 0
        failures = []
        try:
            if goal.request_id in self.completed:
                old_hash, old = self.completed[goal.request_id]
                if old_hash != fingerprint:
                    return self._admission_failure('request_id_conflict')
                return old
            self.logs.mkdir(parents=True, exist_ok=True)
            self.events = (self.logs/(run_id+'.jsonl')).open('w')
            if not self.context.acquire(run_id):
                raise Fault('busy', 'Executor context already has an owner')
            leased = True
            self._emit('validate', goal=asdict(goal), mode=self.context.mode)
            if (not all(isinstance(x,str) and x for x in (goal.request_id,goal.object_id,goal.source_id,goal.destination_id))
                    or not np.isfinite(goal.timeout_s) or goal.timeout_s<=0 or type(goal.max_retries) is not int
                    or goal.max_retries<0 or goal.arm not in ('l','r','auto')
                    or not np.isfinite([*goal.placement_xy,goal.placement_yaw]).all()
                    or len(goal.placement_xy)!=2):
                raise Fault('invalid_goal', 'Invalid timeout, retry budget, arm or placement')
            while True:
                try:
                    self._checkpoint()
                    self._emit('update_object')
                    snapshot = self._observe(goal)
                    geometry = derive(goal, snapshot, self.config)
                    holding = self._classify(snapshot)
                    if holding=='unknown':
                        self._verify_hold(goal)
                        holding='holding_target'
                    if holding=='empty':
                        at_target, supported, _ = on_surface(snapshot.object_pose, goal.dimensions,
                            snapshot.surfaces[goal.destination_id], self.config['execution']['edge_margin_m'],
                            self.config['execution']['position_tolerance_m'])
                        if at_target and supported and self._placement_matches(snapshot, geometry):
                            self._verify_placement(goal, geometry)
                            status, reason = 'succeeded', ''
                            break
                        inside, supported, _ = on_surface(snapshot.object_pose, goal.dimensions,
                            snapshot.surfaces[goal.source_id], 0., self.config['execution']['position_tolerance_m'])
                        if not (inside and supported) or not np.allclose(matrix(snapshot.object_pose)[:3, 2], [0,0,1], atol=1e-4):
                            raise Fault('invalid_observation', 'Cube must rest upright within source support')
                        self.pick(goal, geometry)
                    self.transport(goal, geometry)
                    self.place(goal, geometry)
                    status, reason = 'succeeded', ''
                    break
                except Fault as error:
                    failures.append({'stage':self.stage,'code':error.code,'message':str(error)})
                    if not self.context.hold():
                        raise Fault('control_stop_failed', 'Controller did not confirm hold')
                    self._emit(self.stage, failure=failures[-1])
                    retryable = error.code in ('target_unreachable','no_collision_free_path','grasp_not_established')
                    if not retryable or retries>=goal.max_retries:
                        raise
                    snapshot = self._observe(goal)
                    holding = self._classify(snapshot)
                    if holding!='empty':
                        raise
                    retries += 1
                    self._emit('retry', reason=error.code, attempt=retries)
        except Fault as error:
            status = 'cancelled' if error.code=='cancelled' else 'failed'
            reason = error.code
            if not failures or failures[-1]['code']!=error.code:
                failures.append({'stage':self.stage,'code':error.code,'message':str(error)})
        except Exception as error:
            status, reason = 'failed', 'adapter_error'
            failures.append({'stage':self.stage,'code':reason,'message':repr(error)})
        finally:
            if leased:
                try:
                    stop_confirmed = bool(self.context.hold())
                    if not stop_confirmed:
                        raise RuntimeError('Hold unconfirmed; retaining context ownership')
                    self.context.release(run_id)
                except Exception as error:
                    status, reason = 'failed', 'control_stop_failed'
                    failures.append({'stage':self.stage, 'code':reason, 'message':repr(error)})
            if hasattr(self, 'events') and not self.events.closed:
                self.events.close()
        # Observe even on cancel/timeout; no commands and no default gripper-open.
        try:
            last = self._observe(goal)
            holding = self._classify(last)
        except Exception:
            holding = 'unknown'
        result = dict(status=status, reason=reason, final_stage=self.stage, holding=holding,
                      control_hold_confirmed=stop_confirmed, retries=retries, elapsed_s=time.monotonic()-self.started, run_id=run_id,
                      log=str(self.logs/(run_id+'.jsonl')), mode=self.context.mode, goal=asdict(goal),
                      context_metadata=getattr(self.context, 'metadata', {}),
                      first_failure=failures[0] if failures else None, failures=failures,
                      first_attempt_success=status=='succeeded' and retries==0,
                      recovered_success=status=='succeeded' and retries>0,
                      last_state=asdict(self.last) if self.last else None,
                      config_sha256=hashlib.sha256(json.dumps(self.config,sort_keys=True).encode()).hexdigest())
        (self.logs/(run_id+'.json')).write_text(json.dumps(result, indent=2)+'\n')
        if leased:
            self.completed[goal.request_id] = fingerprint, result
        return result
