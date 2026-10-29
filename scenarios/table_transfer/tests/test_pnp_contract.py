"""Deterministic feedback-model tests, NOT robot/IK/PhysX success evidence."""
from pathlib import Path
from dataclasses import replace
import json,sys,unittest
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pnp import Goal,Pose,Surface,Snapshot,Fault,PickPlaceSkill
from pnp.geometry import derive,on_surface
OUT=ROOT/'tests/artifacts/pnp/contract';OUT.mkdir(parents=True,exist_ok=True)
CONFIG=json.loads((ROOT/'pnp/config.json').read_text())

class FeedbackModel:
 mode='synthetic_feedback_contract_test_not_robot_simulation'
 def __init__(self,xy=(-1.05,1.32),height=.75,yaw=0.,target=(1.2,1.32),fault=None):
  self.t=0.;self.xyz=np.array([*xy,height+.02]);self.quat=Rotation.from_euler('z',yaw).as_quat(scalar_first=True)
  self.tcp=self.xyz.copy();self.tcp_quat=np.array([1.,0.,0.,0.]);self.held=False;self.owner=None;self.action=None;self.fault=fault;self.actions=[];self.closes=0
  self.source=Surface('source',(1.4,.6),Pose((-1.2,1.5,height)))
  self.target=Surface('tray',(.226,.156),Pose((*target,height+.002)))
 def time(self):return self.t
 def acquire(self,id):
  if self.owner:return False
  self.owner=id;return True
 def release(self,id):assert id==self.owner;self.owner=None
 def hold(self):self.action=None;return True
 def observe(self,goal):
  timestamp=self.t-1 if self.fault=='stale' else self.t
  def p(x,q=(1,0,0,0)):return Pose(tuple(x),tuple(q),timestamp=timestamp,source='synthetic',valid=self.fault!='invalid')
  return Snapshot('blue_cube',p(self.xyz,self.quat),p(self.tcp,self.tcp_quat),
   {'source':replace(self.source,pose=p(self.source.pose.xyz)), 'tray':replace(self.target,pose=p(self.target.pose.xyz))},
   (0,0,0),(0,0,0),{'released':not self.held,'opening_m':.04 if self.held else .05})
 def start(self,stage,goal,geometry):
  assert self.owner and self.action is None
  self.actions.append(stage);self.action=stage;self.geometry=geometry
  if self.fault=='unreachable' and stage=='plan_pick':raise Fault('target_unreachable','injected IK exhaustion')
 def step(self):
  self.t+=.01;a=self.action
  if a=='approach':self.tcp=self.xyz.copy()
  if a=='close':
   self.closes+=1;self.held=not(self.fault=='no_grip' or self.fault=='retry_once' and self.closes==1)
  if a=='lift':
   self.tcp[2]+=self.geometry['lift_m']
   if self.held:self.xyz[2]+=self.geometry['lift_m']
  if a=='carry':
   if self.fault=='lost':self.held=False
   offset=self.xyz-self.tcp;self.tcp=np.array(self.geometry['object_goal'])[:3,3]-offset+[0,0,.1]
   if self.held:self.xyz=self.tcp+offset
  if a=='place':
   offset=self.xyz-self.tcp;self.xyz=np.array(self.geometry['object_goal'])[:3,3].copy();self.tcp=self.xyz-offset
   desired=Rotation.from_matrix(np.array(self.geometry['object_goal'])[:3,:3])
   self.tcp_quat=(desired*Rotation.from_quat(self.quat,scalar_first=True).inv()*Rotation.from_quat(self.tcp_quat,scalar_first=True)).as_quat(scalar_first=True)
   self.quat=desired.as_quat(scalar_first=True)
  if a=='release' and self.fault!='release_failed':self.held=False
  if a=='withdraw':
   self.tcp[1]-=.15
   if self.held:self.xyz[1]-=.15
   if self.fault=='outside':self.xyz[0]+=.2
  self.action=None;return True

class SkillTests(unittest.TestCase):
 def goal(self,id='test',**kw):return Goal(id,'blue_cube',(.04,.04,.04),'source','tray',**kw)
 def skill(self,ctx):return PickPlaceSkill(ctx,OUT,CONFIG)
 def test_fixed_normal_geometry_matrix(self):
  rng=np.random.default_rng(29);rows=[]
  for i in range(12):
   ctx=FeedbackModel(xy=(-1.2+rng.uniform(-.2,.2),1.5+rng.uniform(-.1,.1)),height=[.7,.75,.79][i%3],
                     yaw=rng.uniform(-np.pi,np.pi),target=(1.2+rng.uniform(-.1,.1),1.32))
   goal=self.goal(f'normal-{i}',placement_xy=(float(rng.uniform(-.03,.03)),float(rng.uniform(-.02,.02))),placement_yaw=float(rng.uniform(-np.pi,np.pi)))
   result=self.skill(ctx).execute(goal);self.assertEqual(result['status'],'succeeded',result)
   rows.append({'case':i,'seed':29,'goal':goal.__dict__,'initial_source':ctx.source.pose.xyz,'result':result})
  (OUT/'normal_cases.json').write_text(json.dumps(rows,indent=2))
  self.assertEqual(len({r['result']['config_sha256'] for r in rows}),1)
 def test_failure_codes_and_hold_state(self):
  for fault,reason,holding in [('invalid','invalid_observation','unknown'),('stale','stale_observation','unknown'),
   ('no_grip','grasp_not_established','empty'),('lost','object_lost','empty'),
   ('release_failed','result_uncertain','holding_target'),('outside','placement_not_satisfied','empty'),
   ('unreachable','target_unreachable','empty')]:
   with self.subTest(fault=fault):
    result=self.skill(FeedbackModel(fault=fault)).execute(self.goal(fault,max_retries=0))
    self.assertEqual((result['reason'],result['holding']),(reason,holding));self.assertEqual(result['status'],'failed')
 def test_retry_refreshes_and_records_first_failure(self):
  ctx=FeedbackModel(fault='retry_once');r=self.skill(ctx).execute(self.goal('retry',max_retries=1))
  self.assertTrue(r['recovered_success']);self.assertFalse(r['first_attempt_success']);self.assertEqual(r['retries'],1)
  self.assertEqual(r['first_failure']['code'],'grasp_not_established')
 def test_cancel_held_and_resume_without_regrasp(self):
  ctx=FeedbackModel();skill=self.skill(ctx)
  r=skill.execute(self.goal('cancel'),lambda event:skill.cancel() if event['stage']=='carry' else None)
  self.assertEqual((r['status'],r['holding']),('cancelled','holding_target'));self.assertTrue(ctx.held)
  count=ctx.actions.count('close');r=skill.execute(self.goal('resume'))
  self.assertEqual(r['status'],'succeeded');self.assertEqual(count,ctx.actions.count('close'))
 def test_cancel_before_grasp_confirmation_is_unknown(self):
  ctx=FeedbackModel();skill=self.skill(ctx)
  r=skill.execute(self.goal(),lambda e:skill.cancel() if e['stage']=='lift' else None)
  self.assertEqual((r['status'],r['holding']),('cancelled','unknown'));self.assertTrue(ctx.held)
 def test_two_calls_idempotency_conflict_and_busy(self):
  ctx=FeedbackModel();skill=self.skill(ctx);busy=[];goal=self.goal('first')
  def feedback(e):
   if e['stage']=='pregrasp':busy.append(skill.execute(self.goal('concurrent'))['reason'])
  r=skill.execute(goal,feedback);count=len(ctx.actions)
  self.assertEqual(skill.execute(goal),r);self.assertEqual(len(ctx.actions),count)
  self.assertEqual(skill.execute(replace(goal,placement_xy=(.01,0)))['reason'],'request_id_conflict')
  second=skill.execute(self.goal('second'));self.assertEqual(second['status'],'succeeded')
  self.assertEqual(len(ctx.actions),count);self.assertEqual(busy,['busy']);self.assertIsNone(ctx.owner)
 def test_new_target_on_same_surface_is_not_an_already_done_task(self):
  context=FeedbackModel();skill=self.skill(context)
  self.assertEqual(skill.execute(self.goal('first-slot'))['status'],'succeeded')
  goal=replace(self.goal('second-slot'),source_id='tray',placement_xy=(.02,0.))
  result=skill.execute(goal)
  self.assertEqual(result['status'],'succeeded',result)
  self.assertEqual(context.actions.count('close'),2)

 def test_cached_segments_match_full_joint_and_base_start(self):
  import yaml
  from types import SimpleNamespace
  from pnp.isaac_assisted import IsaacAssistedContext
  folder=ROOT/'tests/artifacts/pnp/baseline'
  scene=json.loads((folder/'scene.json').read_text());report=json.loads((folder/'result.json').read_text())
  initial=json.loads(Path(scene['robot']['initial_posture']).read_text())
  values=dict(zip(initial['joint_names'],initial['joint_positions_rad']))
  values.update(yaml.safe_load(Path(scene['robot']['curobo_config']).read_text())['kinematics']['lock_joints'])
  fake=SimpleNamespace(current_joints=values,current_base=np.array(report['initial_base']),paths={})
  for name in ('pregrasp','approach','lift','carry','place','withdraw'):
   data=dict(np.load(folder/(name+'.npz')));fake.paths[name]=data
   self.assertTrue(IsaacAssistedContext._matches_start(fake,name),name)
   fake.current_joints.update(zip(data['joint_names'],data['positions'][-1]))
   fake.current_base=data['base_poses'][-1].copy()
  fake.current_joints[str(data['joint_names'][0])]+=.1
  self.assertFalse(IsaacAssistedContext._matches_start(fake,'withdraw'))

 def test_context_busy_request_can_be_retried_after_owner_releases(self):
  context=FeedbackModel();skill=self.skill(context);goal=self.goal('lease')
  self.assertTrue(context.acquire('other'))
  self.assertEqual(skill.execute(goal)['reason'],'busy')
  self.assertEqual(context.owner,'other')
  context.release('other')
  self.assertEqual(skill.execute(goal)['status'],'succeeded')

 def test_planning_worker_is_terminated_on_cancel(self):
  import subprocess
  from unittest.mock import patch
  from pnp.planner_process import PlannerProcess
  real_popen=subprocess.Popen
  def sleeper(*args,**kwargs):
   return real_popen([sys.executable,'-c','import time;time.sleep(60)'],**kwargs)
  with patch('pnp.planner_process.subprocess.Popen',side_effect=sleeper):
   worker=PlannerProcess(OUT/'worker_cancel',{})
  worker.cancel()
  self.assertIsNotNone(worker.process.poll())

 def test_deadline_and_footprint_preconditions(self):
  r=self.skill(FeedbackModel()).execute(self.goal(timeout_s=1e-9));self.assertEqual(r['reason'],'timeout')
  r=self.skill(FeedbackModel()).execute(self.goal(placement_xy=(.10,0),placement_yaw=np.pi/4))
  self.assertEqual(r['reason'],'invalid_goal')

if __name__=='__main__':unittest.main()
