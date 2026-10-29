"""Object-physics integration, explicitly not a frictional RM75 grasp test."""
from pathlib import Path
import argparse,json,sys
parser=argparse.ArgumentParser();parser.add_argument('--fault',default=None);parser.add_argument('--suite',action='store_true');parser.add_argument('--object-xy',type=float,nargs=2);parser.add_argument('--label',default='verified_suite');parser.add_argument('--plan-cache',type=Path);parser.add_argument('--placement-xy',type=float,nargs=2,default=(0.,0.));args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
# One Isaac application at a time on this workstation. The OS releases the
# lock after process exit/crash; do not use a stale PID file as ownership.
import fcntl
lock_path = ROOT/'tests/artifacts/pnp/isaac.lock'
lock_path.parent.mkdir(parents=True, exist_ok=True)
simulation_lock = lock_path.open('a')
try:
 fcntl.flock(simulation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
 parser.exit(2, 'Another PnP Isaac test is running; finish or cancel it before starting another.\n')
from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
try:
 from pnp import Goal,PickPlaceSkill
 from pnp.isaac_assisted import IsaacAssistedContext
 config=json.loads((ROOT/'pnp/config.json').read_text())
 cases=[None,'grasp_not_established','object_lost','release_failed','cancel_held','invalid_observation'] if args.suite else [args.fault]
 rows=[]
 def physical_readback(context):
  import itertools,numpy as np
  from scipy.spatial.transform import Rotation
  xyz,quat=context.object.get_world_pose()
  vertices=np.array(list(itertools.product((-.02,.02),repeat=3)))@Rotation.from_quat(quat,scalar_first=True).as_matrix().T+xyz
  center=np.array(context.scene['tray']['center_xy_m'])
  support=context.scene['placement_center_m'][2]-.02
  inside=bool(np.all(abs(vertices[:,:2]-center)<=np.array(context.scene['tray']['inner_free_size_m'])/2-config['execution']['edge_margin_m']))
  supported=bool(abs(vertices[:,2].min()-support)<=config['execution']['position_tolerance_m'])
  linear=float(np.linalg.norm(context.object.get_linear_velocity()))
  angular=float(np.linalg.norm(context.object.get_angular_velocity()))
  desired=np.array(context.scene['placement_center_m'])+np.r_[args.placement_xy,0.]
  position_error=float(np.linalg.norm(xyz-desired))
  joint_exists=bool(context.stage.GetPrimAtPath('/World/GraspConstraint').IsValid())
  return {'source':'independent PhysX readback after execute; not skill result fields',
          'object_xyz_m':xyz.tolist(),'position_error_m':position_error,'inside_tray':inside,'supported':supported,
          'fixed_joint_exists':joint_exists,'linear_speed_m_s':linear,'angular_speed_rad_s':angular,
          'placed_and_released':position_error<=config['execution']['position_tolerance_m'] and inside and supported and not joint_exists and linear<=config['execution']['stable_linear_speed_m_s'] and angular<=config['execution']['stable_angular_speed_rad_s']}
 for case in cases:
  context=IsaacAssistedContext(ROOT/'tests/artifacts/pnp/baseline',config,case,object_xy=args.object_xy,plan_cache=args.plan_cache)
  skill=PickPlaceSkill(context,ROOT/'tests/artifacts/pnp/physx',config)
  def feedback(event):
   print('PNP_STAGE',case,json.dumps(event),flush=True)
   if case=='cancel_held' and event['stage']=='carry':skill.cancel()
  result=skill.execute(Goal('physical-'+str(case),'blue_cube',(.04,.04,.04),'source','tray',timeout_s=1800,max_retries=0,arm='l',placement_xy=tuple(args.placement_xy)),feedback)
  (ROOT/'tests/artifacts/pnp'/(args.label+'_'+str(case)+'.json')).write_text(json.dumps(result,indent=2))
  rows.append({'case':case,'result':result,'physical_readback':physical_readback(context)})
  print('PNP_RESULT',case,result['status'],result['reason'],result['holding'],flush=True)
  if case is None and result['status']=='succeeded':
   second=skill.execute(Goal('physical-second','blue_cube',(.04,.04,.04),'source','tray',timeout_s=1800,max_retries=0,arm='l',placement_xy=tuple(args.placement_xy)),feedback)
   rows.append({'case':'consecutive_already_placed','result':second,'physical_readback':physical_readback(context)})
  if case=='cancel_held':
   resumed=skill.execute(Goal('physical-resume','blue_cube',(.04,.04,.04),'source','tray',timeout_s=1800,max_retries=0,arm='l',placement_xy=tuple(args.placement_xy)),
                         feedback=lambda event:print('PNP_RESUME',json.dumps(event),flush=True))
   rows.append({'case':'resume_held_without_regrasp','result':resumed,'physical_readback':physical_readback(context)})
   stages=[json.loads(line)['stage'] for line in Path(resumed['log']).read_text().splitlines()]
   assert not set(stages)&{'open','close','approach','lift'},stages
  context.close()
 (ROOT/'tests/artifacts/pnp'/(args.label+'.json')).write_text(json.dumps(rows,indent=2))
 expected={None:('succeeded','empty'),'consecutive_already_placed':('succeeded','empty'),
  'grasp_not_established':('failed','empty'),'object_lost':('failed','empty'),
  'release_failed':('failed','holding_target'),'cancel_held':('cancelled','holding_target'),
  'invalid_observation':('failed','unknown'),'resume_held_without_regrasp':('succeeded','empty')}
 for row in rows:
  assert (row['result']['status'],row['result']['holding'])==expected[row['case']],row
  if row['result']['status']=='succeeded':assert row['physical_readback']['placed_and_released'],row
finally:app.close()
