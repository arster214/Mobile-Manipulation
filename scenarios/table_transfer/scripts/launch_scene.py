"""Open the two-table task scene paused; --validate runs environment physics and exits."""
from pathlib import Path
import argparse,json
parser=argparse.ArgumentParser();parser.add_argument('--headless',action='store_true');parser.add_argument('--validate',action='store_true');parser.add_argument('--check-posture',action='store_true');args=parser.parse_args()
from isaacsim import SimulationApp
app=SimulationApp({'headless':args.headless or args.validate or args.check_posture,'width':1280,'height':720})
from pxr import Usd,UsdGeom,UsdPhysics,Gf
import omni.usd,omni.timeline
ROOT=Path(__file__).resolve().parents[1];cfg=json.loads((ROOT/'scene_config.json').read_text())
try:
 if args.validate:
  omni.usd.get_context().open_stage(str(ROOT/'assets/environment.usda'))
  for _ in range(5):app.update()
  from isaacsim.core.api import World
  world=World(physics_dt=1/120,rendering_dt=1/60,stage_units_in_meters=1.)
  world.reset()
  for _ in range(360):world.step(render=False)
  stage=omni.usd.get_context().get_stage();cup=stage.GetPrimAtPath('/World/Environment/Cup');pose=UsdGeom.Xformable(cup).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
  expected=cfg['tables'][0]['size_m'][2]+cfg['cup']['height_m']/2
  print('CUP_POSE',list(pose),'EXPECTED_Z',expected,flush=True)
  assert abs(pose[2]-expected)<.005, f'Cup not resting on tabletop: {pose}'
  assert abs(pose[0]-cfg['cup']['initial_center_m'][0])<.01 and abs(pose[1]-cfg['cup']['initial_center_m'][1])<.01
  assert cup.HasAPI(UsdPhysics.RigidBodyAPI)
  assert not stage.GetPrimAtPath('/World/Environment/PlaceTarget').HasAPI(UsdPhysics.CollisionAPI)
  world.stop();World.clear_instance()
  (ROOT/'validation.json').write_text(json.dumps({'physics_steps':360,'physics_dt_s':1/120,'cup_final_center_m':list(pose),'expected_support_center_z_m':expected,'cup_resting_on_source_table':True,'scope':'Environment-only rigid-body physics; full-robot grasp/reach/navigation not tested'},indent=2))
 omni.timeline.get_timeline_interface().stop()
 omni.usd.get_context().open_stage(str(ROOT/'assets/rm75_table_transfer.usda'))
 for _ in range(15):app.update()
 stage=omni.usd.get_context().get_stage();assert stage.GetPrimAtPath('/World/Robot/Geometry').IsValid()
 # Apply the experiment initial state to the simulated articulation on every launch.
 if cfg['robot'].get('initial_posture'):
  import numpy as np, yaml
  from isaacsim.core.experimental.prims import Articulation
  posture=json.loads(Path(cfg['robot']['initial_posture']).read_text())
  robot_config=yaml.safe_load(Path(posture['curobo_config']).read_text())['kinematics']
  positions=dict(zip(posture['joint_names'],posture['joint_positions_rad']))
  positions.update(robot_config.get('lock_joints',{}))
  articulation=Articulation('/World/Robot')
  timeline=omni.timeline.get_timeline_interface()
  timeline.play()
  for _ in range(3):app.update()
  timeline.pause()
  missing=set(positions)-set(articulation.dof_names)
  assert not missing, f'Initial posture joints missing from USD: {missing}'
  indices=articulation.get_dof_indices(list(positions))
  values=np.array(list(positions.values()),dtype=np.float32)[None,:]
  articulation.set_dof_positions(values,dof_indices=indices)
  articulation.set_dof_velocities(np.zeros_like(values),dof_indices=indices)
  articulation.set_dof_position_targets(values,dof_indices=indices)
  articulation.set_default_state(dof_positions=values,dof_velocities=np.zeros_like(values),dof_indices=indices)
  for _ in range(3):app.update()
  actual=articulation.get_dof_positions(dof_indices=indices).numpy()
  error=float(np.max(np.abs(actual-values)))
  assert error<1e-5, f'Initial joint readback mismatch: {error}'
  (ROOT/'initial_posture_validation.json').write_text(json.dumps({'posture':cfg['robot']['initial_posture'],'joint_names':list(positions),'readback_positions':actual.tolist(),'max_joint_error':error,'scope':'Simulated articulation initialization and readback; no grasp trajectory validation'},indent=2))
  print('INITIAL_POSTURE_OK',error,flush=True)
 from omni.kit.viewport.utility import get_active_viewport
 vp=get_active_viewport()
 if vp:vp.set_active_camera('/World/OverviewCamera')
 if args.check_posture:
  pass
 elif args.validate:
  import omni.replicator.core as rep
  import numpy as np
  from PIL import Image
  product=rep.create.render_product('/World/OverviewCamera',(1280,720));rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product])
  for _ in range(8):rep.orchestrator.step(rt_subframes=4,pause_timeline=True)
  data=rgb.get_data();assert data.size>0
  Image.fromarray(np.asarray(data)).save(ROOT/'scene_preview.png')
  print('VALIDATION_OK preview',ROOT/'scene_preview.png',flush=True)
 else:
  print('Scene ready, paused. Environment configured; grasp controller not connected.',flush=True)
  while app.is_running():app.update()
finally:app.close()
