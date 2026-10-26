"""Environment-only PhysX check: release a bottle at the planned tray pose."""
from pathlib import Path
import json
from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
try:
 import omni.usd
 from pxr import Usd,UsdGeom,Gf
 from isaacsim.core.api import World
 ROOT=Path(__file__).resolve().parents[1];cfg=json.loads((ROOT/'scene_config.json').read_text())
 omni.usd.get_context().open_stage(str(ROOT/'assets/environment.usda'))
 for _ in range(5):app.update()
 stage=omni.usd.get_context().get_stage();cup=stage.GetPrimAtPath('/World/Environment/Cup');xform=UsdGeom.Xformable(cup)
 xform.ClearXformOpOrder();xform.AddTranslateOp().Set(Gf.Vec3d(*cfg['placement_center_m']))
 world=World(physics_dt=1/120,rendering_dt=1/60,stage_units_in_meters=1.);world.reset()
 for _ in range(360):world.step(render=False)
 transform=xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default());position=transform.ExtractTranslation();rotation=transform.ExtractRotation().GetAngle()
 error=max(abs(position[i]-cfg['placement_center_m'][i]) for i in range(3))
 result={'passed':bool(error<.005 and abs(rotation)<5),'final_position_m':list(position),'rotation_deg':float(rotation),'max_position_error_m':float(error),'physics_steps':360,'scope':'Environment-only release on tray; no robot, grip friction or controller tracking verification'}
 (ROOT/'tests/artifacts/transport/tray_release_physics.json').write_text(json.dumps(result,indent=2));print('TRAY_RELEASE',result,flush=True);assert result['passed']
 world.stop();World.clear_instance()
finally:app.close()
