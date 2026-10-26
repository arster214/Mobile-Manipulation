from pathlib import Path
import json,os
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':1280,'height':720})
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,UsdLux,Gf,Sdf
ROOT=Path(__file__).resolve().parents[1];cfg=json.loads((ROOT/'scene_config.json').read_text());assets=ROOT/'assets';assets.mkdir(exist_ok=True)
s=Usd.Stage.CreateNew(str(assets/'environment.usda'));UsdGeom.SetStageMetersPerUnit(s,1);UsdGeom.SetStageUpAxis(s,UsdGeom.Tokens.z)
w=UsdGeom.Xform.Define(s,'/World');s.SetDefaultPrim(w.GetPrim());UsdGeom.Xform.Define(s,'/World/Environment')
scene=UsdPhysics.Scene.Define(s,'/World/PhysicsScene');scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81)
mat=UsdShade.Material.Define(s,'/World/ContactMaterial');pm=UsdPhysics.MaterialAPI.Apply(mat.GetPrim());pm.CreateStaticFrictionAttr(.7);pm.CreateDynamicFrictionAttr(.6);pm.CreateRestitutionAttr(0.)
boxes={}
def box(name,center,dims,color):
 p=UsdGeom.Cube.Define(s,'/World/Environment/'+name);p.CreateSizeAttr(1);p.CreateDisplayColorAttr([Gf.Vec3f(*color)])
 x=UsdGeom.Xformable(p);x.AddTranslateOp().Set(Gf.Vec3d(*center));x.AddScaleOp().Set(Gf.Vec3f(*dims));UsdPhysics.CollisionAPI.Apply(p.GetPrim());UsdShade.MaterialBindingAPI.Apply(p.GetPrim()).Bind(mat,materialPurpose='physics')
 boxes[name]={'dims':dims,'pose':[*center,1,0,0,0]};return p
room=cfg['room'];a,b=room['interior_size_m'];t=room['wall_thickness_m'];h=room['wall_height_m']
box('Floor',[0,0,-.05],[a+2*t,b+2*t,.1],[.65,.69,.73])
for name,c,d in [('NorthWall',[0,b/2+t/2,h/2],[a+2*t,t,h]),('SouthWall',[0,-b/2-t/2,h/2],[a+2*t,t,h]),('EastWall',[a/2+t/2,0,h/2],[t,b,h]),('WestWall',[-a/2-t/2,0,h/2],[t,b,h])]:box(name,c,d,[.83,.87,.9])
for table in cfg['tables']:
 name=table['name'];x,y=table['center_xy_m'];lx,ly,z=table['size_m'];th=cfg['tabletop_thickness_m'];leg=cfg['leg_width_m'];UsdGeom.Xform.Define(s,'/World/Environment/'+name)
 box(name+'/Top',[x,y,z-th/2],[lx,ly,th],[.58,.34,.16] if name=='SourceTable' else [.28,.47,.54])
 for i,(dx,dy) in enumerate([(u,v) for u in [-1,1] for v in [-1,1]]):box(name+f'/Leg{i}',[x+dx*(lx/2-.08),y+dy*(ly/2-.08),(z-th)/2],[leg,leg,z-th],[.16,.18,.21])
tray=cfg.get('tray')
if tray:
 tx,ty=tray['center_xy_m'];sx,sy=tray['outer_size_m'];th=tray['base_thickness_m'];rim=tray['rim_height_m'];rt=tray['rim_thickness_m']
 top=next(table['size_m'][2] for table in cfg['tables'] if table['name']==tray['table_name'])
 name=tray['name'];UsdGeom.Xform.Define(s,'/World/Environment/'+name)
 box(name+'/Base',[tx,ty,top+th/2],[sx,sy,th],[.2,.55,.3])
 for label,c,d in [('Left',[tx-sx/2+rt/2,ty,top+th+rim/2],[rt,sy,rim]),('Right',[tx+sx/2-rt/2,ty,top+th+rim/2],[rt,sy,rim]),('Front',[tx,ty-sy/2+rt/2,top+th+rim/2],[sx-2*rt,rt,rim]),('Back',[tx,ty+sy/2-rt/2,top+th+rim/2],[sx-2*rt,rt,rim])]:
  box(name+'/'+label,c,d,[.15,.4,.22])
cfgcup=cfg['cup'];cup=UsdGeom.Cylinder.Define(s,'/World/Environment/Cup');cup.CreateRadiusAttr(cfgcup['radius_m']);cup.CreateHeightAttr(cfgcup['height_m']);cup.CreateAxisAttr('Z');cup.CreateDisplayColorAttr([Gf.Vec3f(.9,.22,.08)]);UsdGeom.Xformable(cup).AddTranslateOp().Set(Gf.Vec3d(*cfgcup['initial_center_m']));UsdPhysics.CollisionAPI.Apply(cup.GetPrim());UsdPhysics.RigidBodyAPI.Apply(cup.GetPrim());UsdPhysics.MassAPI.Apply(cup.GetPrim()).CreateMassAttr(cfgcup['mass_kg']);UsdShade.MaterialBindingAPI.Apply(cup.GetPrim()).Bind(mat,materialPurpose='physics')
p=UsdGeom.Cylinder.Define(s,'/World/Environment/PlaceTarget');p.CreateRadiusAttr(.05);p.CreateHeightAttr(.002);p.CreateDisplayColorAttr([Gf.Vec3f(.12,.85,.28)]);xyz=cfg['placement_center_m'];UsdGeom.Xformable(p).AddTranslateOp().Set(Gf.Vec3d(xyz[0],xyz[1],xyz[2]-cfgcup['height_m']/2+.001));p.GetPrim().SetCustomDataByKey('task:visualMarkerOnly',True)
light=UsdLux.DomeLight.Define(s,'/World/DomeLight');light.CreateIntensityAttr(700)
sun=UsdLux.DistantLight.Define(s,'/World/Sun');sun.CreateIntensityAttr(1800);UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(30,-35,-20))
cam=UsdGeom.Camera.Define(s,'/World/OverviewCamera');eye=Gf.Vec3d(7,-8,9);target=Gf.Vec3d(0,.6,.5);view=Gf.Matrix4d().SetLookAt(eye,target,Gf.Vec3d(0,0,1));UsdGeom.Xformable(cam).AddTransformOp().Set(view.GetInverse());cam.CreateClippingRangeAttr(Gf.Vec2f(.05,100));cam.CreateFocalLengthAttr(24)
s.GetRootLayer().Save()
# Robot composition stays separate: environment testing does not alter shared robot assets.
full=Usd.Stage.CreateNew(str(assets/'rm75_table_transfer.usda'));full.GetRootLayer().subLayerPaths=['environment.usda'];full.SetDefaultPrim(full.GetPrimAtPath('/World'));UsdGeom.SetStageMetersPerUnit(full,1);UsdGeom.SetStageUpAxis(full,'Z')
robot=UsdGeom.Xform.Define(full,'/World/Robot');robot.GetPrim().GetReferences().AddReference(cfg['robot']['usd']);bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy]).ComputeWorldBound(robot.GetPrim()).ComputeAlignedRange();ground_offset=-float(bounds.GetMin()[2])+.002
xform=UsdGeom.Xformable(robot);xform.AddTranslateOp().Set(Gf.Vec3d(*cfg['robot']['spawn_xy_m'],ground_offset));xform.AddRotateZOp().Set(cfg['robot']['yaw_deg']);full.GetRootLayer().Save()
# Canonical static-world obstacles; moving cup pose must be refreshed from simulation.
import yaml
(ROOT/'static_world.yml').write_text(yaml.safe_dump({'cuboid':boxes},sort_keys=False))
manifest={'environment_usd':str(assets/'environment.usda'),'robot_scene_usd':str(assets/'rm75_table_transfer.usda'),'robot_root_z_m':ground_offset,'table_top_z_m':.75,'static_box_count':len(boxes),'cup_prim':'/World/Environment/Cup','target_center_m':cfg['placement_center_m'],'note':'Robot scene opens paused. Right-arm reachability and base orientation still require planning validation. Cup ground truth must be read at runtime after settling.'}
(ROOT/'scene_manifest.json').write_text(json.dumps(manifest,indent=2));print('SCENE_BUILT',json.dumps(manifest),flush=True)
app.close()
