"""Persist approved simulation TCP frames in source URDFs, cuRobo and USD."""
from pathlib import Path
import shutil,json,xml.etree.ElementTree as ET
from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
try:
 from pxr import Usd,UsdGeom,Gf
 import yaml
 R=Path('/home/fang/robot_ws');A=Path('/home/fang/isaacsim_rm75');spec=json.loads((R/'postures/grasp_tcp_definition.json').read_text())
 source=ET.parse(A/'assets/urdf/rm75_dual_rmg24_curobo.urdf').getroot()
 p=A/'assets/urdf/rm75_dual_rmg24_sim.urdf';text=p.read_text()
 if 'name="l_grasp_tcp"' not in text:
  shutil.copy2(p,p.with_suffix('.urdf.before_grasp_tcp'))
  nodes=[e for e in source if e.get('name') in ['l_grasp_tcp','r_grasp_tcp','l_grasp_tcp_joint','r_grasp_tcp_joint']]
  text=text.replace('</robot>','\n'.join(ET.tostring(e,encoding='unicode') for e in nodes)+'\n</robot>');p.write_text(text)
 # Preserve existing frame ordering for older callers; append new frames.
 p=A/'config/rm75_dual.yml';b=p.with_suffix('.yml.before_grasp_tcp')
 if not b.exists():shutil.copy2(p,b)
 c=yaml.safe_load(p.read_text());k=c.get('kinematics',c.get('robot_cfg',{}).get('kinematics',{}))
 assert k,'Unknown robot config schema'
 for name in spec['frames']:
  if name not in k['tool_frames']:k['tool_frames'].append(name)
 p.write_text(yaml.safe_dump(c,sort_keys=False))
 p=A/'assets/usd/rm75_dual_rmg24_sim_1/rm75_dual_rmg24_sim.usda';b=p.with_suffix('.usda.before_grasp_tcp')
 if not b.exists():shutil.copy2(p,b)
 stage=Usd.Stage.Open(str(p));paths={}
 for side in ['l','r']:
  links=[x for x in stage.Traverse() if x.GetName()==side+'_tool0' and x.GetTypeName()=='Xform'];assert len(links)==1,[str(x.GetPath()) for x in links]
  path=str(links[0].GetPath())+'/'+side+'_grasp_tcp';frame=UsdGeom.Xform.Define(stage,path)
  xf=UsdGeom.Xformable(frame);xf.ClearXformOpOrder();xf.AddTranslateOp().Set(Gf.Vec3d(0,0,.1168))
  frame.GetPrim().SetCustomDataByKey('tcp_definition','Simulation finger pad center; tool0 +Z 0.1168 m; same axes; not hardware calibrated')
  paths[side+'_grasp_tcp']=path
 stage.GetRootLayer().Save()
 # Reopen to verify authored data was saved and is composed into the experiment.
 stage=Usd.Stage.Open(str(p))
 for name,path in paths.items():
  m=UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation();assert (m.ExtractTranslation()-Gf.Vec3d(0,0,.1168)).GetLength()<1e-9
 scene=Usd.Stage.Open(str(R/'scenarios/table_transfer/assets/rm75_table_transfer.usda'))
 for name in paths:assert any(x.GetName()==name for x in scene.Traverse()),name
 report={'usd_asset':str(p),'usd_prim_paths':paths,'offset_from_link7_m':[0,0,.1613],'offset_from_tool0_m':[0,0,.1168],'source_urdfs':['rm75_dual_rmg24_sim.urdf','rm75_dual_rmg24_curobo.urdf'],'scene_composition_verified':True}
 (R/'postures/grasp_tcp_model_sync.json').write_text(json.dumps(report,indent=2));print('TCP_MODEL_SYNC_OK',report,flush=True)
except Exception:
 import traceback
 traceback.print_exc()
finally:app.close()
