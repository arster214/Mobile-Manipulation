from pathlib import Path
import json,shutil,yaml
R=Path('/home/fang/robot_ws');p=Path('/home/fang/isaacsim_rm75/assets/urdf/rm75_dual_rmg24_curobo.urdf')
backup=p.with_suffix('.urdf.before_grasp_tcp')
if not backup.exists():shutil.copy2(p,backup)
s=p.read_text()
if 'name="l_grasp_tcp"' not in s:
 blocks=[]
 for side in ['l','r']:
  blocks.append(f'''    <!-- Simulation grasp point: finger pad depth center 0.0891 + (0.0059+0.0495)/2 m.
         Orientation is identical to tool0; not a physical TCP calibration. -->
    <link name="{side}_grasp_tcp" />
    <joint name="{side}_grasp_tcp_joint" type="fixed">
        <origin xyz="0 0 0.1168" rpy="0 0 0" />
        <parent link="{side}_tool0" />
        <child link="{side}_grasp_tcp" />
    </joint>''')
 p.write_text(s.replace('</robot>','\n'.join(blocks)+'\n</robot>'))
p=R/'postures/rm75_mobile_manipulation.yml';b=p.with_suffix('.yml.before_grasp_tcp')
if not b.exists():shutil.copy2(p,b)
c=yaml.safe_load(p.read_text());c['kinematics']['tool_frames']=['l_grasp_tcp','r_grasp_tcp','l_tool0','r_tool0'];p.write_text(yaml.safe_dump(c,sort_keys=False))
spec={'status':'simulation_mesh_based_not_hardware_calibrated','definition':'midpoint between symmetric finger pad centers','finger_pad_z_range_in_finger_m':[.0059,.0495],'finger_joint_z_in_gripper_m':.0891,'frames':{side+'_grasp_tcp':{'parent':side+'_tool0','xyz_m':[0,0,.1168],'rpy_rad':[0,0,0]} for side in ['l','r']},'axes':'Same as tool0; confirmed initial posture X down, Y horizontal, Z forward. Finger closing axis is approximately tool Y.','candidate_frame_conversion':'For existing abstract candidates (X closing, Y up, Z approach), R_world_tcp = R_world_candidate @ Rz(-pi/2); candidate position is the grasp_tcp position.'}
(R/'postures/grasp_tcp_definition.json').write_text(json.dumps(spec,indent=2))
p=R/'postures/mobile_manipulation_initial.json';d=json.loads(p.read_text());d['grasp_tcp_definition']=str(R/'postures/grasp_tcp_definition.json');d['planning_tool_frames']=['l_grasp_tcp','r_grasp_tcp'];p.write_text(json.dumps(d,indent=2))
p=R/'scenarios/table_transfer/scene_config.json';d=json.loads(p.read_text());d['robot']['grasp_tcp_frames']={'left':'l_grasp_tcp','right':'r_grasp_tcp'};d['robot']['grasp_tcp_definition']=str(R/'postures/grasp_tcp_definition.json');p.write_text(json.dumps(d,indent=2))
p=R/'scenarios/table_transfer/grasp_policy.json';d=json.loads(p.read_text());d['robot_grasp_tcp']={'frame':'r_grasp_tcp','definition':str(R/'postures/grasp_tcp_definition.json'),'candidate_to_tcp_rotation_local_z_rad':-1.5707963267948966,'status':'simulation_transform_defined; existing candidates retain abstract axes; apply rotation before IK'};p.write_text(json.dumps(d,indent=2))
