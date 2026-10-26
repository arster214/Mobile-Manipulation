GPU whole-body IK collision experiment

Viewer: http://127.0.0.1:8092/
Blue transparent = start; green = accepted endpoint; red = failed candidate. Dropdown selects pairs 1-20. Collision sphere checkbox overlays the saved model.

11 DoF: base X/Y/yaw + lift + right arm; left arm fixed at saved joint angles.
20 original random starts reused and independently rechecked with the new sphere model.
cuRobo V2 on RTX 5080, 32 parallel seeds per query, 20 sequential queries, CUDA graphs enabled.
Includes self collision and static world obstacles; bottle uses conservative AABB. Fixed floor support is excluded from obstacle checks.
Position tolerance 0.1 mm, orientation tolerance 0.001 rad. Timing includes synchronized solve call, excludes setup/warmup and independent audit.
Pinocchio independent FK and sphere-distance checks: 16 accepted, 4 failed (11,13,15,20). Some accepted sphere gaps are very small; no hard positive safety margin is required.
Only endpoints validated. No nonholonomic path, swept collision, grasp execution or dynamics validation. Base angular velocity is a pose-model placeholder, not a measured hardware limit.

Reproduce:
/home/fang/isaacsim/python.sh /home/fang/robot_ws/scenarios/table_transfer/scripts/benchmark_gpu_collision_ik.py
/home/fang/robot_ws/pin_pink/python.sh /home/fang/robot_ws/scenarios/table_transfer/scripts/audit_gpu_ik_pairs.py
/home/fang/isaacsim/python.sh /home/fang/robot_ws/scenarios/table_transfer/scripts/view_ik_pairs.py

source_robot.yml and source_sha256 in result.json preserve collision model provenance.
