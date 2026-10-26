Pinocchio + Pink offline pregrasp IK

Run solver:
/home/fang/robot_ws/pin_pink/python.sh /home/fang/robot_ws/scenarios/table_transfer/scripts/solve_pregrasp_pink.py

Run preview:
/home/fang/isaacsim/python.sh /home/fang/robot_ws/scenarios/table_transfer/scripts/view_pregrasp_ik.py
http://127.0.0.1:8091/

Uses saved initial posture and URDF r_grasp_tcp (tool0 Z offset 0.1168 m).
Bottle proxy: existing cylinder, radius 0.03 m, height 0.12 m, center [-1.05, 1.32, 0.81] m.
Bottle placement is isolated to this offline test/preview, not written into the running Isaac stage.
Base remains at scene spawn pose; left-arm joint angles stay fixed.
7-DoF right arm and 8-DoF right arm + lift tested from the same saved initial posture.
All five height candidates converged for both modes and satisfy tested joint limits.
Selected result: right arm + lift, middle height candidate.
This is kinematic reachability only, not collision validation, a feasible execution trajectory, or proof of stable grasp.
Pink iteration time step is numerical integration time, not a commanded execution schedule.
Lift [0, 0.79] m is a provisional interval pending calibration; TCP is mesh-defined for simulation.
