# RM75 + RMG24 migration for Isaac Sim and cuRobo

Target environment:

- Isaac Sim `6.0.1-rc.7`
- cuRobo `0.8.0.post1.dev41` (config format 2.0)
- Fixed mobile base; left and right RM75 arms are the 14 active planning joints
- Lift, head and grippers are locked in the initial cuRobo configuration

## Files

- `assets/urdf/rm75_dual_rmg24_sim.urdf`: Isaac Sim import model.
- `assets/urdf/rm75_dual_rmg24_curobo.urdf`: planning model with wheel joints fixed.
- `config/rm75_generated.yml`: reproducible RobotBuilder output (313 spheres).
- `config/rm75_dual.yml`: final 14-DOF cuRobo configuration.
- `scripts/import_urdf.py`: Isaac Sim 6 URDF-to-USD importer.
- `scripts/validate_config.py`: cuRobo configuration validation.
- `assets/usd/obstacles/simple_obstacles.usda`: standalone static obstacle asset.
- `assets/usd/scenes/rm75_simple_obstacles.usda`: scene that references the robot and obstacles.
- `config/simple_obstacles.yml`: matching cuRobo cuboid collision world.
- `scripts/create_simple_obstacle_scene.py`: regenerates both obstacle representations.

The source RM75 RMG24 model incorrectly referenced the RM65 package for 12 gripper mesh
entries. Both migrated URDFs correct these references to the local `assets/meshes` directory.

## Validate cuRobo

```bash
/home/fang/isaacsim/python.sh scripts/validate_config.py
```

## Generate USD

```bash
/home/fang/isaacsim/python.sh scripts/import_urdf.py
```

The generated USD is written below `assets/usd`. The importer deliberately keeps fixed joints
unmerged so link names remain aligned with cuRobo.

## Generate the obstacle scene

```bash
/home/fang/isaacsim/python.sh scripts/create_simple_obstacle_scene.py
```

Open `assets/usd/scenes/rm75_simple_obstacles.usda` to see the composed scene. The robot and
obstacles remain separate referenced assets, so regenerating obstacles cannot overwrite the robot.
Obstacle poses use the old MoveIt `base_link_underpan` frame; this is equivalent to world
coordinates only while the composed robot is placed at the world origin.

## Launch the complete scene in the Isaac Sim GUI

```bash
cd /home/fang/isaacsim_rm75
/home/fang/isaacsim/python.sh scripts/launch_scene.py
```

This opens the composed robot-and-obstacles Stage directly. In the viewport, select
`/World/Robot` or `/World/Obstacles/Obstacles/<name>` and press `F` to focus it. Stop the
application with `Ctrl+C` in the terminal after closing the GUI.

The launcher also creates two Robot Poser named poses:

- `left_arm_poser`: `l_base_link1` to the `l_link7` flange-center frame `l_tool0`.
- `right_arm_poser`: `r_base_link1` to the `r_link7` flange-center frame `r_tool0`.

The timeline starts automatically. Robot Poser `Track Target` remains available for direct IK
pose editing. For collision-free planning, leave `Track Target` disabled and drag the named-pose
target prim in the viewport so the robot itself stays at the trajectory start state.

The RGB axes parented to `l_tool0` and `r_tool0` show the current TCP and always move with the
grippers. The draggable Robot Poser axes show desired targets, so they intentionally remain
independent after being moved.

When either RM75 Robot Poser target is selected, its viewport gizmo shows the three translation
arrows and three rotation rings together. `W` and `E` remain available as standard Isaac Sim
shortcuts, but switching between them is not required for positioning these targets.

Dragging either target also runs Isaac Robot Poser IK in real time. The cyan translucent
`/World/IKPreview` robot shows the resulting arm configuration without moving the simulated robot
or participating in collision checking.

The `RM75 Motion Planner` window provides three planning commands:

- `left`: plan the left arm to `left_arm_poser`; the right arm is locked at its measured state.
- `right`: plan the right arm to `right_arm_poser`; the left arm is locked at its measured state.
- `dual`: jointly plan all 14 arm joints to both Robot Poser targets.

Planning only creates a collision-free trajectory. Use `Execute` to send the latest trajectory to
the simulated articulation, or `Stop` to interrupt it. The static cuboids from
`config/simple_obstacles.yml`, the robot body, and both arms participate in collision checking.

## Important calibration items

- `l_tool0` and `r_tool0` are centered on the RM75 link7 flanges at the existing gripper-mount
  offset of `0.0445 m`; they are intentionally not grasp-center frames.
- Acceleration (`3 rad/s^2`) and jerk (`30 rad/s^3`) limits are conservative placeholders.
- Collision spheres were generated automatically at density `0.7`; inspect them visually before
  operating near the body or coordinating both arms.
- The planning configuration locks the lift at `0.5 m` and both grippers at `0.0325 m`.

## RViz2 Interactive Marker + cuRobo workflow

This is the preferred replacement for the earlier Isaac Robot Poser / MoveIt-like UI.
RViz2 owns the 6-DoF target markers and Isaac Sim owns cuRobo planning/execution.

Current transport note: on this workstation, Isaac Sim 6's internal Humble `rclpy`
imports and initializes, but crashes in `rcl_interfaces` when constructing a Python
`Node`. To keep the workflow usable, the ROS2/RViz process exposes the ROS2 topics and
services, then forwards commands to Isaac through a localhost JSON socket
(`127.0.0.1:50075`). The RViz-facing interface remains ROS2-native and can later be
swapped back to an Isaac ROS2 Bridge transport.

### Terminal 1: Isaac/cuRobo backend

```bash
cd /home/fang/isaacsim_rm75
/home/fang/isaacsim/python.sh scripts/launch_ros2_curobo_scene.py
```

Use `--headless` for a backend-only run:

```bash
/home/fang/isaacsim/python.sh scripts/launch_ros2_curobo_scene.py --headless
```

The backend opens `assets/usd/scenes/rm75_simple_obstacles.usda`, initializes the robot
from `config/rm75_dual.yml`, starts cuRobo planners on demand, and listens on
`127.0.0.1:50075`.

### Terminal 2: RViz2 frontend

```bash
cd /home/fang/isaacsim_rm75
./scripts/start_rviz_frontend.sh
```

This starts:

- `robot_state_publisher` with an RViz-friendly generated URDF at `build/rm75_rviz.urdf`.
- `scripts/rm75_rviz_interactive_markers.py`.
- `rviz2` with `rviz/rm75_interactive_markers.rviz`.

In RViz, use the **Interact** tool and drag:

- `left_goal`
- `right_goal`

Right-click either marker for:

- `Plan left`
- `Plan right`
- `Plan dual`
- `Execute latest`
- `Reset goals from Isaac`

### ROS2 interface exposed by the RViz frontend

Topics:

- `/rm75/left_goal` (`geometry_msgs/PoseStamped`, frame `base_link_underpan`)
- `/rm75/right_goal` (`geometry_msgs/PoseStamped`, frame `base_link_underpan`)
- `/rm75/left_goal_state`, `/rm75/right_goal_state`
- `/joint_states`
- `/rm75/planned_trajectory` (`trajectory_msgs/JointTrajectory`)
- `/rm75/status` (`std_msgs/String`)

Services:

```bash
ros2 service call /rm75/plan_left std_srvs/srv/Trigger {}
ros2 service call /rm75/plan_right std_srvs/srv/Trigger {}
ros2 service call /rm75/plan_dual std_srvs/srv/Trigger {}
ros2 service call /rm75/execute std_srvs/srv/Trigger {}
ros2 service call /rm75/reset_goals std_srvs/srv/Trigger {}
```

All target poses are interpreted in the cuRobo base frame `base_link_underpan`. ROS
quaternions are `xyzw`; the backend converts them to cuRobo `wxyz` internally.

### Pose IK contract and failure diagnosis

The successful path uses this exact IK sequence when `Plan` is clicked:

1. `scripts/rm75_planner.py` reads the current Isaac articulation and builds
   `q_start` in the fixed order `l_joint1..l_joint7, r_joint1..r_joint7`.
2. In single-arm mode, the other arm remains in the collision model and is
   locked at its measured current joints. In dual-arm mode all 14 arm joints
   are active. Lift, head and gripper joints remain locked to the values in
   `config/rm75_dual.yml`.
3. The bridge sends only the workspace target pose, not MoveIt's Goal State
   joint values. The target is `l_tool0` or `r_tool0`, expressed in
   `base_link_underpan`. `link7` targets are converted to `tool0` by the fixed
   44.5 mm flange offset, exactly once.
4. The socket backend converts ROS `xyzw` to cuRobo `wxyz`, then calls
   `MotionPlanner.plan_pose()`. cuRobo runs multi-seed collision-aware IK and
   TrajOpt; `q_start` is the initial state and IK seed, not the joint goal.

For a server-side IK failure, compare the backend log in this order:

- `q_start`: names, order, radians, and whether it is the measured robot state;
- `X_goal`: `base_link_underpan`, `l_tool0/r_tool0`, and `xyzw -> wxyz` conversion;
- `FK(q_start)`: this must be in the same base/tool convention as `X_goal`;
- `target delta from current FK`: a large delta usually indicates a frame,
  quaternion, or link7/tool0 offset error;
- `cuRobo IK n/4`, `position_error`, `rotation_error`, and `feasible`: this
  distinguishes geometric IK failure from collision/joint-limit rejection.

A decisive closed-loop test is to take a known-valid `q_valid`, compute
`FK(q_valid)` with the same cuRobo model, and feed that pose back into IK. If
that fails, the server's URDF/configuration or collision model is wrong. If it
passes but the real target fails, the real target's frame, quaternion order,
tool frame, or start state is wrong.

## One-command RViz2 front-end + headless Isaac/cuRobo backend

The recommended interactive workflow is now RViz2 as the visible front-end and Isaac Sim as a
headless cuRobo/trajectory backend:

```bash
cd /home/fang/isaacsim_rm75
./scripts/start_headless_isaac_rviz.sh
```

This starts, in order:

1. headless Isaac Sim with the RM75 scene and cuRobo backend;
2. `robot_state_publisher` for the RViz robot model;
3. the RM75 RViz2 Interactive Marker node;
4. RViz2 with the saved RM75 display config.

In RViz2, select the **Interact** tool, drag the blue/orange goal balls, and right-click a goal
ball for `Plan left`, `Plan right`, `Plan dual`, `Execute latest`, or `Reset goals from Isaac`.
The robot is visualized from `/joint_states`; static cuboid obstacles are published on
`/rm75/obstacles` from `config/simple_obstacles.yml`.

To display the actual cuRobo collision-sphere model in RViz, enable the optional
read-only sphere publisher when starting the stack:

```bash
RM75_SHOW_CUROBO_SPHERES=1 ./scripts/start_headless_isaac_moveit2.sh
```

The spheres are published on `/rm75/curobo_collision_spheres` as a `MarkerArray` in
`base_link_underpan`. Their positions are computed by cuRobo FK from the live Isaac
joint state, so this display uses the same sphere model as collision checking. The
first request may take longer because it initializes the dual-arm cuRobo planner.
Blue spheres belong to the left arm, orange spheres to the right arm, and gray/purple
spheres belong to the base, lift, head, and other locked links. The RViz display is
already present in `moveit.rviz` and can also be toggled manually under Displays.

Logs are written to `logs/isaac_curobo_backend.log`, `logs/robot_state_publisher.log`, and
`logs/rviz_interactive_markers.log`.

## Stability / SAFE_MODE ladder

The one-command launcher supports staged bring-up so problems can be isolated before enabling
planning and execution:

```bash
RM75_SAFE_MODE=A ./scripts/start_headless_isaac_rviz.sh
RM75_SAFE_MODE=B ./scripts/start_headless_isaac_rviz.sh
RM75_SAFE_MODE=C ./scripts/start_headless_isaac_rviz.sh
RM75_SAFE_MODE=D ./scripts/start_headless_isaac_rviz.sh
RM75_SAFE_MODE=F ./scripts/start_headless_isaac_rviz.sh
RM75_SAFE_MODE=G ./scripts/start_headless_isaac_rviz.sh
```

Levels:

- `A`: RViz + static robot model only. Isaac is not started.
- `B`: add headless Isaac backend and bridge only real `q_actual` to `/joint_states`.
- `C`: add RViz Interactive Markers for left/right goals. Dragging goals does **not** plan.
- `D`: add live IK preview while dragging goals. RViz publishes a visual-only
  `/rm75/ik_preview` `MarkerArray` skeleton; it does **not** publish ghost TF, ghost
  `/joint_states`, planning commands, execution commands, or real articulation targets.
- `E`: backend/planner initialization level; planning services remain disabled.
- `F`: enable explicit `Plan left/right/dual`; one click creates at most one cuRobo plan.
- `G`: enable `Execute latest` after a successful plan. This is the normal full mode.

Stability rules implemented in the current transport:

- `/joint_states` is published only by the RViz socket adapter and represents real Isaac
  articulation state, not goals, ghost state, or planned state.
- RViz TF display is disabled by default to avoid the red/green/blue axis cloud. Robot TF should
  come from exactly one `robot_state_publisher`.
- In `SAFE_MODE=C`, dragging an Interactive Marker only updates a latest-value goal buffer.
  In `SAFE_MODE=D` and above, dragging additionally requests low-rate Isaac Robot Poser IK for
  visual preview only. Dragging still never triggers cuRobo planning, execution, or real robot
  motion.
- Socket `set_goal` uses latest-value semantics. Old unconsumed goal updates are overwritten.
- Expensive commands (`plan`, `execute`, `reset`) go through a bounded main-thread command queue.
- Network threads never touch USD, Physics, Articulation, or cuRobo directly.
- IK preview commands are queued onto the Isaac main thread and throttled on the RViz side
  (`RM75_IK_PREVIEW_HZ`, default `8 Hz`) with latest-target semantics.
- Planning has request IDs and a state gate (`IDLE/GOAL_READY/PLANNING/PLAN_READY/EXECUTING`).
- Reset stops execution and clears the previous trajectory.

Backend diagnostics are printed at most once per second in `logs/isaac_curobo_backend.log`, e.g.:

```text
[DIAG] state=IDLE socket_rx_rate=... goal_update_rate=... joint_snapshot_rate=...
       plan_count=... queue_size=... threads=... rss=...GB
```

Use the ROS graph diagnostic helper after the system is running:

```bash
./scripts/diagnose_ros2_graph.sh
```

Key acceptance checks:

```bash
ros2 topic info /joint_states -v   # Publisher count should be 1
ros2 topic info /tf -v             # robot TF should have one authority
ros2 topic hz /joint_states        # should be stable, not thousands of Hz
ros2 topic hz /tf                  # should be stable, not a storm
```

### Backend safety guards

The launcher now protects against accidentally starting multiple Isaac Sim backends, which can
exhaust GPU/VRAM or trigger driver hangs:

- If `127.0.0.1:50075` already serves an RM75 backend, the launcher reuses it instead of starting
  another Isaac process.
- If the backend port is occupied by something else, startup is refused.
- If a stale `launch_ros2_curobo_scene.py` process is detected but the socket is unhealthy, startup
  is refused. Stop the stale process first.
- The backend script itself also takes an exclusive file lock at `/tmp/rm75_isaac_backend.lock`, so
  direct manual launches are protected too.

Only override this guard if you are certain no backend is running:

```bash
RM75_FORCE_START_ISAAC=1 ./scripts/start_headless_isaac_rviz.sh
```

`SAFE_MODE=B/C/D/E` use a lightweight disabled planner facade; the full cuRobo planner wrapper is
created only in `SAFE_MODE=F/G`. Locked non-planning joints such as `head_joint1/head_joint2` are
clamped in published `/joint_states` so passive Isaac articulation wobble does not make RViz appear
to shake its head. Optional low-rate target hold can be disabled with:

```bash
RM75_LOCKED_TARGET_HOLD=0 RM75_SAFE_MODE=B ./scripts/start_headless_isaac_rviz.sh
```

### Joint drive gravity-hold tuning

If the arms sag or oscillate in Isaac/RViz while no planning or execution is active, the imported
USD drive gains are probably too soft. The generated USD originally contains angular drive gains
around `stiffness=6.98`, `damping=0.70`, with some wrist `maxForce=10`, which is not enough to hold
the RM75 arms against gravity.

`launch_ros2_curobo_scene.py` now patches the composed stage at startup before physics playback:

- `l_joint1..l_joint7` and `r_joint1..r_joint7` default to
  `stiffness=1000`, `damping=100`, `maxForce=1000`.
- `head_joint1/head_joint2` default to `stiffness=200`, `damping=20`, `maxForce=50`.
- `platform_joint` default to `stiffness=1500`, `damping=200`, `maxForce=500`.

Tune without editing USD:

```bash
RM75_SAFE_MODE=B \
RM75_ARM_DRIVE_STIFFNESS=1000 \
RM75_ARM_DRIVE_DAMPING=100 \
RM75_ARM_DRIVE_MAX_FORCE=1000 \
./scripts/start_headless_isaac_rviz.sh
```

If the arm still sags, increase `RM75_ARM_DRIVE_MAX_FORCE` or `RM75_ARM_DRIVE_STIFFNESS`. If it
jitters/oscillates, increase `RM75_ARM_DRIVE_DAMPING` or reduce stiffness.

Because the launcher reuses an already running backend, restart the backend to pick up gain changes:

```bash
./scripts/stop_rm75_stack.sh
RM75_SAFE_MODE=B ./scripts/start_headless_isaac_rviz.sh
```

## MoveIt1 package migration to MoveIt2 frontend

The closest ROS1/MoveIt1 package in `/home/fang/catkin_wsfrj` is:

```text
/home/fang/catkin_wsfrj/src/dual_arm_control/dual_arm_moveit/dual_75B_arm_rmg24_moveit_config
```

It has been migrated into a ROS2/MoveIt2-style config package at:

```text
moveit2/rm75_dual_moveit_config
```

Important migration choices:

- Reused the old MoveIt1 self-collision matrix from
  `overseas_75_b_v_description.srdf`.
- Used the current Isaac/RViz URDF `assets/urdf/rm75_dual_rmg24_sim.urdf`, because it adds
  the useful TCP links `l_tool0` and `r_tool0`.
- Updated arm planning chains:
  - `l_arm`: `l_base_link1 -> l_tool0`
  - `r_arm`: `r_base_link1 -> r_tool0`
- Added `dual_arm` as a subgroup of `l_arm + r_arm`.
- Kept KDL IK for individual arm groups.
- MoveIt2 execution is disabled by default; MoveIt2 is currently intended as the RViz
  MotionPlanning frontend/ghost-state UI, while Isaac/cuRobo remains the backend path.

Launch only the MoveIt2 frontend config, without Isaac:

```bash
source /opt/ros/humble/setup.bash
ros2 launch moveit2/rm75_dual_moveit_config/launch/moveit_frontend.launch.py
```

Launch headless Isaac for real `/joint_states` plus the MoveIt2 RViz frontend:

```bash
./scripts/stop_rm75_stack.sh
./scripts/start_headless_isaac_moveit2.sh
```

This starts Isaac/cuRobo in `SAFE_MODE=F`, bridges the real Isaac articulation state to
`/joint_states`, and keeps the draggable goal ghost in MoveIt2's native RViz MotionPlanning
plugin rather than the legacy custom Interactive Marker node.

### Shared obstacle scene

`config/simple_obstacles.yml` is the only obstacle geometry input for this workflow:

- Isaac authors the YAML cuboids into the live USD stage as visible Physics Collision objects.
- cuRobo receives the same validated cuboids as its collision world.
- `scripts/rm75_moveit_obstacles.py` applies them to MoveIt's native PlanningScene, which RViz
  displays from `/curobo/monitored_planning_scene`.

The adapter also publishes the canonical scene on `/curobo/obstacle_scene`. Every complete stack
startup compares a deterministic scene fingerprint reported by Isaac with the local YAML before
MoveIt is allowed to accept the objects. Edit the YAML and restart the complete stack; partial
hot reload is deliberately unsupported because it could leave Isaac/cuRobo and MoveIt on different
scene versions.

Verify the MoveIt copy with:

```bash
ros2 service call /curobo/get_planning_scene moveit_msgs/srv/GetPlanningScene \
  "{components: {components: 16}}"
```
