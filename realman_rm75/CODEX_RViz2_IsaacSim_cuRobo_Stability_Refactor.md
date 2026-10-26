# Codex 修改任务：RViz2 + Isaac Sim + cuRobo 稳定性排查与架构重构

## 0. 当前环境与已知问题

当前项目环境：

- Ubuntu 22.04
- ROS2 Humble
- RViz2
- Isaac Sim 6.0.1
- cuRobo
- Isaac Sim 内部 `rclpy` 当前不可直接使用：
  - 在创建 ROS2 Node 时会触发 `rcl_interfaces` C 扩展断言
  - 随后发生 `Segmentation Fault`
- 因此当前通信方案保持 RViz 侧 ROS2 接口不变，但 Isaac 后端暂时使用：
  - `127.0.0.1:50075`
  - TCP JSON socket
  - 作为 transport layer
- 后续 internal `rclpy` 修复后，只替换 transport，不修改上层业务逻辑

当前存在的问题：

1. RViz 启动后机器人周围出现大量红/绿/蓝 TF 坐标轴。
2. RobotModel、TF 坐标轴或机器人姿态持续抽搐。
3. 运行一段时间后系统负载异常，严重时整台电脑卡死或假死。
4. 当前系统同时包含：
   - 真实机器人状态
   - RViz Interactive Marker Goal
   - Ghost / 目标虚像
   - Socket 通信
   - Isaac Sim Articulation
   - cuRobo Planner
5. 怀疑存在以下问题中的一个或多个：
   - 重复 `/joint_states`
   - 重复 TF 发布
   - Ghost 和真实机器人 frame 名冲突
   - 消息反馈环
   - Interactive Marker 高频更新
   - 拖动 Goal 时重复规划
   - Socket busy loop
   - 网络线程直接操作 Isaac USD / Articulation
   - callback / timer / thread 重复注册
   - cuRobo 重复初始化或高频 planning
   - 内存 / VRAM 泄漏
   - GPU 资源竞争导致 NVIDIA driver hang

---

# 1. 总体修改目标

不要继续增加新功能。

优先完成：

> **诊断 → 隔离 → 稳定 → 再恢复功能**

必须保证以下五个模块彻底解耦：

```text
Goal Editing
    ↓
Goal Pose

Robot State
    ↓
q_actual

Planning
    ↓
q_plan(t)

Execution
    ↓
q_command(t)

Ghost Visualization
    ↓
纯可视化
```

严格禁止 Goal 编辑直接修改真实机器人状态。

---

# 2. 第一阶段：检查 ROS2 是否存在消息冲突或消息风暴

启动系统后执行并记录：

```bash
ros2 node list

ros2 topic info /joint_states -v
ros2 topic info /tf -v
ros2 topic info /tf_static -v

ros2 topic hz /joint_states
ros2 topic hz /tf

ros2 topic bw /joint_states
ros2 topic bw /tf
```

## 2.1 `/joint_states` 必须只有一个真实状态源

目标：

```text
Publisher count: 1
```

唯一推荐数据流：

```text
Isaac Sim Articulation
        ↓
q_actual
        ↓
JSON socket
        ↓
ROS2 adapter
        ↓
/joint_states
```

检查并关闭任何额外发布者，例如：

```text
joint_state_publisher
joint_state_publisher_gui
ghost_joint_state_publisher
测试 publisher
第二个 Isaac state publisher
```

必须保证：

```text
/joint_states
只描述真实机器人 q_actual
```

绝对禁止把这些状态发布到 `/joint_states`：

```text
q_goal
q_ghost
q_plan
```

特别检查 launch 文件中是否同时启动：

```text
joint_state_publisher
+
isaac_joint_state_adapter
```

如果两个节点都发布同一批 joint names，会导致 RobotModel 在两套状态之间来回跳动。

---

# 3. 第二阶段：检查 TF authority 是否重复

执行：

```bash
ros2 topic info /tf -v
ros2 topic info /tf_static -v

ros2 node list | grep -E 'robot_state|joint_state|tf'
```

重点检查是否存在多个：

```text
robot_state_publisher
ghost_robot_state_publisher
isaac_tf_publisher
static_transform_publisher
```

同时发布相同 frame。

## 推荐结构

```text
Isaac Sim
只输出 q_actual
      ↓
/joint_states
      ↓
唯一 robot_state_publisher
      ↓
/tf
```

不要同时：

```text
Isaac 自己发布机器人 link TF
+
robot_state_publisher 再根据 /joint_states 发布同一套 TF
```

二选一。

推荐当前阶段只保留：

```text
/joint_states
      ↓
一个 robot_state_publisher
      ↓
/tf
```

作为唯一机器人 TF authority。

---

# 4. 第三阶段：彻底重构 Ghost Robot

Ghost 是高风险模块之一。

禁止真实机器人和 Ghost 使用相同 frame 名称。

错误：

```text
Real:
base_link
left_link_1
left_link_2

Ghost:
base_link
left_link_1
left_link_2
```

TF frame name 是全局唯一的。

---

## 4.1 第一阶段先取消完整 Ghost Robot

优先使用：

```text
Left Goal:
6DoF Interactive Marker
+
半透明 gripper mesh

Right Goal:
6DoF Interactive Marker
+
半透明 gripper mesh
```

不要显示完整目标机器人。

---

## 4.2 Ghost 必须是纯可视化

推荐：

```text
Goal Pose
   ↓
可选 IK/FK
   ↓
visualization_msgs/MarkerArray
   ↓
RViz 半透明 Mesh
```

Ghost 必须满足：

```text
不发布 /joint_states
不启动 robot_state_publisher
不发布真实机器人 TF
不控制 Isaac Articulation
不包含 Collider
不包含 RigidBody
不参与 Physics
```

最推荐：

```text
Real Robot
→ RobotModel + TF

Ghost
→ MarkerArray
```

---

# 5. 第四阶段：RViz 默认关闭全 TF 坐标轴

截图中的大量红/绿/蓝杆首先确认是否来自：

```text
RViz
→ Displays
→ TF
```

默认关闭：

```text
Show Axes   = false
Show Names  = false
Show Arrows = false
```

TF Display 只在调试时临时打开。

正常运行只保留：

```text
RobotModel
Interactive Markers
Goal Gripper Markers
Trajectory Markers
Environment
```

注意：

> 关闭 Show Axes 只能解决视觉上的“刺猬”，不能解决 TF 冲突。

---

# 6. 第五阶段：Interactive Marker 严禁拖动即规划

检查 Interactive Marker callback。

错误设计：

```python
def feedback_callback(feedback):
    update_goal(feedback.pose)
    plan()
```

或者：

```text
POSE_UPDATE
→ Socket
→ Isaac
→ IK
→ Plan
→ Preview
→ ROS2
→ Marker update
→ 再产生 Feedback
```

这种结构可能造成消息反馈环和高频规划。

---

## 正确逻辑

拖动过程中：

```text
POSE_UPDATE
    ↓
只更新 latest_goal_pose
    ↓
可选 10~20 Hz 限频可视化
```

鼠标释放：

```text
MOUSE_UP
    ↓
保存最终 Goal
```

只有显式点击：

```text
PLAN
```

才调用 cuRobo。

必须满足：

```text
拖动 Goal ≠ IK
拖动 Goal ≠ Plan
拖动 Goal ≠ Execute
```

---

# 7. 第六阶段：Goal 消息采用 latest-value semantics

不要让每一次鼠标拖动产生的 Goal 都进入无限队列。

错误：

```text
Goal1
Goal2
Goal3
...
Goal500
```

Isaac 逐个处理。

旧 Goal 没有价值。

正确：

```python
latest_left_goal = ...
latest_right_goal = ...
```

新值覆盖旧值。

逻辑：

```text
Goal1
↓
Goal2 覆盖
↓
Goal3 覆盖
↓
只消费最终最新 Goal
```

如果必须使用 Queue：

```python
queue.Queue(maxsize=1)
```

新 Goal 到来时丢弃旧 Goal。

---

# 8. 第七阶段：Socket Thread 模型必须重构

当前 transport：

```text
127.0.0.1:50075
TCP JSON socket
```

## 8.1 禁止在 Isaac 主线程阻塞 recv

错误：

```python
while True:
    data = socket.recv(...)
```

如果运行在 Isaac main/update thread，会造成：

```text
UI 卡死
Timeline 卡死
Physics 卡顿
整机假死
```

---

## 8.2 禁止网络线程直接操作 Isaac

Socket thread 中禁止直接调用：

```python
prim.set_world_pose(...)
articulation.apply_action(...)
world.reset(...)
stage.RemovePrim(...)
motion_gen.plan_single(...)
```

推荐结构：

```text
Network Thread
        │
        │ recv JSON
        ▼
Message Parser
        │
        ▼
Thread-safe Command Queue
或 Latest-value Buffer
        │
        ▼
Isaac Main / Update Thread
        │
        ├── SET_GOAL
        ├── PLAN_REQUEST
        ├── EXECUTE_REQUEST
        └── RESET_REQUEST
```

网络线程只负责：

```text
socket accept
socket recv
JSON decode
queue.put
更新线程安全变量
```

所有这些操作：

```text
USD
Stage
Prim
Articulation
Physics
Timeline
cuRobo
```

必须在明确安全的 Isaac execution path 中执行。

---

## 8.3 Socket 必须可正常退出

设置：

```python
socket.settimeout(...)
```

线程建议：

```python
daemon=True
```

Extension shutdown 时必须：

```text
stop_event.set()
socket.shutdown()
socket.close()
thread.join(timeout=...)
```

禁止 reload extension 后遗留旧线程。

---

# 9. 第八阶段：检查数据反馈闭环

必须严格保证数据方向：

```text
Goal command
RViz ─────────────→ Isaac

q_actual
RViz ←───────────── Isaac
```

禁止：

```text
RViz Goal
↓
Isaac Goal
↓
Isaac 又发布 Goal
↓
RViz 更新 Interactive Marker
↓
再次产生 feedback
↓
再次发送 Isaac
↓
循环
```

也禁止：

```text
Isaac q_actual
↓
/joint_states
↓
TF
↓
某节点读取 TF
↓
反向写回 Isaac robot pose
↓
q_actual 改变
↓
循环
```

---

# 10. 建立 Single Ownership：每类数据只能有一个权威源

必须明确：

| 数据 | 唯一权威源 |
|---|---|
| `q_actual` | Isaac Articulation |
| Left/Right `goal_pose` | RViz Interactive Marker |
| `q_plan(t)` | cuRobo Planner |
| `q_command(t)` | Trajectory Executor |
| Robot TF | 唯一 robot_state_publisher |
| Ghost Visualization | RViz MarkerArray |

规则：

> 任何模块只能消费其他模块的权威数据，不允许反向覆盖。

---

# 11. 第九阶段：实现严格状态机

实现：

```text
IDLE
  ↓ drag
EDITING_GOAL
  ↓ mouse up
GOAL_READY
  ↓ PLAN
PLANNING
  ↓ success
PLAN_READY
  ↓ EXECUTE
EXECUTING
  ↓ complete
IDLE
```

---

## PLANNING 状态

禁止：

```text
再次 PLAN
并发 Execute
并发 Reset
修改 world structure
启动第二个 cuRobo planning task
```

新 Goal 可以记录，但不能自动触发新规划。

---

## EXECUTING 状态

禁止：

```text
set_joint_positions() 连续瞬移
Goal 拖动驱动真实 robot
重新初始化 Articulation
同时执行第二条 trajectory
```

---

## RESET

必须：

```text
cancel execution
clear trajectory
clear planning task
reset articulation
重新同步 world
```

不能和 Plan / Execute 并发。

---

# 12. 第十阶段：确认 cuRobo 每次只执行一次规划

所有 planner 入口加入日志：

```text
[PLAN_REQUEST] id=17
[PLAN_START] id=17
[PLAN_END] id=17 duration=...
```

统计：

```text
plan_calls_per_second
```

正常：

```text
点击一次 PLAN
→ plan() exactly once
```

如果出现：

```text
1 秒内 plan() 被调用几十次
```

必须立即追踪调用链。

增加规划互斥：

```python
if self._planning:
    reject_or_ignore_new_plan()
```

禁止并发多个 MotionGen / cuRobo planning。

原因：

```text
Isaac RTX Renderer
+
cuRobo CUDA
+
RViz
```

都在使用同一张 GPU。

重复规划可能造成：

```text
VRAM 暴涨
CUDA workload 堆积
GPU 持续 100%
桌面冻结
driver hang
```

---

# 13. 第十一阶段：cuRobo 初始化只能做一次

严禁在 update callback 中反复执行：

```python
MotionGenConfig.load_from_robot_config(...)
MotionGen(...)
motion_gen.warmup()
WorldConfig(...)
load robot yaml
build collision checker
```

正确：

```text
Extension startup
    ↓
初始化 MotionGen
    ↓
warmup 一次

每次 PLAN
    ↓
读取 q_start
读取 goal
读取当前 world transform
    ↓
调用 plan
```

禁止每次拖 Goal：

```text
重新加载机器人
重新初始化 cuRobo
重新 warmup
重新构建 collision checker
```

---

# 14. 第十二阶段：World 更新区分 Transform 与 Structure

已有 obstacle：

```text
只移动 position / orientation
→ 更新 transform
```

以下操作属于结构变化：

```text
新增 obstacle
删除 obstacle
修改尺寸
修改 collision geometry
```

应：

```text
STOP
↓
rebuild world binding / collision world
↓
重新规划
```

禁止每个 physics frame：

```text
重新扫描所有 collider
重新创建 WorldConfig
重新加载 mesh
```

---

# 15. 第十三阶段：检查 callback / timer / thread 泄漏

全局搜索：

```text
while True
while self._running
get_update_event_stream
subscribe_physics_step_events
add_physics_callback
asyncio.ensure_future
create_task
threading.Thread
Timer
create_timer
```

逐个确认：

1. 是否可能重复注册；
2. Extension reload 后旧 callback 是否注销；
3. Reset 是否又创建新 callback；
4. 每按一次 Load/Run 是否创建新线程；
5. 是否缺少 unsubscribe；
6. Timer 是否重复创建。

危险模式：

```text
第一次 Load
→ 1 个 callback

Reset
→ 2 个

再次 Reset
→ 3 个

运行一段时间
→ 几十个 callback 同时执行
```

这种问题会表现为：

```text
刚启动正常
↓
逐渐变卡
↓
抽搐越来越严重
↓
最后系统卡死
```

所有 callback / thread / subscription 必须保存 handle，并在 cleanup 中显式释放。

---

# 16. 第十四阶段：检查 Socket busy loop

禁止：

```python
while self.running:
    try:
        recv_nonblocking()
    except:
        pass
```

如果没有：

```text
sleep
select
blocking recv
timeout
```

会让 CPU 核心持续 100%。

推荐：

```text
blocking recv + timeout
```

或者：

```text
select / selectors
```

空闲状态下 socket thread CPU 使用率应接近 0。

---

# 17. 第十五阶段：加入资源监控

复现问题时同时打开三个终端。

## Terminal 1：CPU/RAM

```bash
htop
```

观察：

```text
CPU 是否持续 100%
RAM 是否不断上涨
Swap 是否持续上涨
是否出现大量重复 Python/Isaac/ROS 进程
```

---

## Terminal 2：GPU/VRAM

```bash
watch -n 0.5 nvidia-smi
```

观察：

```text
GPU Util
Memory-Usage
Isaac Sim
RViz
Python/CUDA
```

特别检查：

```text
VRAM 是否持续上涨而不回落
```

例如：

```text
5 GB
→ 8 GB
→ 12 GB
→ 15 GB
→ 继续上涨
```

如果每次拖 Goal 或每次 Plan 后显存都增加，怀疑资源泄漏或重复初始化。

---

## Terminal 3：ROS2 消息频率

```bash
ros2 topic hz /joint_states
```

以及：

```bash
ros2 topic hz /tf
```

正常可以几十到几百 Hz。

异常情况：

```text
几千 Hz
几十 kHz
```

说明存在消息风暴或反馈环。

---

# 18. 第十六阶段：死机后检查系统日志

如果再次整机死机，重启后执行：

```bash
journalctl -k -b -1 | grep -Ei \
'oom|out of memory|killed process|nvrm|xid|gpu|segfault|hung|watchdog'
```

再执行：

```bash
journalctl -b -1 -p err..alert
```

---

## 18.1 如果出现 OOM

例如：

```text
Out of memory
Killed process
oom-killer
```

说明：

```text
RAM / Swap 被耗尽
```

重点排查：

```text
队列无限增长
Marker 消息积压
trajectory 对象未释放
重复 callback
重复 cuRobo 初始化
日志缓存
```

---

## 18.2 如果出现 NVIDIA Xid

例如：

```text
NVRM: Xid
GPU has fallen off the bus
```

重点排查：

```text
重复 cuRobo planning
RTX Renderer + CUDA 争抢
VRAM 耗尽
GPU driver hang
```

---

## 18.3 如果没有内核错误，但 CPU 长期 100%

重点排查：

```text
socket busy loop
消息反馈环
重复 callback
高频 marker callback
重复 planner
```

---

# 19. 第十七阶段：建立 SAFE_MODE 分级启动

不要一次启动全部功能。

增加：

```text
SAFE_MODE
```

按以下阶段逐步启用。

---

## Level A

只运行：

```text
RViz
+
静态 RobotModel
```

运行至少 5 分钟。

要求：

```text
无抽搐
CPU稳定
RAM稳定
```

---

## Level B

增加：

```text
Isaac q_actual
→ socket
→ ROS2 adapter
→ /joint_states
→ RobotModel
```

不启：

```text
Interactive Marker
Ghost
cuRobo
```

要求：

```text
/joint_states publisher = 1
RobotModel稳定
TF稳定
CPU/RAM/GPU稳定
```

---

## Level C

增加：

```text
Interactive Marker
```

只拖 Goal。

要求：

```text
真实 Robot 完全不动
/joint_states 不受影响
TF 不受影响
无高频 plan
```

---

## Level D

增加：

```text
Ghost Gripper Marker
```

只显示半透明末端，不显示完整 Ghost Robot。

---

## Level E

增加：

```text
cuRobo 初始化
```

只初始化，不规划。

要求：

```text
只初始化一次
只 warmup 一次
显存稳定
```

---

## Level F

点击一次：

```text
PLAN
```

要求：

```text
exactly one planning call
```

规划结束后：

```text
真实机器人仍然不动
只显示 preview
```

---

## Level G

增加：

```text
EXECUTE
```

执行最近一次成功规划。

---

# 20. 最终推荐模块架构

## ROS2 / RViz Process

```text
InteractiveMarkerManager
    │
    ├── left_goal_pose
    └── right_goal_pose
           │
           ▼
SocketTransportClient
           │
           │ localhost JSON
           ▼
```

---

## Isaac Sim Process

```text
SocketTransportServer
           │
           ▼
CommandDispatcher
           │
   ┌───────┼────────┐
   ▼       ▼        ▼
Goal    Plan      Execute
Manager Request   Request
           │
           ▼
PlanningManager
   │
   ├── RobotStateManager
   ├── GoalManager
   ├── WorldManager
   ├── CuroboBackend
   └── TrajectoryExecutor
           │
           ▼
Isaac Articulation
```

---

# 21. 数据方向必须固定

```text
q_actual
Isaac → ROS2
单向
```

```text
goal_pose
RViz → Isaac
单向
```

```text
q_plan
cuRobo 内部生成
```

```text
Plan
≠
Execute
```

```text
Ghost
纯可视化
绝不控制真实机器人
```

---

# 22. 建议暂时删除或禁用的功能

优先删除/禁用：

```text
Robot Poser Track Target

拖 Goal 时实时 IK 驱动真实 Articulation

完整物理 Ghost Articulation

Ghost 发布 /joint_states

Ghost 发布真实机器人 link TF

拖 Goal 自动 cuRobo planning

每 frame 重建 world

每 frame 重扫 collision geometry

多个 robot_state_publisher

joint_state_publisher 与 Isaac joint state 同时运行
```

这些功能稳定前不要恢复。

---

# 23. 增加诊断日志

每秒最多输出一次系统摘要，不要每帧 print。

示例：

```text
[DIAG]
state=GOAL_READY
socket_rx_rate=12.3 msg/s
joint_state_tx_rate=60.0 Hz
goal_update_rate=8.1 Hz
plan_count=0
queue_size=0
threads=3
rss_memory=4.8 GB
```

规划：

```text
[PLAN]
request_id=17
start
success=true
duration=0.043s
trajectory_points=126
```

要求记录：

```text
plan_count
active_threads
queue_size
socket_rx_rate
goal_update_rate
joint_state_publish_rate
process RSS
GPU memory if practical
```

---

# 24. 最优先执行顺序

## Priority 1

检查：

```bash
ros2 topic info /joint_states -v
ros2 topic info /tf -v
ros2 topic info /tf_static -v
```

确保：

```text
真实 joint state 只有一个 publisher
机器人 TF 只有一个 authority
```

---

## Priority 2

Ghost 简化为：

```text
Interactive Marker
+
半透明 gripper marker
```

取消：

```text
Ghost Articulation
Ghost /joint_states
Ghost robot_state_publisher
Ghost 真实 link TF
```

---

## Priority 3

检查：

```text
拖 Marker
→ 只更新 latest_goal

点击 PLAN
→ exactly one cuRobo plan()

Socket thread
→ 只接收和解析
→ Queue / latest-value buffer
→ 不直接操作 Isaac
```

---

## Priority 4

建立 SAFE_MODE，按 Level A → G 分级恢复功能。

---

# 25. 验收标准

完成修改后必须满足：

### RViz

```text
无大量默认 TF axes
RobotModel 不抽搐
Interactive Marker 拖动流畅
```

### ROS2

```text
/joint_states 真实 publisher = 1
无重复 TF authority
无异常高频 topic
```

### Goal Editing

```text
拖动 Goal
→ 真实机器人完全不动
→ 不触发 planning
→ 不修改 /joint_states
```

### Planning

```text
点击一次 PLAN
→ exactly one planning request
→ Plan 和 Execute 分离
```

### Ghost

```text
纯可视化
无 Physics
无 Articulation
无真实 TF 冲突
```

### Socket

```text
无 busy loop
无主线程 blocking recv
网络线程不直接操作 USD / Articulation
支持 clean shutdown
```

### 资源

```text
CPU 稳定
RAM 不持续上涨
VRAM 不持续上涨
无 callback/thread 数量持续增长
```

### 长时间测试

至少连续运行：

```text
30 分钟
```

期间重复：

```text
拖 Goal
Plan
Reset
Plan
Execute
Reset
```

要求：

```text
无抽搐加剧
无明显资源泄漏
无系统假死
无 GPU driver error
```

---

# 26. 最终设计原则

整个系统必须遵守：

```text
Goal Editing
    ≠
Robot Control
```

```text
Plan
    ≠
Execute
```

```text
Ghost
    ≠
Real Robot
```

```text
q_actual
    ≠
q_goal
    ≠
q_plan
    ≠
q_command
```

```text
每类数据只有一个权威源
```

```text
所有跨线程通信通过 Queue / thread-safe buffer
```

```text
所有高开销操作必须显式触发并防止重复执行
```

当前阶段的目标不是增加功能，而是先建立一个：

> **单状态源、无 TF 冲突、无反馈环、无线程泄漏、Goal 与真实机器人彻底解耦、Plan 与 Execute 分离的稳定基础架构。**
