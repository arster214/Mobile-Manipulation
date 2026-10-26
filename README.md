# Mobile-Manipulation

RM75 双臂、差速底盘及升降平台的仿真规划工作空间。当前唯一维护分支为 `main`。

## 当前状态

- `scenarios/table_transfer/planning/`：全身预抓取、固定底盘/升降的深入、轨迹验证和重规划接口。
- `scenarios/table_transfer/tests/`：所有本阶段实验和测试；`artifacts/` 保留复现实验所需的配置、轨迹和数值结果，日志与截图不入库。
- 正式碰撞模型为511球；16层三排两列柱体仍为未启用候选。按用户要求，柱体与左右肩部安装座的自碰撞对已排除，环境碰撞保留。
- `realman_rm75/` 是原工程代码和机器人资产的实际副本；本仓库不使用本机绝对路径软链接。
- `curobo_v2/` 为官方源码子模块，固定提交；使用 `git submodule update --init` 获取。
- 仓库不包含Isaac Sim安装、Python虚拟环境、构建输出和运行日志；依赖版本见 `env/locks/`。
- 目前代码和配置仍含本实验机的 `/home/fang/robot_ws`、`/home/fang/isaacsim_rm75` 与 `/home/fang/isaacsim` 路径。换机器需调整这些路径；本次是当前工作状态提交，并非可一键部署的软件包。
- 轨迹验证为离散数值检查，尚未完成真实抓持反馈、释放反馈与实机执行验证。

详细实验说明见 `scenarios/table_transfer/README.txt`。

## 原环境搭建说明（历史记录）

# robot_ws 移动操作验证工作空间

## 目录与运行环境

| 目录 | 用途 |
| --- | --- |
| `curobo_v2/` | 已有 cuRobo v2 源码仓库 |
| `env/pin-pink/` | 新迁入的独立 Python 3.10 环境：Pinocchio 4.1.0、Pink 4.4.0、quadprog、Meshcat |
| `env/locks/pin-pink-py310.txt` | Pinocchio/Pink 完整依赖版本清单 |
| `pin_pink/` | 独立 Python 启动入口、基础验证、RM75 双臂 IK 示例 |
| `realman_rm75/` | 指向 `/home/fang/isaacsim_rm75` 的软链接，复用已有模型、配置和仿真脚本 |

模型原始工程在工作空间外，通过软链接统一入口；此目录中的修改会直接修改原工程。已有 cuRobo 源码、Isaac Sim 安装和模型内容未做变更。

## Pinocchio / Pink

推荐入口清理继承的 ROS/Isaac Python 和动态库路径，仅影响启动的子进程：

```bash
cd /home/fang/robot_ws
./pin_pink/python.sh pin_pink/check_environment.py
./pin_pink/python.sh pin_pink/check_rm75.py
./pin_pink/python.sh -m pip check
./pin_pink/python.sh your_script.py
```

也可常规激活：

```bash
source /home/fang/robot_ws/env/pin-pink/bin/activate
python -c "import pinocchio, pink; print(pinocchio.__version__)"
deactivate
```

常规激活保留终端已有 ROS 变量，安装依赖和独立运行时优先使用 `pin_pink/python.sh`。
VS Code 解释器选择 `/home/fang/robot_ws/env/pin-pink/bin/python`。

## RM75 模型与验证范围

URDF：`realman_rm75/assets/urdf/rm75_dual_rmg24_curobo.urdf`；网格：`realman_rm75/assets/meshes/`。
原仿真模型为 `rm75_dual_rmg24_sim.urdf`，现有 cuRobo 配置为 `realman_rm75/config/rm75_dual.yml`。

`check_rm75.py` 显式提供网格搜索目录，加载可视与碰撞几何，并运行两种局部双臂逆运动学：

- 固定底座：锁定升降、头部、夹爪，保留左右臂 14 个关节。
- 平面底座：在上述模型前增加 `JointModelPlanar`，提供 x、y、yaw 的 3 个速度自由度，总计 nv=17；平面旋转采用 cos/sin 表示，因此 nq=18。

升降固定于 0.5 m、夹爪固定于 0.0325 m；测试目标由邻近关节配置生成。TCP 使用 `l_tool0`、`r_tool0`，沿用原工程的法兰参考系。该测试仅验证模型加载和数值求解；几何已加载但未在 IK 中添加避碰约束。平面底座未加入非完整约束、轮速映射或实机控制。

现有 cuRobo 配置仍是固定底座、双臂 14 自由度规划；这里的平面底座示例没有改动该配置。移动底座与双臂的协调、全局避障，以及 ROS/Isaac 联动需要后续单独实现和验证。

## 现有 cuRobo / Isaac Sim

本次确认已有解释器 `/home/fang/isaacsim/python.sh` 使用 Python 3.12.13，可导入 cuRobo `0.8.0.post1.dev41`。它与 Pinocchio/Pink 环境独立运行；本次仅检查 cuRobo 导入，没有执行 GPU 规划或启动仿真。

原工程命令从软链接入口也可以运行：

```bash
cd /home/fang/robot_ws/realman_rm75
/home/fang/isaacsim/python.sh scripts/validate_config.py
```

现有工程 README 提供完整 Isaac / RViz / MoveIt 工作流。后续跨环境共享状态和轨迹时，应明确关节名称、TCP、世界坐标系与底座姿态约定。

## 重建 Pinocchio/Pink 环境

以下新目录应不存在，面向 Linux x86_64 / Python 3.10：

```bash
python3 -m venv --without-pip /path/to/new-env
env -u PYTHONPATH -u LD_LIBRARY_PATH python3 -m pip --python /path/to/new-env/bin/python install pip==26.2.2 -r /home/fang/robot_ws/env/locks/pin-pink-py310.txt
```
