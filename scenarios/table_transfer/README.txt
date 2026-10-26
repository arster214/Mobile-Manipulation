2026-10-26 抓取构型优先（最新流程）
规划顺序：最终抓取全身IK -> 固定底盘/升降/非抓取臂回退15cm -> 初始状态连接该预抓取构型 -> 回退轨迹时间反向作为深入。
planning/arm_comparison.py 的 plan_arm_from_grasp/compare_arms 实现该流程；起终点都满足约束还不够，必须整段回退及连接验证成功才选中。
接触阶段仅从环境副本移除目标Cup；全局环境和连接段仍保留Cup；自碰撞、桌面和墙体碰撞始终检查。
保留左右20度扇区、双臂相对原始实测姿态各关节<=0.8rad、抓取臂 sigma_min>=0.005 和 condition<=100。
深入由同一条合格回退轨迹反向生成，包含速度符号与时间反转；执行时仍读取实际状态、重建固定平台模型并复查。

同场景中间向两侧共5个高度候选搜索结果：选中右臂、rank4；抓取点(-1.05,1.32,0.8442857)，
预抓取目标(-1.05,1.17,0.8442857)。杯心仍为(-1.05,1.32,0.81)，没有移动杯子来凑结果。
右臂预抓取连接段5.85s，深入1.80688s，总代价4.32003；最终选中状态为selected_arm=r。
独立Pinocchio审计：深入所有采样最小奇异值0.054981、最大条件数31.6225、最大双臂关节变化0.798614rad。
深入TCP采样最大目标误差4.8887mm，最终误差约0.00415mm，最大姿态误差0.002142rad，均在当前容差内。
左臂在当前搜索预算和5个高度候选内没有完整可行路径，不代表数学上不可达。
这只是数值规划与离散采样验证，不是PhysX、真实抓取或连续碰撞/动力学验证；初始姿态的奇异性限制说明仍适用。

复现：/home/fang/isaacsim/python.sh tests/test_grasp_first.py
独立审计：/home/fang/robot_ws/pin_pink/python.sh tests/audit_grasp_first.py
结果与分段轨迹：tests/artifacts/grasp_first/result.json、r/pregrasp.npz、r/approach.npz。
新增相位环境排除和回退反向测试2项通过；模块边界7项通过。历史8094/8095回放未覆盖，仍显示旧实验。

2026-10-26 左右臂严格约束与路径代价比较（最新）
planning/arm_comparison.py 从同一初始状态/世界快照比较左右臂，返回两条候选、预抓取优先臂、完整可行臂。
planning/selection.py 定义扇区、奇异性指标、统一代价；planning/config.json 保存阈值和权重。
左臂：物体在正前方左侧 [0,20] 度；右臂：右侧 [0,20] 度。正前方沿 URDF -Y。
双臂各关节相对原始实测姿态的变化均 <=0.8 rad，求解 URDF 边界与路径采样双重检查；
深入阶段仍沿用原始参考姿态，不重置 0.8 rad 预算。底盘、升降及非抓取臂锁定于预抓取实测值。
奇异性：抓取臂 TCP 的 6x7 几何 Jacobian（线速度 m/rad、角速度 rad/rad，不加入底盘/升降），
新 IK 终点与深入采样要求 sigma_min>=0.005、condition<=100。
初始姿态不满足该指标：左臂 sigma_min=0.000251、condition=6936；右臂 0.003861、451。
因此移动段报告完整采样指标，但只对新 IK 终点施加奇异性门槛；不能声称移动段全程避开奇异。
若要求包括起点在内的全程奇异性约束，必须先调整初始姿态；当前没有擅自替换原始初始姿态。

代价 J = 路程/1m + 累计转角/1rad + 升降累计行程/0.79m
       + 抓取臂关节累计行程的 RMS/0.8rad + 2*另一臂对应项 + 0.1*时长/1s。
在固定同等搜索预算内比较所有搜索到的 IK 解及其直达/两种折线路线、前进/倒车、三种上身时序。
在代价排序后做验证，不再返回生成顺序中的第一条可行路线。不是全局最优性证明。

本次瓶子位置 (-1.05,1.32,0.81)：
左臂：左侧16.7445度；max_joint_delta=0.63010rad；sigma_min=0.067807；condition=25.535；
      预抓取代价4.39707，底盘路程0.45641m、累计转角2.71130rad、时长7.25s。
右臂：右侧18.4234度；max_joint_delta=0.56636rad；sigma_min=0.082040；condition=21.231；
      预抓取代价2.70254，底盘路程0.03346m、累计转角1.67363rad、时长4.875s。
两侧非抓取臂变化接近浮点误差。Pinocchio 独立审计确认上述终点指标与整条关节变化范围。
右臂预抓取代价低38.5%，preferred_pregrasp_arm=r；左/右深入分别在53.3%/73.3%失败，
selected_arm=null，不把预抓取偏好冒充完整抓取成功。没有发送实机控制命令。
16项离线测试通过（6同步运动+7模块边界+3新约束），两臂GPU实验和独立Pinocchio审计已运行。
结果/轨迹：tests/artifacts/arm_comparison/result.json、l/pregrasp.npz、r/pregrasp.npz。
复现：/home/fang/isaacsim/python.sh tests/compare_arm_paths.py
审计：/home/fang/robot_ws/pin_pink/python.sh tests/audit_arm_constraints.py
新约束测试：/home/fang/robot_ws/pin_pink/python.sh -m unittest discover -s tests -p 'test_arm_constraints.py'
8094 仍为之前失败轨迹回放，不是本次新路径。

2026-10-26 模块化规划基线（最新进度，以下历史记录保留）
代码位于 planning/；实验程序统一放 tests/，新产物放 tests/artifacts/。
A model.py / interfaces.py：统一 SI 单位、world 位姿 wxyz、18 自由度与关节名称映射。
  cuRobo 内部左右臂顺序与配置顺序不同，种子、碰撞输入和 IK 输出均显式按名称转换。
B perception.py：仿真真值输入、环境版本、最近长边一侧的中间向外抓取候选和 TCP 转换。
C curobo_backend.py / trajectory.py / differential_drive.py：GPU IK、自碰撞/环境检查；
  三种差速底盘路线与三种上身时间安排的有限候选搜索。底盘、升降、双臂同时间轴运动。
  这不是 cuRobo 原生全局轨迹优化器，也不是完备导航搜索。尚需扩展路线搜索和轨迹优化。
  CUDA Graph 在本机 18 自由度测试遇到 illegal instruction，当前显式关闭；IK/碰撞仍使用 GPU。
D execution.py：经过验证的轨迹分发、锁定关节保持、反馈误差和分段完成判定；控制器用协议注入。
E replanning.py / pipeline.py：目标/环境版本变更后 hold，读取实际反馈并请求重规划；
  当前采用停止后重规划，不支持运动中无缝切换。预抓取完成后才能创建实测锁定的右臂深入模型。
  深入失败返回重新规划预抓取，绝不自动释放底盘/升降/左臂。
F local_control.py：接口预留，PINK 局部控制明确关闭，遵循先完成 cuRobo 基线的要求。

验证：12 项离线单元测试通过。原始初始姿态通过 GPU 碰撞检查和独立 Pinocchio 球体审计。
一次完整预抓取数值规划约 1.95 秒，444 点、时长 11.075 秒；检查命令点和中点碰撞、
关节速度/加速度、底盘差速积分与 1.5m/s 上限。最终 TCP 误差约 2.8e-7 m（数值模型）。
这些结果不是 PhysX 执行验证，也不是连续碰撞证明；没有发送实机命令。
锁定后的深入测试在 73.3% 处 IK 失败；仅诊断移除目标杯碰撞后仍在 80% 处失败，
所以不能仅靠忽略杯子解决。应联合筛选预抓取构型及整段固定底盘的接近可行性。
该诊断世界不会进入执行模块。详细结果保存在 tests/artifacts/baseline/result.json。
尚未完成：Isaac Sim 控制器适配与实测反馈闭环、整段深入可行性筛选、抓取/附着/放置流程、
实时轨迹碰撞复查与跟踪偏差处理；因此不可将这些模块接口视作整套搬运任务已完成。
仿真角速度/加速度采用 config.json 中明确命名的 provisional simulation 参数，非实机标定；
真实底盘对应限制仍为 null。固定高度模型的地面支撑从规划障碍物中排除。

复现（在本目录）：
/home/fang/robot_ws/pin_pink/python.sh -m unittest discover -s tests -p 'test_synchronized_motion.py'
/home/fang/robot_ws/pin_pink/python.sh -m unittest discover -s tests -p 'test_module_pipeline.py'
/home/fang/robot_ws/pin_pink/python.sh tests/audit_pipeline_start.py
/home/fang/isaacsim/python.sh tests/test_gpu_pipeline.py
GPU 测试仅断言全身预抓取通过，深入阶段成功/失败另行写入结果，不把失败伪装为成功。

RM75 右臂双桌搬运环境（环境搭建阶段）

入口：./start_scene.sh
环境物理验证与截图：./start_scene.sh --validate
重新生成：/home/fang/isaacsim/python.sh scripts/build_scene.py

主场景 assets/rm75_table_transfer.usda；独立环境 assets/environment.usda。
配置 scene_config.json；场景清单 scene_manifest.json；测试结果 validation.json。
所有坐标单位米，Z向上，地面顶面z=0，房间中心xy=(0,0)。
房间内部净尺寸7×7米，四面墙高2.5米、厚0.12米，无顶盖便于观察。
两桌尺寸1.4×0.6×0.75米，0.04米厚桌面、四条0.06米方腿。
源桌中心xy=(-1.2,1.5)，目标桌中心xy=(1.2,1.5)，桌间净间隔1米。
杯子为动态实心圆柱体，直径0.06米、高0.12米、质量0.15kg。
初始中心(-1.2,1.32,0.812)，落稳约(-1.2,1.32,0.810)。
目标杯心(1.2,1.32,0.810)，绿色圆片仅作目标显示，没有碰撞属性。
机器人初始xy=(-1.3,0.5)，绕Z旋转180度；该URDF头部朝向为局部-Y，旋转后面对桌子。
机器人root的z由原始USD包围盒底面计算，以确保轮子在地面上方约2mm。
使用现有机器人USD引用，不更改共享URDF、机器人USD或碰撞球文件。

任务约定：右臂抓源桌上的杯子，经底盘移动后放到目标桌。
当前只完成环境：没有自动运行底盘、右臂IK、抓取闭合或搬运动作。
场景默认暂停，后续需初始化升降/关节姿态、验证停靠点与可达性、接入控制器。
碰撞球模型由另一agent负责，当前场景未绑定实验碰撞球配置。

static_world.yml导出地面、四墙和两桌的15个静态碰撞盒。
该文件坐标是房间世界系，后续cuRobo若以机器人base_link为参考，需要随底盘运动更新坐标变换。
水杯为动态物体，不写死进static_world.yml；运行时从/World/Environment/Cup读取真实位姿。
抓取成功后须切换杯子为附着物体的规划碰撞表示，放置后再回到环境物体；目前未实现此逻辑。

验证：独立环境中用120Hz步长运行360步，检查杯子落稳、桌面高度、标记无碰撞。
完整机器人场景已成功加载并渲染。尚未验证机器人动态稳定性、右臂轨迹或移动导航。

2026-10-25 移动操作初始姿态
- 初始姿态文件：/home/fang/robot_ws/postures/mobile_manipulation_initial.json
- cuRobo V2 配置：/home/fang/robot_ws/postures/rm75_mobile_manipulation.yml
- 工具 Z 朝机器人本体 -Y；场景底盘 yaw=180 度时朝世界 +Y。
- 双肘下垂、小臂水平；为避开当前平台碰撞球，肘部相对肩部向外 0.15 m、向前 0.07 m。
- 升降关节 0.5 m；头部和夹爪沿用配置锁定值。
- start_scene.sh 每次启动读取该姿态并设置仿真关节位置、驱动目标与默认状态，随后暂停。
- start_scene.sh --check-posture 可无界面检查关节初始化及读回，结果写 initial_posture_validation.json。
- 已验证 cuRobo V2 静态自碰撞代价为 0；沿用当前碰撞排除表，额外球缓冲为 0。
- 环境碰撞、导航及抓取轨迹尚未验证，不代表整个实验轨迹已经可执行。

工具姿态更新：左右工具 Z 朝本体 -Y，工具 X 朝本体 +X（与地面平行），工具 Y 朝上，腕部相机在上侧。仅调整腕部姿态，保留原肘部和小臂位置。已重新通过 cuRobo V2 静态自碰撞检查。

最新姿态覆盖上述朝向：按用户要求，左右工具绕自身 Z 轴顺时针 90 度（从 +Z 看向原点）。工具 X 朝下，Y 水平朝本体 +X，Z 朝前。未重新定义 URDF 工具系；实际腕部关节姿态发生变化。相机光轴尚未独立确认。

抓取 TCP：l_grasp_tcp / r_grasp_tcp，tool0 局部平移 [0,0,0.1168] m，旋转为零。参考点选两指指腹接触区中心深度，来自网格，尚非实机标定。当前规划 YAML 注册新旧四个工具系，场景 grasp_tcp_frames 指定新抓取工具。已有 grasp_candidates 的抽象轴定义不同，送入 IK 前需按 grasp_policy.json 的转换绕局部 Z -pi/2；切勿仅替换帧名后直接使用旧候选朝向。

TCP 已固化至仿真源 URDF、规划源 URDF、Isaac Sim USD 资产及基础/实验 cuRobo 配置。USD grasp_tcp 挂在对应 tool0 下，局部平移 (0,0,0.1168) m、旋转为零；不增加刚体、质量或碰撞几何。场景引用加载已验证。完整路径见 scene_config.json 的 robot.grasp_tcp_prim_paths，核对报告见 /home/fang/robot_ws/postures/grasp_tcp_model_sync.json。

躯干两侧微调：实验配置 body_base_link 移除原85/86/95/96号球，加入17对镜像半径25mm小球。整机432球。独立采样躯干覆盖99.37%，调整区域94.72%，原覆盖区域最大新遗漏15.84mm。20组终态躯干侧球体重叠消除，平台/手臂不变，不能视为20组完整解已无碰撞。原配置备份 collision_models/hybrid/torso_sides_refined/before.yml。

平台两侧微调：保留躯干调整，平台移除10球，新增21对镜像小球，整机464球。平台整体覆盖99.08%，局部96.07%，原覆盖区域最大新遗漏8.53mm。20终态平台相关重叠全部消除，完整球体自碰撞筛查7/20通过，另外13组仍有右肩座与r_link2重叠。备份与报告：collision_models/hybrid/platform_sides_refined/。

用户确认启用躯干及平台调整后的464球版本。已同步基础rm75_dual.yml与实验配置，当前模型登记文件 collision_models/current_model.json；历史候选保留。

2026-10-26 架构 V1 与同步移动要求（优先于之前的分阶段建议）
- 主规划：cuRobo V2；Pinocchio + Pink 仅保留为后续可选局部修正。首阶段不接双规划器切换。
- 目标是底盘、升降和双臂同时运动：状态共18自由度（底盘x/y/yaw、升降、左右臂各7关节）。右臂抓取，左臂可调整避障。
- 底盘不是全向底盘：前向为URDF局部-Y，控制为前向线速度v和角速度omega，支持原地转向；最高线速度1.5m/s。
- 底盘控制与上身关节轨迹必须共用时间轴，检查整条全身运动。不得以独立轨迹重叠播放或全向关节插值代替差速同步规划。
- 现有±30度规则约束最终瓶子相对底盘朝向。角速度和加速度的实机上限尚未核实，配置保留空值，不编造实机参数。
- planning/ 已增加统一状态/目标/轨迹接口、18自由度语义及差速轨迹积分/检查；这些基础代码不等于同步运动规划器已经完成。
- 所有实验、比较、审计和可视化测试脚本已从 scripts/ 集中到 tests/；历史命令中的相关 scripts/ 路径需替换为 tests/。历史实验结果仍保存在 ik_test/，不重复复制。
- 新增基础测试：/home/fang/robot_ws/pin_pink/python.sh /home/fang/robot_ws/scenarios/table_transfer/tests/test_synchronized_motion.py
- 已通过6项基础测试：18自由度组成、按名称映射关节、-Y前进、原地旋转/圆弧、拒绝横移轨迹、线速度限制。
- 当前验证仍停留在IK终态及基础运动学；同步无碰撞轨迹生成、Isaac动态执行与跟踪、运行中重规划尚未实现，不能宣称已可边走边抓。

失败深入回放：tests/view_failed_approach.py；viser http://127.0.0.1:8094/。
深入点按同配置/种子重新求解补存于 tests/artifacts/baseline/approach_partial.npz；
只回放至最后有效 IK 点（66.7%），73.3% 失败目标以红色标记，绝不虚构失败机器人姿态。
深入回放每点 0.5 秒为展示时间，不代表通过验证的执行轨迹；支持拖动帧、碰撞球开关及跳转。

最新预抓取对比回放：http://127.0.0.1:8095/；tests/view_pregrasp_comparison.py。
只读取 arm_comparison/l/pregrasp.npz 和 r/pregrasp.npz，不加载任何深入段。
左右臂下拉切换、播放/暂停、进度拖动、终点跳转、碰撞球开关；数值姿态回放，非物理执行。
导出：/home/fang/robot_ws/pin_pink/python.sh tests/view_pregrasp_comparison.py --export
启动：/home/fang/isaacsim/python.sh tests/view_pregrasp_comparison.py

深入失败诊断（左右臂严格约束版本）：tests/diagnose_approach.py；结果 arm_comparison/diagnosis/result.json。
固定同一预抓取终点的底盘、升降和另一臂，以 Pinocchio+SciPy 24种子纯几何IK做消融，再由GPU检查碰撞/奇异性。
右臂73.3%：0.8rad范围内找到15个几何解，全部通过奇异性和自碰撞检查；全部与Cup重叠，
去掉Cup后全部无环境碰撞。重叠链接为r_Link_finger1、r_Link_finger2，所以该失败点是手指球与杯子表示冲突。
右臂最终抓取点：0.8rad范围内未找到合格几何解，最好位置误差约11.6mm；放宽至物理限位后找到19解，
其中最小最大关节变化约1.113rad，对应条件数约120.7，仍不满足用户约束。
左臂53.3%：0.8rad范围内无几何合格解，最好约12.6mm；物理限位下21解均无自/环境碰撞，
最小最大关节变化约1.164rad，对应条件数约166.9。最终点物理范围24种子也未找到合格解，最好约51.6mm。
以上是有限多起点数值证据，不是严格不可达证明；消融只用于诊断，没有修改生产碰撞或关节约束。

抓取优先完整回放：http://127.0.0.1:8096/；tests/view_grasp_first.py。
从 grasp_first/r/pregrasp.npz 与 approach.npz 读取；深入按关节位置线性插值显示，底盘/升降/左臂保持预抓取终值。
支持全程播放、从深入起点播放、抓取位姿跳转、进度和碰撞球开关；仅可视化，不含夹爪闭合。

2026-10-26 目标托盘与搬运扩展
托盘外尺寸X=0.25m、Y=0.18m，中心(1.2,1.32)，位于TargetTable；前沿距桌边0.03m。
底板厚0.01m、挡边高0.025m，底板上表面z=0.76m；瓶子放置中心(1.2,1.32,0.82)。
build_scene.py已生成托盘USD碰撞与static_world.yml的5个托盘盒，场景共20个静态盒。
托盘预览：http://127.0.0.1:8097/；tests/view_tray.py，橙色半透明瓶子是目标预览，并非已执行放置。
从保存的抓取终点进行数值搬运实验：tests/test_table_transport.py；产物tests/artifacts/transport/。
10cm平台抬升触发自碰撞，当前采用5cm，保持碰撞检查。四段lift/carry/place/withdraw均通过本轮采样检查。
瓶子随TCP刚性变换，以圆柱外包围盒检查环境；抓持接触排除右夹爪指和夹爪基座，其他机器人链接仍检查。
放置用升降平台下放，抽离时平台和底盘固定。释放在本实验中为模拟任务边界，没有夹持/释放实测反馈或PhysX执行。
用户目前要求先检查托盘位置；搬运结果尚待进一步独立审计和控制器接入，不代表实机搬运已完成。

2026-10-26 升降柱自碰撞局部微调实验
脚本统一在tests/refine_lift_spheres.py、rebuild_lift_patch.py、fit_shoulder_spheres.py、validate_lift_patch.py、view_lift_patch.py；产物tests/artifacts/lift_spheres/。
当前实际10cm抬升的误报来自body_base_link与l_base_link1/r_base_link1；平台本体不是本次直接碰撞对。
固定保存的抓取臂姿态扫描升降q=0..0.79m，原模型33个球对曾重叠，网格三角面检查对应姿态未相交；此检查不构成闭合实体包含关系证明。
可比较候选radius_only.yml仅缩小body_base_link的0基索引85、86、93、94、144、145，464球数量不变，ignore/buffer均不变。
200000个柱体表面采样覆盖99.442%→99.289%，新漏覆盖最大4.815mm；这是表面点到球并集的距离，不是体积覆盖指标。
GPU在实际抓取姿态上升10cm、1001个位置采样：自碰撞阳性376→0；新候选同时通过这段的机器人自/环境碰撞检查。
保存的5cm抬升、搬运、放置、抽离轨迹及相邻点中点，机器人碰撞复查全部通过；本轮未重算物体附着检查或PhysX。
候选仅解决上述局部升降范围，未解决全行程；原正式配置未替换，scene仍保持5cm抬升。
另测试局部移心补球和安装座分簇重建，未找到同时保持局部覆盖与全行程间隙的满意解，因此refined.yml/shoulder_fit.yml均为未采用试验。
对照界面http://127.0.0.1:8098/，可切换原模型/局部缩球候选、拖动0..10cm抬升、显示平台和其余柱体球。

2026-10-26 三排两列柱段重建及减球
脚本tests/rebuild_column_grid.py、validate_column_grid.py、view_column_grid.py；产物tests/artifacts/column_grid/。
仅替换body_base_link局部z=0.57..1.30m中心范围内的球，保留其余柱体、平台及安装座配置。
按截面X三排、Y两列生成每层6球；尝试10/12/15/18/24层后展示24层144球。
用户认为过密，现改为16层96球，减少三分之一；完整模型共510球（原正式464球，上一候选558球）。
16层柱球半径52.07..58.04mm，间距45.625mm；半径按网格点重新拟合，避免直接删层造成漏覆盖。
独立25万表面采样：直柱段覆盖98.2948%→99.9834%，body_base_link整体覆盖99.3756%→99.9876%；新暴露表面最大距离0.830mm。
仍有肩部安装座自碰撞，候选未替换正式配置；GPU结果见gpu_validation.json，不代表可执行模型。
8099界面显示16层版；橙色为柱段重建球、蓝色为安装座，绿色为原上下端/底座过渡球，紫色为原平台球。
绿色和紫色分别设显示开关，默认隐藏仅为查看清晰，碰撞计算中没有删除。

2026-10-26 平台后侧大球重建
scripts在tests/refine_platform_rear.py、validate_platform_rear.py；数据tests/artifacts/platform_rear/。
平台局部Y>=-0.001m、半径>45mm的16球替换为网格表面分区拟合球，保留其他球和所有ignore关系。
平台球数98→145，整机正式模型464→511；新增拟合球半径4.12..44.64mm。
独立25万点覆盖99.0936%→99.9916%，新漏覆盖最深0.073mm，平台后向最大包络0.149124m→0.121720m。
GPU复查保存的5cm抬升/搬运/放置/抽离均通过机器人碰撞检查；全行程原肩部—柱体误报未改善也未新增。
只将平台球应用于正式三份配置；16层柱体重建仍是候选，叠加新平台后共557球。
8100可切换修改前464球/平台修改后511球/柱体候选557球。备份位置见collision_models/current_model.json。

2026-10-26 用户明确要求排除body_base_link与l_base_link1/r_base_link1自碰撞对，双向ignore已同步正式三份配置及柱体候选；环境碰撞球保留。
复测tests/validate_shoulder_ignore.py：固定保存抓取臂姿态，0..0.79m的791个升降采样自碰撞阳性728→0；10cm抬升1001点通过机器人碰撞检查，四段已有搬运轨迹也通过。

2026-10-26 移动搬运续接
正式511球及用户指定的肩部—柱体ignore配置下，transfer.lift_height_m恢复0.10m。
重跑tests/test_table_transport.py，抬升1.817s、搬运9.525s（底盘2.25m）、放置1.754s、抽离3.961s均通过机器人及附着瓶子的采样检查。
完整回放tests/view_table_transport.py：http://127.0.0.1:8101/；串联先前预抓取/深入与本轮四段轨迹，总24.714s，数据tests/artifacts/transport/full_playback.npz。
独立yourdfpy正运动学审计tests/audit_table_transport.py：最大TCP位置差0.0106mm，旋转差0.000107rad；段间连续，瓶子刚性附着位置一致，目标在托盘内，X/Y余量87/52mm。
审计使用已有规划位姿容差判定，原始误差完整记录；不是同一实现自比对。
Isaac Sim环境独立释放测试tests/test_tray_release_physics.py：瓶子在目标托盘上模拟3秒，中心(1.2000005,1.3200009,0.8199999)，直立；只验证托盘支撑，不包含机器人夹持接触与闭环执行。
当前回放中夹持/释放是任务事件假设，夹爪没有闭合物理过程；完整PhysX机器人控制、夹爪摩擦抓持与释放反馈仍待实现。
