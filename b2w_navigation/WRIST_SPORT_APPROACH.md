# 检测海绵后，用 Move 速度闭环移到左/右1m

默认 `--approach` 使用自编的里程计点位控制器，持续调用 `Move(vx, vy, yaw_rate)`（API1008），不再调用固件未实现的MoveToPos(1036)，也不调用SpeedLevel(1015)。接近不依赖Nav2、SLAM地图或TF。本版已完成离线运动仿真和隔离ROS接口测试，尚未执行新版真机动作；默认不抓取、未加入绕障。增加 `--grasp` 可接入到位后的侧向扫描、抓取和归位，见 [WRIST_GRASP.md](WRIST_GRASP.md)。

## 坐标与动作

X向前、Y向左、Z向上；左右按定位时的机身坐标轴定义：

```text
p_sponge_base = T_base_arm × FK(q) × T_end_camera × p_sponge_camera

左侧目标 = (sponge_x, sponge_y + 1.0)
右侧目标 = (sponge_x, sponge_y - 1.0)
```

这里是**底盘中心**到海绵的水平侧向距离。左右两侧停车的最终朝向均为局部yaw=0，即保持开始接近时的机身朝向，侧对海绵，方便后续机械臂抓取。停在海绵左侧时海绵位于机身右侧；停在右侧时海绵位于机身左侧。左右与朝向均使用冻结的开始接近时坐标轴。

到位下蹲时，在上述理想侧向点基础上仅将指令目标X前移 `approach.stand_down_forward_compensation_m=0.07` 米。该值来自最近四次记录的下蹲后移6.9、7.0、6.9、7.8cm，可按现场实测调整，未改里程计反馈。`sport_side_goal.json` 同时记录 `ideal_goal_base_xy_m` 和带补偿的 `goal_base_xy_m`；关闭到位下蹲时不应用此补偿。

例如海绵在机身坐标 `(1.5, 0.1, z)m`，左侧停车点为 `(1.5, 1.1)m`，右侧为 `(1.5, -0.9)m`。

安装外参沿用用户提供的 base_link→机械臂底座平移 `(0.078,0,0.09602)m`、R=I，手眼外参沿用 grasp_demo/config.json，FK使用实时SDK关节角。

1. 启动不复位，直接到观察点；最近10帧5票确认sponge。
2. 停车停腕后用一帧有效深度计算上述局部目标。
3. 关闭其他运动输出，冻结开始接近时的Sport位置/yaw为锚点；把后续里程计转换回这个固定局部坐标系。目标只提交一次，控制器持续计算速度。
4. **先转向目标停车点 → 前进并减速 → 到点后恢复初始机身朝向、侧对海绵 → 停车**。每秒10次调用Move，vy=0，不倒车；默认前进速度≤0.30m/s、角速度≤0.20rad/s，并限制速度变化率为0.15m/s²、0.40rad/s²。这是直线参考点位跟踪，不是障碍路径规划。
5. 默认同时满足停车点0.25m范围、前后误差≤0.06m、最终朝向误差≤0.10rad（约6°）才锁定调整并StopMove；转向过程中继续检查位置，不能仅凭进入宽松半径提前停车。实测线/角速度均低于0.05并连续0.3秒后，只发一次StandDown(1005)。发出StandDown后等待3秒，实测线/角速度低于0.03并连续稳定0.3秒后报告ARRIVED；无需API成功回复；不再补发会打断趴下的StopMove。左右1m为名义目标，允许上述停车误差及制动偏移。
6. Ctrl+C先停车，再机械臂复位并退出，与前一版一致。q/stop只停车保持。

## 推荐运行：不启动Nav2、不移动探索

先结束旧搜索程序（Ctrl+C复位退出），再在原导航launch终端结束旧导航。不能同时运行两套Sport桥。让B2W站稳、保持已有z1_ctrl运行。

终端一，只启动Sport里程计与底盘桥（启动本身不发送运动命令）：

```bash
cd /home/robot/unitree_ws/b2w_navigation
source setup.bash
ros2 launch b2w_hardware sport_position_bringup.launch.py
```

终端二，左侧1m：

```bash
cd /home/robot/unitree_ws/b2w_navigation
./scripts/run_wrist_search.sh --execute --approach --side left --view
```

右侧1m：

```bash
./scripts/run_wrist_search.sh --execute --approach --side right --view
```

没有 `--explore` 时，检测前底盘静止，只有机械臂扫描。找到并定位后才执行Move速度闭环。

如果还要原来的底盘探索，终端一改为更新后的 `ros2 launch b2w_nav arm_camera_search.launch.py start_rviz:=true`，终端二加 `--explore`。此时**搜索探索仍用原Frontier/Nav2，找到后的接近使用Move速度闭环**；本次没有重写探索算法。旧版正在运行的桥不会热更新。本次更换了局部目标服务和底层控制器，必须先退出搜索，再重启底盘桥launch和搜索两个进程，不能只重启搜索。

要明确使用原Nav2接近模式，可加 `--approach-backend nav2 --explore`，说明见WRIST_APPROACH.md。

## 约束与反馈

速度参数在 `src/b2w_hardware/config/b2w_sport_bridge.yaml` 的 `local_goal_*` 四项中。控制器另有最大命令速度边界0.30m/s、0.30rad/s；默认使用0.30m/s、0.20rad/s。修改后重启桥。Twist门关闭时，该局部目标控制器通过独立服务拥有API输出。

本实现**未增加自动绕障或雷达碰撞检测**。它只按目标方向生成转向/前进速度，障碍物不会触发绕行。`--explore` 仍用已有Frontier/Nav2；此次只更换检测后的接近控制。

底盘桥保留独立看门狗：客户端心跳或Sport反馈超过0.5秒、控制调度中断超过0.5秒、动作超过120秒、实测速度超过0.5m/s、连续10秒无实测进展、其他速度控制重新启用或收到Move错误响应，均StopMove。Move的成功回复不是到位依据，也不要求每帧都有成功回复。进程被杀后，仍运行的桥会因心跳丢失停车。

- `SPORT_POSITION_ACCEPTING`：等待本机底盘桥接受局部目标。
- `SPORT_POSITION_MOVING`：软件控制器正在闭环跟踪目标。
- `ARRIVED`：已到位停车，发出趴下指令后完成3秒等待且底盘静止；API成功回复不是前提，物理趴下姿态未独立验证。
- `FAULT`：反馈、Move返回或看门狗异常，停车保持。
- `base_command_source=sport_velocity_goal`：软件点位控制拥有运动输出；其他Twist门保持关闭。

记录：

- `output/wrist_search/sport_side_goal.json`：相机坐标、完整变换链、机身坐标海绵位置、左/右目标与相对yaw。
- `output/wrist_search/sport_position_result.json`：软件控制状态/返回故障或到位信息。
- `/b2w_sport_cmd_bridge/position_status`：位置控制状态、控制阶段、当前速度、距离/yaw误差、command_id和起始Sport锚点。

参数在config/wrist_search.json的approach段：backend、side、side_offset_m、sport_max_distance_m。

## 源码

- `scripts/wrist_sport_approach.py`：搜索到Sport服务的衔接、关停其他运动源、保持心跳和到位结果。
- `src/b2w_perception/b2w_perception/wrist_approach_core.py: local_side_goal()`：相机到机身变换及左右偏移。
- `src/b2w_hardware/b2w_hardware/sport_position.py`：固定局部坐标、转向/直线速度计算、加速度限制、到位监测和独立看门狗。
- `src/b2w_hardware/b2w_hardware/b2w_sport_cmd_bridge.py`：ROS服务与宇树API请求/响应。
- `src/b2w_interfaces/srv/SportLocalGoal.srv`：局部位置请求。

## `--explore` 启动时缺少传感器

`--explore` 仍表示运行已有的Frontier/Nav2探索，并非由Move控制器生成探索目标。它需要 `/scan`、`/b2w/sport/odom` 及完整探索服务；精简 `sport_position_bringup.launch.py` 不提供雷达扫描和探索。

- 只验证检测后接近：使用精简launch，搜索命令去掉 `--explore`。
- 需要边走边搜索：先退出搜索、结束精简launch，再启动 `ros2 launch b2w_nav arm_camera_search.launch.py start_rviz:=true`，搜索命令加 `--explore`。找到后仍切换为Move闭环接近。

准备阶段最多等待5秒收齐新鲜传感器数据，并要求底盘连续静止0.5秒；超时会明确报告缺失/过期的topic。`/scan` 只有收到至少一个有限距离值才视为有效，因此全inf/NaN扫描也会报告无有效扫描。该阶段不启动机械臂SDK或抬臂动作。

## 底盘只动一下：区分指令与执行

Move必须持续发送；本机宇树B2W示例说明0.5秒未收到新Move会自动停止（unitree_sdk2/example/b2w/b2w_sport_client.cpp）。本控制器在MOVING状态默认每0.1秒发送一次API1008，并非只提交一条速度指令。

每秒heartbeat现在附带：`move_command_count`（本目标累计Move发布次数）、`controller_phase`、`commanded_vx_m_s`、`commanded_yaw_rate_rad_s`、`measured_speed_m_s`、`distance_error_m`及`measured_local_xy_yaw`。发布次数增长只证明本机持续发布，不保证固件实际执行；若次数增长、命令非零，而实测速度/位移接近零，应排查底盘模式、命令覆盖或反馈，不能仅删除无进展检查或提高速度。若计数停止，查看桥状态/FAULT原因。

收到的接近状态逐条保存到 `output/wrist_search/sport_position_trace.jsonl`，含接收单调时钟与command_id，跨运行应按command_id区分。旧 `sport_position_result.json` 在下一次终态更新前可能保留上一次目标结果，不能直接当作本轮故障。新增日志需要重启底盘桥和搜索两个进程才能完整生效。


## 前进与转向死区补偿

`local_goal_linear_bias: 0.20`（m/s）与 `local_goal_angular_bias: 0.20`（rad/s）分别用于前进与转向；两者独立可调，设0可关闭。当前是按用户指定的假设死区进行补偿，尚未标定真实硬件死区。

控制器先根据位置/角度误差生成原始速度，再进行：非零指令加同方向偏置 → 按local_goal_linear_speed/local_goal_angular_speed限幅 → 加速度限制 → Move。原始指令为0时，补偿结果仍为0；vy始终为0。例：角速度0.198+0.20=0.398rad/s，在当前0.30rad/s上限下输出目标为0.30rad/s；前进0.06+0.20=0.26m/s。启动、换向与停车仍经过已有加速度限制，瞬时指令可能尚未超过假设死区。

原地转向、途中转向和到点恢复初始朝向都应用角速度补偿。途中朝向误差≤0.02rad时不修正转向，避免微小误差被放大；最终朝向使用0.10rad到位容差。停车、取消与FAULT均保持零输出/StopMove，未偏置反馈数据、到位容差或相机/机械臂参数。该补偿用于检测后的Sport闭环接近；原Frontier/Nav2探索控制未增加补偿。

新增ROS参数需要重启底盘桥及搜索程序。本次补偿未进行真机测试。

## 确认补偿代码已加载

底盘桥常驻进程不会随着Python源文件或YAML修改热更新。新桥启动即在position_status发布 `controller_version=5` 和两项实际加载的偏置，且此状态发布不产生运动。接近控制器在交接前验证这些字段；若连接的是旧桥，会提示 `deadzone compensation is not loaded` 并要求重启桥，避免静默使用旧算法。

运动heartbeat也报告 `controller_version`、`linear_deadzone_bias_m_s`、`angular_deadzone_bias_rad_s`；当前预期为5、0.2、0.2。这些值来自桥实际运行实例。相机中无海绵时投票保持0，底盘按不探索模式保持静止。


## 到位停车后趴下

YAML参数：`local_goal_position_tolerance: 0.25`（m）、`local_goal_yaw_tolerance: 0.10`（rad）、`local_goal_longitudinal_tolerance: 0.06`（m）、`local_goal_stand_down_on_arrival: true`。位置半径、独立前后误差、朝向三项同时通过后才锁定调整；转向时离开位置范围会继续对齐。控制阶段为 `STOPPING_AT_GOAL → STAND_DOWN_WAIT → ARRIVED`，期间仍保持客户端心跳和里程计监测。趴下请求明确拒绝、下蹲前5秒不能停稳或发出下蹲后8秒仍不能静止均FAULT；无回复不触发故障，不重试位置或姿态命令。普通Twist输出全程关闭，StandDown只由本次位置目标拥有，不需重新打开速度门。

`stand_down_acknowledged=true`表示固件接口接受，`physical_posture_verified=false`明确表示尚未通过姿态传感器确认已经完全趴下；`stand_down_wait_completed=true`表示发出下蹲指令后3秒已到且底盘静止；定时等待不是姿态测量。带`--grasp`时机械臂随后开始侧向搜索与抓取，否则继续保持观察姿态，Ctrl+C仍先停车、机械臂复位退出。此次修改未进行真机趴下测试。版本为5，需先结束旧搜索并复位，再重启底盘桥和搜索程序。
