# RL2 GO2-W 自主巡检与操作系统

> 初赛完成整理版（2026-10-10）
> 本分支面向团队命令行复现，只保留当前主流程代码、运行配置、必要模型与部署脚本；

本目录包含一条经过真机验证的导航链路、一条只读目标感知支路，以及默认关闭的
ARX X5 机械臂支路：拓展坞运行宇树
`unitree_slam` 和 Mid360 驱动，API `1801` 持续建图并提供官方里程计；
用户电脑运行 Nav2 和 Frontier 探索，最终通过宇树 Sport API `1008`
控制 GO2-W，使用 API `1003` 停车。

```text
Mid360 + unitree_slam（拓展坞）
  -> /unitree/slam_mapping/points + /unitree/slam_mapping/odom
  -> 地图、TF、二维扫描桥接（用户电脑）
  -> Nav2 + Frontier Explorer
  -> /cmd_vel_nav
  -> cmd_vel_safety_node
  -> /cmd_vel_sport
  -> go2w_sport_cmd_bridge
  -> /api/sport/request（1008 Move / 1003 StopMove）

D435i（拓展坞 ROS 2 Foxy）
  -> RGB + 对齐深度 + CameraInfo
  -> rl2_perception（用户电脑 ONNX Runtime）
  -> 标注图 + 检测状态 + map 系目标位姿

Mission Supervisor（默认不启动）
  -> Frontier 搜索
  -> 目标候选确认
  -> 导航到目标附近并停留
  -> 继续搜索或返回起点
```

当前线路禁止同时启动 `rl_sar`，也不在 API `1801` 建图期间调用宇树
内置导航 API `1102`。

## 当前验证状态

- API `1801` 持续建图期间，API `1008` 能够控制机器人运动。
- 官方 odom、Nav2 单目标和约 `2.29m` 单 Frontier 闭环已完成真机验证。
- 机器人已完成转向、行驶和停车，无碰撞、异常抖动或明显不合理路线。
- 任务监督器已具备目标稳定确认、记忆坐标队列、近场视觉盲区接近、目标
  蹲起、失败换点和返回基地逻辑。
- 三目标人工装筐流程已完成真机演练：依次寻找并接近三个目标，目标处蹲起，
  完成后返回基地并打开储物筐。
- 命令行运维入口已集成栈启停、状态检查、任务启停、双运动门和储物筐控制，
  团队复现不依赖图形界面。
- ARX5 Action 与 Gemini 视觉抓取链保留在同一版本中；机械臂真机启动前仍须
  使用对应硬件的夹爪标定值并完成单独预检。

## 默认安全策略

主 launch 只启动节点，Sport 输出、速度安全门和 Frontier 均需手动放行。
未显式指定参数时：

- Frontier 每次放行最多尝试 1 个目标；
- 自动离墙、侧移和倒车脱困关闭；
- 普通导航与 Sport 桥均禁止倒车；
- 目标任务监督器不启动。

连续探索或自动恢复属于真机动作，必须先在锁轮状态下检查话题、TF、
代价地图和零速停车，再于空旷场地分项放行。

## 目录结构

```text
.
├── README.md
├── scripts/              # 部署、验证与受限真机测试脚本
├── deploy/               # 拓展坞部署文件
├── 舵机交接/             # 储物筐舵机控制程序与用户服务模板
└── src/
    ├── rl2_interfaces/   # 官方 SLAM 桥需要的自定义消息和服务
    ├── rl2_hardware/     # unitree_slam、地图、点云、健康和 Sport API 桥
    ├── robot_arm/        # ARX X5 规划、安全、真机 SDK 和仿真功能库
    ├── rl2_manipulation/ # 机械臂 Action、JointState、TF 和手眼视觉 ROS 接口
    ├── rl2_nav/          # Nav2、安全门控、Frontier、轨迹和 RViz
    └── rl2_perception/   # D435i + ONNX 目标识别、深度定位和标注图
```

## 环境要求

用户电脑：

- Ubuntu 22.04、ROS 2 Humble
- `rmw_cyclonedds_cpp`
- Nav2、RViz2、`pointcloud_to_laserscan`
- 已配置的 `~/learning/unitree_ros2/setup_go2w_humble.sh`
- 有线网卡地址位于 `192.168.123.0/24`，当前网卡名为 `enp59s0`

设备默认地址：

- `192.168.123.161`：机器人运控 PC
- `192.168.123.18`：拓展坞 PC
- `192.168.123.20`：Mid360

拓展坞需要安装并配置宇树 `unitree_slam`、Unitree SDK2、CycloneDDS 和
Mid360 驱动，雷达型号、YSN、外参及网卡名必须与实物一致。

安装用户电脑依赖：

```bash
sudo apt install \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-rviz2 \
  ros-humble-pointcloud-to-laserscan \
  python3-numpy python3-pytest
```

## 编译

项目统一使用工作区 `.venv`。`setup_local_humble.sh` 会自动激活它，再加载
Unitree、ROS 2 和当前安装空间；不要使用旧 Conda 环境或单独的 `pip`。

当前环境复用根用户环境中的 NumPy、ONNX Runtime 和 Orbbec SDK，视觉抓取使用
安装在 `.venv` 中的 CPU 版 PyTorch。依赖版本记录在
`requirements-workspace-venv.txt`。

真机在线导航感知使用 ONNX Runtime：

```bash
source ./setup_local_humble.sh
python3 -c "import numpy, onnxruntime, torch, ultralytics, pyorbbecsdk"
```

```bash
cd <仓库目录>

source ./setup_local_humble.sh
.venv/bin/python /usr/bin/colcon build --symlink-install \
  --packages-select rl2_interfaces robot_arm rl2_manipulation \
  rl2_hardware rl2_nav rl2_perception

```

编译完成后，新终端推荐统一使用根目录环境脚本（不会启动机器人）：

```bash
cd <仓库目录>
source ./setup_local_humble.sh
```

该脚本会兼容当前 shell 已开启 `set -u` 的情况，并将 ROS 日志定向到
项目所在的 learning 分区，避免继续占用已接近满载的系统根分区。

运行六个当前 ROS 包的测试：

```bash
colcon test --packages-select \
  rl2_interfaces robot_arm rl2_manipulation \
  rl2_hardware rl2_nav rl2_perception
colcon test-result --verbose
```

机械臂、机械臂视觉和底盘安装 TF 均默认关闭；只启动导航不会访问 `can1`。

## 拓展坞启动

进入拓展坞时不要选择 ROS 环境：

```bash
ssh unitree@192.168.123.18
pgrep -af 'unitree_slam|mid360_driver'
```

如果进程未运行：

```bash
cd /unitree/module/unitree_slam/bin

LD_LIBRARY_PATH="/home/unitree/cyclonedds_ws/install/cyclonedds/lib:/usr/local/lib" \
nohup ./unitree_slam >/tmp/unitree_slam.log 2>&1 &

sleep 3

LD_LIBRARY_PATH="/home/unitree/cyclonedds_ws/install/cyclonedds/lib:/usr/local/lib" \
nohup ./mid360_driver >/tmp/mid360_driver.log 2>&1 &
```

## 用户电脑快速启动

每个新终端都要加载：

```bash
cd <仓库目录>
source ./setup_local_humble.sh
```

确认没有另一套低层运控：

```bash
pgrep -af 'rl_real_go2|rl_sar|rl2_go2w_bridge'
```

启动完整链路和 RViz。运动门控默认关闭，不会因启动而自动运动：

```bash
ros2 launch rl2_nav autonomous_mapping_sport_bringup.launch.py \
  start_rviz:=true
```

相机驱动已在拓展坞部署好；需要目标识别时启动主链路时增加：

```bash
ros2 launch rl2_nav autonomous_mapping_sport_bringup.launch.py \
  start_rviz:=true start_perception:=true
```

感知默认关闭且不发布运动命令；外参未标定前不要用目标位姿触发自主运动。

### 命令行运维与三目标任务

推荐从统一命令行入口启动。启动栈不会自动打开运动门，也不会自动输出速度：

```bash
./scripts/rl2_stack.sh start
```

另开终端查看状态：

```bash
./scripts/rl2_stack.sh status --watch
```

三目标人工装筐协调器由主启动脚本拉起，也可单独启动：

```bash
./scripts/start_three_target_box_coordinator.sh
```

开始任务前必须确认 SLAM、目标感知、任务监督器、两级运动门和储物筐桥均为
健康状态。任务结束后协调器会等待返回基地完成，再调用储物筐打开服务。

确认机器人和场地安全后，使用原始 ROS 2 服务依次打开两级运动门并开始任务：

```bash
ros2 service call /go2w_sport_cmd_bridge/enable \
  std_srvs/srv/SetBool '{data: true}'
ros2 service call /cmd_vel_safety_node/enable \
  std_srvs/srv/SetBool '{data: true}'
ros2 service call /three_target_box_mission/start \
  std_srvs/srv/Trigger '{}'
```

正常终止、复位和重新开始：

```bash
ros2 service call /three_target_box_mission/abort std_srvs/srv/Trigger '{}'
ros2 service call /three_target_box_mission/reset std_srvs/srv/Trigger '{}'
ros2 service call /three_target_box_mission/start std_srvs/srv/Trigger '{}'
```

完全停止用户电脑上的运行栈：

```bash
./scripts/rl2_stack.sh kill
```

机器人异常运动时不要等待软件命令，应立即使用遥控器锁轮或急停。

储物筐 ROS 桥首次安装或仓库换目录后运行：

```bash
./scripts/start_storage_box_bridge.sh
./scripts/storage_box_control.sh probe
```

启动脚本会根据当前仓库绝对路径生成用户级 systemd 服务，不依赖固定的克隆
目录。开闭属于真实机构动作，只能在机构运动范围无障碍时执行。

两目标受控测试可显式设置：

```bash
ros2 launch rl2_nav autonomous_mapping_sport_bringup.launch.py \
  start_rviz:=true \
  frontier_max_goals:=2
```

`frontier_max_goals` 限制本次启用后最多下发多少次目标尝试；无论成功、
失败或拒绝都计数。默认值为 `1`，`0` 才表示不限制，真机测试阶段不要
使用 `0`。

Frontier 目标回调带有代次校验，过期结果不会抢占或清除新目标；失败后
等待固定重规划周期再换点。即使 `frontier_max_goals:=0`，默认连续失败
8 次也会发布 `frontier_failure_limit_reached` 并自动撤防。阈值可通过
`frontier_failure_stop_threshold` 调整；真机不得设为 `0`。

目标看门狗的 `goal_timeout_sec` 表示“连续没有取得位置进展”的时间，
不是单个目标允许的总运行时间。机器人每接近目标 `0.08m`（参数
`goal_progress_distance_m`）就刷新看门狗，因此远距离 Frontier 在持续
移动时不会被固定 35 秒总时长误杀。

Frontier 在下发前还会用 `/global_costmap/costmap` 做一次可达性检查，排除
被实时障碍与膨胀层隔断的候选。建图期间全局路径以 `1Hz` 更新；规划失败
时清理代价地图后快速换点，不再执行约 10 秒的原地 Spin。

若机器人真实起点栅格为致命/不可通行状态，Frontier 不再用附近自由栅格
冒充起点，也不会累计无意义的失败次数；它会发布
`frontier_waiting_for_reachable_goal` 并等待代价地图恢复。

由于 Frontier 位于已知/未知边界，候选的原始栅格可能被全局代价地图标为
未知或膨胀区。节点会把候选投影到 `0.50m` 内最近的连通自由栅格，并将
投影后的真实可达坐标发送给 Nav2；找不到投影点时才继续等待地图更新。

实体墙会自然形成边界。如果场地边界只是地面标线或人为规定，可在启动时
设置地图坐标系下的矩形地理围栏：

```bash
ros2 launch rl2_nav autonomous_mapping_sport_bringup.launch.py \
  start_rviz:=true frontier_max_goals:=0 \
  exploration_min_x:=-5.0 exploration_max_x:=5.0 \
  exploration_min_y:=-4.0 exploration_max_y:=4.0
```

四个边界参数默认均为 `0.0`，表示不启用地理围栏。

## GO2-W 横移

导航速度链已支持 `Twist.linear.y`：DWB 横向采样、安全层限幅/斜坡以及
Sport API 1008 的 `y` 字段均已接通。真机标定结果为 `y>0` 向左、`y<0`
向右；横移上限为 `±0.50m/s`，最小有效目标为 `0.25m/s`，横向加速度
上限为 `0.40m/s²`。倒退仍保持禁用（负 `linear.x` 会被截为零）。

真机低速横移已验证平稳；每次修改横移参数后仍应关闭 Frontier 做短时
正负方向测试，确认停车与避障后才可恢复自动探索。

地图桥会在每次更新后清除机器人自身矩形轮廓（半长 `0.34m`、半宽
`0.22m`）内的静态点，避免自反射把起点写成障碍。可选的单侧墙脱困默认
关闭；受控验收后通过 `frontier_auto_wall_escape:=true` 开启。它仅在雷达
确认一侧小于 `0.55m`、另一侧至少 `0.80m` 时，以 `0.25m/s` 横移最多
`1.2s`，每次受阻最多尝试两次，不会自动倒退。

## 开始建图和健康检查

```bash
ros2 service call /unitree_slam/start_mapping std_srvs/srv/Trigger '{}'

ros2 topic echo /rl2_nav/slam_health --once
timeout 10s ros2 topic hz /unitree/slam_mapping/points --wall-time
timeout 10s ros2 topic hz /unitree/slam_mapping/odom --wall-time
timeout 10s ros2 topic hz /slam/odom --wall-time
timeout 10s ros2 topic hz /scan --wall-time
timeout 5s ros2 run tf2_ros tf2_echo map base_footprint
```

必须确认点云、官方 odom、桥接 odom、扫描和 TF 都持续更新，才能启用
运动。

## 启用与停止运动

机器人保持锁轮时，依次启用两级运动门；确认零速度状态后再解锁：

```bash
ros2 service call /go2w_sport_cmd_bridge/enable \
  std_srvs/srv/SetBool '{data: true}'

ros2 service call /cmd_vel_safety_node/enable \
  std_srvs/srv/SetBool '{data: true}'
```

场地安全后启用一次 Frontier：

```bash
ros2 service call /frontier_explorer_node/enable \
  std_srvs/srv/SetBool '{data: true}'
```

正常停车必须从上游向下游关闭：

```bash
ros2 service call /frontier_explorer_node/enable \
  std_srvs/srv/SetBool '{data: false}'

ros2 service call /cmd_vel_safety_node/enable \
  std_srvs/srv/SetBool '{data: false}'

ros2 service call /go2w_sport_cmd_bridge/enable \
  std_srvs/srv/SetBool '{data: false}'
```

出现异常运动时不要等待软件命令，立即使用遥控器锁轮/急停。

## RViz 与轨迹

完整 launch 已提供 RViz。链路已经运行时也可单独打开：

```bash
rviz2 -d \
  <仓库目录>/install/rl2_nav/share/rl2_nav/rviz/navigation_only.rviz
```

清空本次实际运动轨迹：

```bash
ros2 service call /odom_path_node/reset std_srvs/srv/Trigger '{}'
```

## 保存地图

机器人停止并锁轮后：

```bash
ros2 service call /unitree_slam/end_mapping \
  rl2_interfaces/srv/UnitreeEndMapping \
  "{address: '/home/unitree/rl2_sport_mapping.pcd'}"
```

等待 `/unitree_slam/command_status` 出现 `api_id:1802` 和
`Save pcd successfully`。

跨重启复用同一地图和基地/洞口坐标时，必须使用 API 1804 重定位专用链路；不要重新执行
1801 后直接复用旧坐标。
