> 当前目标已切换为海绵，使用工作区根目录的 `sponge_best.pt`（类别 `sponge`）。腕部搜索配置见 `config/wrist_search.json`，海绵抓取入口为 `../unitree_z1/yolo/grasp_demo/demo_self.py`。历史易拉罐验证记录不代表海绵真机验收。

> 这是机械臂收拢时的底盘单独导航参考，不是当前机械臂相机搜索流程。当前流程请看 [README.md](README.md)。

# B2W 易拉罐检测与导航迁移

新增当前任务入口：[机械臂相机边探索边搜索（第一步）](WRIST_SEARCH.md)。该模式取代内置视频搜索，发现易拉罐后只停车保持，暂不接近或抓取。

从 `../rl2-challenge-cup-main` 独立复制导航、感知、硬件桥和接口包，改名为 `b2w_*`；原工程和 Z1 抓取程序未修改。模型直接使用 `../unitree_z1/yolo/best.pt`，已读取类别 `{0: can}`。

**依赖已安装，官方标称参数已填入；完整导航栈已完成真机静止验证。** Nav2 的 7 个核心节点均 active，已收到扫描、位姿并生成 5 cm 分辨率地图，两级运动门保持关闭。请直接按 [RUNBOOK.md](RUNBOOK.md) 分终端运行。

易拉罐检测可独立运行；“识别后自动靠近”仍缺内置相机的原生深度图、内参和光学坐标系对应关系。官方机械安装坐标不能代替这些输入。

## 1. 先测试易拉罐检测（不控制底盘）

```bash
cd /home/robot/unitree_ws/b2w_navigation
./scripts/run_can.sh --camera front --view
```

把易拉罐放在前方画面内，`q` 或 Ctrl+C 退出。后方用 `--camera back`，同时查看用 `--camera both`。只抓一轮：

```bash
../unitree_z1/yolo/.venv/bin/python scripts/detect_builtin.py --camera both --once
```

默认网卡 `eno1`；可以用 `--interface` 修改。结果保存到：

- `output/builtin/front.jpg`、`back.jpg`：原图；
- `front_detections.jpg`、`back_detections.jpg`：标注图；
- `front_detections.json`、`back_detections.json`：置信度、检测框和耗时。

入口仅调用 `front_videohub` / `back_videohub` 的取图 API 1001，不创建 SportClient，不发底盘指令。当前 API 返回图像字节，没有深度和内参；因此该入口只检测，不发布导航目标。取图失败会退出，不沿用旧图。

## 2. 本次实际检查结果（2026-10-27）

通过 `.164` 进行只读检查，并在本机 `eno1` 上测试 SDK：

| 输入 | 实测结果 |
| --- | --- |
| B2W 内置前后视频 API | 两路均成功取图，1920×1080；完成模型推理，本次画面未检出易拉罐 |
| `rt/sportmodestate` | 约 500 Hz，3 秒收到 1499 个有效状态；未发运动指令 |
| `/rslidar_points` | 10 秒收到 100 帧；frame=`rslidar`，含 xyz/intensity/ring/timestamp |
| `/dog_odom` | 10 秒收到 4989 帧；`odom -> robot_center` |
| LIO-SAM / 官方建图点云等 | 有话题名，但本次 10 秒订阅未收到数据，不能作为现成 SLAM 使用 |
| 原生 RGB / 深度 / CameraInfo | 本次未发现持续发布的数据 |
| `.164` USB RealSense | 序列号 `244222076021`，是之前 Z1 抓取使用的相机，本项目不打开它 |

原始传感器检查结果：`output/sensor_inventory_164.json`。机械臂相机搜索的 SSH 密码现按用户要求保存在 `config/wrist_search.json`。

前后视频的获取成功不代表已经开放内置 D435i 的原生深度流。需要宇树提供当前固件的深度图/CameraInfo 出口，或在相机所在机载电脑启动对应驱动；不要在 `.164` 启动不指定设备的 RealSense 驱动，以免接到 Z1 相机。

## 3. 从零建图的导航链路

默认不依赖 Go2W 的 `/unitree/slam_mapping/odom`，也不调用建图或内置导航 API：

```text
/lf/sportmodestate -> sport_odom_node -> /b2w/sport/odom + TF odom->base_footprint
/rslidar_points + 实测雷达安装TF -> pointcloud_to_laserscan -> /scan
/scan + odom TF -> slam_toolbox -> /navigation_map + TF map->odom
TF map->base_footprint -> map_pose_node -> /slam/current_pose + /slam/odom
地图 + 位姿 -> 原 Frontier / Nav2 -> /cmd_vel_nav
           -> cmd_vel_safety_node -> /cmd_vel_sport
           -> b2w_sport_cmd_bridge -> B2 Sport API 1008 / 1003
```

`/dog_odom` 已验证可读，但其 `robot_center` 高度/坐标定义还未核实，所以新链路用已明确字段的 SportModeState 重建平面 odom。运动前仍需用已知距离/转角检查精度；Sport 状态不是 SLAM。

原 Go2W 的官方 SLAM 桥保留为 `sensor_backend:=official` 可选分支，只有确认固件协议、数据坐标系和真实输入后才能使用。两条分支不能同时发布 map/base TF。

## 4. 安装依赖与编译

本机运行依赖已通过 `scripts/install_runtime.py` 安装到 `.runtime/`；包含 Nav2、slam_toolbox、Cyclone RMW 和点云转扫描等，共解析 99 个 Debian 包。无需 sudo，不修改系统安装目录。

```bash
source setup.bash
python scripts/preflight.py
```

若将来重建此项目的本地运行环境：

```bash
/usr/bin/python3 scripts/install_runtime.py
./scripts/build.sh
source setup.bash
```

Unitree 的 `unitree_api` / `unitree_go` 消息源码已放在 `src/vendor`，不需要再另外创建 Unitree ROS 工作区。来源、版本及适配说明见 `src/vendor/PROVENANCE.md`。

`setup.bash` 加载 Humble、本目录 overlay 和已有 YOLO Python 环境，指定 `eno1` 上的 CycloneDDS（domain 0）；它不启动机器人。网卡改变时，同时修改 `config/cyclonedds.xml` 与下述 robot 配置。其他终端也应加载它。

## 5. 已填写的官方参数

参考 `config/official/b2w_description.urdf` 和 [参数来源](config/official/PROVENANCE.md)：

- 官方站立尺寸：1.098 × 0.550 × 0.758 m；导航使用带余量的 1.30 × 0.80 m 包络，Z1 应收拢在此范围内。
- `base_link -> rslidar`：xyz=(0.34218,0,0.17851) m，rpy=(0,0,0)。
- 前后深度相机的机械安装坐标也已从 URDF 填入，并发布 `f_dc_link` / `r_dc_link` TF。
- `base_footprint -> base_link` 使用实时反馈的高度、roll、pitch。当前固件 `body_height=0`，平地测试回退使用 `position.z`，避免把整机总高度当作机身坐标原点高度。

这些参数按用户要求采用官方模型作为初值，不是实物标定。相机光学帧和原生图像内参未确认，感知的 `camera_extrinsics_verified` 仍为 false。

原始雷达时间戳与本机相差约 20 秒，已加入仅改时间戳、保留 `rslidar` 坐标系的本地桥。

## 6. 分阶段调试导航

先保持底盘停止，查看地图、扫描和 TF：

```bash
source setup.bash
ros2 launch b2w_nav autonomous_mapping_sport_bringup.launch.py start_rviz:=true
```

默认 `sensor_backend:=builtin`。建图自动启动；Sport 输出、速度安全门、Frontier 都关闭，任务监督器不启动。

另一终端：

```bash
cd /home/robot/unitree_ws/b2w_navigation
source setup.bash
python scripts/preflight.py --live --motion
ros2 run tf2_ros tf2_echo map base_footprint
ros2 topic echo /scan --once
```

预检通过且遥控器可随时停车后，由你手动放行两级运动门：

```bash
python scripts/control.py enable
```

先在 RViz 使用 Nav2 Goal 设置短距离目标，检查实际前进/转向/停车。默认上限：前进 0.15 m/s、转向 0.20 rad/s；横移、倒车和自动脱困关闭。初次自动探索最多尝试 1 个 Frontier 目标：

```bash
ros2 service call /frontier_explorer_node/enable std_srvs/srv/SetBool '{data: true}'
```

软件停车会依次终止任务/Frontier、关闭速度门和 Sport 门：

```bash
python scripts/control.py stop
```

异常时直接用遥控器停车/锁轮。不要同时运行原 Go2W 控制桥或另一套 RL/底盘控制程序。保存地图可用：

```bash
ros2 run nav2_map_server map_saver_cli -f "$PWD/output/b2w_map" --ros-args -r map:=/navigation_map
```

## 7. 接上内置深度后，测试导航接近易拉罐

感知节点已支持你的 `.pt` 模型，并保留原逻辑中的深度 ROI 测距、稳定确认、坐标平滑、目标记忆、Nav2 接近及任务状态机。接口约定为：

```text
/b2w/front/color/image_raw                  sensor_msgs/Image
/b2w/front/aligned_depth_to_color/image_raw sensor_msgs/Image
/b2w/front/color/camera_info                sensor_msgs/CameraInfo
```

先通过宇树当前固件的接口获得同一内置相机的 RGB、**对齐到 RGB 的深度**、内参。可以在 `d435i_target_perception.launch.py` 传 `color_topic` / `depth_topic` / `camera_info_topic` 使用实际话题。不能将 SDK 前视频和另一台 RealSense 的深度混用。

在 `b2w_robot.yaml` 填写固定机身相机的外参（偏移角为度，旋转按 `Rz*Ry*Rx*optical_axes`），核实后把感知节点和预检中的 `camera_extrinsics_verified` 都设 true。默认 false 时允许识别，但不会发布目标 map 坐标或 target_seen 事件。

```bash
ros2 launch b2w_nav autonomous_mapping_sport_bringup.launch.py \
  start_rviz:=true start_perception:=true start_supervisor:=true
```

确认可视化的目标坐标正确、深度与图像新鲜后：

```bash
python scripts/control.py enable
python scripts/control.py mission-start
```

首轮目标数为 1；到易拉罐约 1 m 处停留，随后任务空闲。不会蹲起或执行 Z1 抓取。自动搜索和接近只在明确开始任务后发生。可用 `python scripts/control.py stop` 终止。

## 8. 验证与目录

- `scripts/detect_builtin.py`：当前可直接使用的前/后视频检测入口。
- `native/`：只读 SDK 取图工具。
- `scripts/sensor_inventory.py`：在已加载 Humble 的机载电脑运行，订阅 10 秒传感器数据，不发运动指令。
- `src/b2w_nav/`：复制的 Nav2、Frontier、安全门、任务监督器和新建图 launch。
- `src/b2w_hardware/`：Sport 桥、平面 odom、TF/map 位姿适配，以及保留的官方 SLAM 桥。
- `src/b2w_perception/`：PT / ONNX 易拉罐感知。
- `scripts/test.sh`：测试在 domain 139、localhost 下运行，与真机 domain 0 隔离。

当前测试与硬件检查的结果记录在 `VALIDATION.md`。完整导航栈已完成静止验证，尚未完成底盘运动或易拉罐自动接近的闭环验收。
