> 底盘单独导航参考。当前机械臂相机搜索请看 [WRIST_SEARCH.md](WRIST_SEARCH.md)。

# B2W 实机运行流程

新增当前任务入口：[机械臂相机边探索边搜索（第一步）](WRIST_SEARCH.md)。该模式取代内置视频搜索，发现海绵后只停车保持，暂不接近或抓取。

依赖已安装在本项目 `.runtime/`；不需要再安装。每个导航终端先运行：

```bash
cd /home/robot/unitree_ws/b2w_navigation
source setup.bash
```

当前采用官方 B2W 模型的机械安装坐标和官方标称尺寸，不需要手动填写这些数值。来源见 [参数记录](config/official/PROVENANCE.md)。这不是实物标定报告。

## 终端 1：海绵检测

```bash
cd /home/robot/unitree_ws/b2w_navigation
./scripts/run_can.sh --camera front --view
```

把海绵摆进前方画面。识别成功会显示 `sponge` 框和置信度，默认阈值 0.65。按 `q` 或 Ctrl+C 退出。后方相机用 `--camera back`，同时查看用 `--camera both`。

最新结果：`output/builtin/front_detections.jpg` 和 `front_detections.json`。取图失败时程序退出，不复用旧图。

这一步使用原生前后视频 API；不使用 Z1 相机。当前接口没有深度图/内参，**检测不会使机器人自动靠近**。

## 终端 2：雷达建图、定位与导航可视化

```bash
cd /home/robot/unitree_ws/b2w_navigation
source setup.bash
ros2 launch b2w_nav autonomous_mapping_sport_bringup.launch.py start_rviz:=true
```

启动后应看到地图、激光、机器人轮廓和轨迹。第一次地图主要覆盖当前雷达可见区域；移动后才会扩展。地图分辨率 0.05 m。启动本身不打开运动门。

若无图形界面，改用 `start_rviz:=false`，其余流程不变。

## 终端 3：预检、手动指定短距离导航目标

先用遥控器使机器人站稳、处于正常可运动模式，Z1 收拢到导航轮廓内，保持行驶空间空旷。程序不会自动站起、切换步态或移动机械臂。

```bash
cd /home/robot/unitree_ws/b2w_navigation
source setup.bash
python scripts/preflight.py --live --motion
```

出现 `PASS` 后打开两级运动门：

```bash
python scripts/control.py enable
```

在 RViz 点击 **Nav2 Goal**，先设置约 0.3–0.5 m 的短距离、前方无遮挡目标。

预期：机器人规划路径，以不超过 0.15 m/s 前进、0.20 rad/s 转向，到指定点后停车；地图/轨迹随之更新。这一步的目标来自你的 RViz 点击，与检测框没有自动关联。

软件停车：

```bash
python scripts/control.py stop
```

异常时直接用遥控器停车/锁轮。不要同时启动另一套 Go2W/RL/底盘控制程序。

## 可选：尝试一次自动探索

短距离导航与停车确认正常后，在两级运动门已打开时执行：

```bash
ros2 service call /frontier_explorer_node/enable std_srvs/srv/SetBool '{data: true}'
```

预期：Frontier 在已知/未知地图边界选一个可达候选点，交给 Nav2 行驶。默认只尝试 1 个目标，之后停止继续下发；失败或拒绝也计数。该流程目前不以海绵作为终点。

## 保存地图与结束

停止底盘后保存：

```bash
ros2 run nav2_map_server map_saver_cli -f "$PWD/output/b2w_map" \
  --ros-args -r map:=/navigation_map
```

结束顺序：先在终端 3 运行 `python scripts/control.py stop`，再在终端 2 Ctrl+C 结束导航，最后退出检测窗口。若程序异常，先用遥控器停车。

## “识别海绵后自动导航过去”还缺什么

复制的任务状态机已具备“搜索 → 稳定确认 → 计算目标 map 坐标 → 接近约 1 m → 停留 → 本轮任务结束”。它要求同一机身相机的：

```text
/b2w/front/color/image_raw
/b2w/front/aligned_depth_to_color/image_raw
/b2w/front/color/camera_info
```

当前 SDK 前后视频接口返回图像，官方 URDF 提供的是机械安装坐标，没有提供这两路图像的 CameraInfo、光学坐标系对应关系和原生深度流地址。不能用整机尺寸或机械安装坐标补出深度图。

因此当前已验证的是**原生视频检测 + 雷达建图/Nav2 就绪**，尚不能宣称自动接近海绵闭环已完成。接到真实深度/内参、确认光学坐标系后，才使用下列完整任务命令：

```bash
# 替代终端 2 的普通导航启动；仅在真实 RGB-D 输入已接通后使用。
ros2 launch b2w_nav autonomous_mapping_sport_bringup.launch.py \
  start_rviz:=true start_perception:=true start_supervisor:=true

# 在另一个已加载 setup.bash 的终端中：
python scripts/preflight.py --live --motion --target
python scripts/control.py enable
python scripts/control.py mission-start
```

最终预期：搜索过程中发现海绵，连续确认并通过深度定位，Nav2 绕开地图中的障碍，到海绵约 1 m 处停留后结束本轮任务。不抓取、不蹲起，首次任务也不自动返回基地。

## 安装/修复环境的命令

本机已完成，无需重复：

```bash
/usr/bin/python3 scripts/install_runtime.py
./scripts/build.sh
source setup.bash
python scripts/preflight.py
./scripts/test.sh
```

`.runtime` 是从本机官方 apt 软件源解析、下载并解压的二进制依赖，约 202 MB；只对加载此项目环境的终端生效，不修改 `/opt/ros` 或系统软件包数据库。安装清单在 `.runtime/manifest.json`。

若以后希望改成系统安装，可在你自己的交互终端执行 `./scripts/install_dependencies.sh`，输入本机 sudo 密码；当前运行不需要这一步。
