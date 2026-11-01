# 查看 SLAM 地图与导航代价地图

可视化命令不会连接机械臂 SDK、发布导航目标或使能底盘。

已有导航栈正在运行时，另开终端：

```bash
cd /home/robot/unitree_ws/b2w_navigation
./scripts/view_navigation_map.sh
```

如果尚未运行 SLAM/Nav2，先启动原导航入口（本身不自动运动），已有一套则无需重复启动：

```bash
cd /home/robot/unitree_ws/b2w_navigation
source setup.bash
ros2 launch b2w_nav arm_camera_search.launch.py start_rviz:=false
```

诊断 RViz 的 Fixed Frame 为 map；只提供查看、选择、移动视角、测量工具。

| 左侧显示项 | 内容与用途 |
| --- | --- |
| 1 SLAM Map | /navigation_map；白色空闲、黑色占据、灰色未知。先单独勾选它查看原地图。 |
| 2 Global Costmap | /global_costmap/costmap；叠加全局规划实际使用的障碍及膨胀。对比原图与此图，检查中心是否落入高代价。 |
| 3 Local Costmap | /local_costmap/costmap；默认关闭，按需与全局代价地图切换查看。 |
| 4 Live Scan | /scan；橙红色扫描点。如果疑似自身障碍仍有实时扫描点对应，说明要检查建图输入，而不只是旧地图。 |
| 5 Robot Pose | /slam/current_pose；机器人位置及朝向。 |
| 6 Nav2 Footprint | /global_costmap/published_footprint；青色规划轮廓，当前为1.8m方形，不是B2W/Z1真实外形模型。 |
| 7 Raw Lidar (toggle for 3D) | /b2w/lidar/points；默认关闭。勾选后可在Views将TopDownOrtho换为Orbit，查看原始三维点是否贴着机身或机械臂。 |

先关闭两个Costmap只看原地图，再打开Global Costmap比较，避免彩色膨胀遮住黑色原始障碍。鼠标滚轮缩放，地图不在视野时调整Views的X/Y或选中Robot Pose后用Focus Camera。

当前无人发布地图时会显示No map received：检查SLAM导航终端是否仍运行、是否收到雷达和里程计；不要以此判断地图没有障碍。工具不会自动重启或清除旧地图。

## 保存可分享的快照

```bash
cd /home/robot/unitree_ws/b2w_navigation
source setup.bash
python scripts/snapshot_navigation_map.py
```

默认接收4秒话题和TF，保存：

- output/map_diagnosis/slam_map.png：完整SLAM图。
- output/map_diagnosis/comparison.png：机器人周围4m范围的原图、全局和局部代价地图并排对比。
- output/map_diagnosis/report.json：机器人中心格数值、最近占据格坐标/距离、栅格原点与分辨率。
- output/map_diagnosis/grids.npz：三张图的原始栅格数组（与report中的原点/分辨率配合读取）。

快照里的蓝色箭头为机器人位置/朝向，青框为Nav2规划轮廓，紫叉为该图最近的占据格。代价地图值99表示内切膨胀高代价区，100为致命障碍，负值为未知；原图0为空闲、100为占据。

排查顺序：

1. 原SLAM图机器人附近已有黑点：排查真实障碍、自身回波、历史残留或定位/坐标误差。
2. 原图中心空闲但全局中心99：附近障碍与当前轮廓/膨胀使起点不可通行；查看最近黑点的位置。
3. 全局与局部图差异明显：核对静态SLAM层与实时扫描障碍层的差别；实时轮廓清除不会直接抹去SLAM地图里的历史障碍。

保持底盘静止观察，不需要启动搜索/探索或发送Nav2目标。单张俯视图不能独立证明某个点来自机械臂；需再对照三维点云和机械臂实际位置。
