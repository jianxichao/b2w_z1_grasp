> 当前目标已切换为海绵，使用工作区根目录的 `sponge_best.pt`（类别 `sponge`）。腕部搜索配置见 `config/wrist_search.json`，海绵抓取入口为 `../unitree_z1/yolo/grasp_demo/demo_self.py`。历史易拉罐验证记录不代表海绵真机验收。

# B2W + Z1：搜索易拉罐并移到左/右1m

默认使用**Move速度指令 + Sport里程计闭环**：直接抬臂观察 → 最近3帧2票找到can → 停车取深度 → 转到机身坐标 → 移到左侧或右侧1m → 到位停车后趴下。Ctrl+C会先停车、机械臂复位后退出。当前不抓取，未加入自动避障。

详细流程、坐标示例、参数与源码见 [WRIST_SPORT_APPROACH.md](WRIST_SPORT_APPROACH.md)。新版尚未真机执行；已完成离线与模拟接口验证。

## 终端一：启动Sport桥和里程计

先让B2W站稳，保持z1_ctrl运行。本次更新必须重启底盘桥和搜索程序，旧桥不会热更新。若旧搜索/导航还运行，先Ctrl+C退出搜索并完成机械臂复位，再结束原导航launch；不要开两套底盘桥或机械臂SDK客户端。

```bash
cd /home/robot/unitree_ws/b2w_navigation
source setup.bash
ros2 launch b2w_hardware sport_position_bringup.launch.py
```

这一入口不启动SLAM/Nav2，本身不发运动指令。

## 终端二：运行搜索与左右接近

左侧：

```bash
cd /home/robot/unitree_ws/b2w_navigation
./scripts/run_wrist_search.sh --execute --approach --side left --view
```

右侧：

```bash
./scripts/run_wrist_search.sh --execute --approach --side right --view
```

X向前、Y向左。左侧目标为 `(can_x, can_y+1.0)`，右侧为 `(can_x, can_y-1.0)`；距离指底盘中心到目标水平距离，到位后面向易拉罐。先转向停车点，再低速前进，到点后面向易拉罐。默认前进≤0.30m/s、转向≤0.20rad/s；使用Move(1008)，不调用1036或1015。

不加 `--approach` 则只搜索、找到后保持；不加 `--execute` 则只预览相机。没有 `--explore` 时，检测前底盘静止，机械臂转腕扫描，检测后才使用Move速度闭环移动。

## 保留的探索与Nav2模式

需要原来的底盘探索时，终端一使用以下入口，终端二加 `--explore`。检测前仍用Frontier/Nav2，检测后的接近默认改用Move速度闭环。

```bash
source setup.bash
ros2 launch b2w_nav arm_camera_search.launch.py start_rviz:=true
```

原Nav2接近模式需明确加 `--approach-backend nav2 --explore`，操作见 [WRIST_APPROACH.md](WRIST_APPROACH.md)。原Nav2配置仍按其自身代价地图/轮廓规划；搜索脚本已移除Nav2的额外前置校验。

## 停止与说明

- Ctrl+C：停车、机械臂复位成功后自动退出；复位失败保持通信。复位中再次Ctrl+C可中断。
- 窗口q或 `/wrist_search/stop`：只停车停腕，保持通信。
- [WRIST_SEARCH.md](WRIST_SEARCH.md)：抬臂、扫描、投票与复位说明。
- [MAP_DIAGNOSIS.md](MAP_DIAGNOSIS.md)：Nav2模式的地图/代价地图查看。
- [VALIDATION.md](VALIDATION.md)：验证记录。

依赖位于`.runtime`，每个ROS终端加载 `source setup.bash`。安装外参、相机密码、观察点和接近参数位于config/wrist_search.json。


命令：
1：cd /home/robot/unitree_ws/b2w_navigation
source setup.bash
ros2 launch b2w_hardware sport_position_bringup.launch.py

2：cd /home/robot/unitree_ws/unitree_z1/z1_controller/build
./z1_ctrl

3：cd /home/robot/unitree_ws/b2w_navigation
./scripts/run_wrist_search.sh --execute --approach --approach-backend sport --grasp --side left --view

带`--grasp`时默认开启分段视觉更新，参数见`config/wrist_search.json`的`post_grasp.visual_servo`及[WRIST_GRASP.md](WRIST_GRASP.md)。每段约3cm，停稳后更新位置；中途丢目标直接用最后有效抓取点继续，不等待或重新投票。相机/机械臂通信故障仍中止。更新轨迹见`output/wrist_search/grasp_visual_trace.jsonl`。
