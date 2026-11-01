# Nav2模式：腕部相机搜索后导航到海绵附近

当前默认已切换为Sport Move速度闭环左右0.5m接近，见[WRIST_SPORT_APPROACH.md](WRIST_SPORT_APPROACH.md)。本页仅说明明确选择 `--approach-backend nav2` 时的旧Nav2模式。

第一阶段搜索已由用户完成真机验证。本页接上第二阶段：3帧2票确认 sponge → 停车停腕并取得有效深度 → 转换地图目标 → Nav2 接近 → 到位停车保持。代码和离线检查已完成，新增接近流程尚待真机验收；不执行抓取、夹爪动作或自动返回。

## 运行

B2W 先用遥控器站稳、保持静止，保持已有 `z1_ctrl` 运行。一个机械臂只能有一个 SDK 控制进程；旧搜索进程即使显示 FOUND/FAULT 仍占有控制，按原来的停车、支撑机械臂、释放控制流程结束后再启动新版。不要直接再开第二个客户端。

终端一（与第一阶段相同；已经运行这一套导航时无需重复启动）：

```bash
cd /home/robot/unitree_ws/b2w_navigation
source setup.bash
ros2 launch b2w_nav arm_camera_search.launch.py start_rviz:=true
```

终端二，启动搜索与接近：

```bash
cd /home/robot/unitree_ws/b2w_navigation
./scripts/run_wrist_search.sh --execute --explore --approach --approach-backend nav2 --view
```

不加 `--approach` 就保持第一阶段行为：找到后停车观察。`--approach` 必须同时带 `--execute --explore`；命令行帮助不会连接硬件：

```bash
./scripts/run_wrist_search.sh --help
```

## 动作与状态

1. 与第一阶段一样停车，从机械臂当前姿态直接抬到观察点（启动不复位），没找到就 Frontier 探索并慢速转腕。
2. 搜索时对最近3个新检测帧做滚动投票，can 置信度达到0.65计1票，漏检计0票；累计2票就停车停腕，不要求连续，也不等待满10帧。已移除 VERIFYING 及其速度/静止计时/连续命中要求。
3. 进入 `TARGET_LOCALIZING`，保持机械臂，用停车后的一帧新曝光有效深度、当前关节姿态和历史 TF 定位。只要求底盘实测已停止，不再要求三帧深度或额外的连续静止计时。暂时没有深度继续等待；默认10秒仍无有效深度则报告 `target_found_but_depth_unavailable`，保持 FOUND 停车，不误报 FAULT。有效深度到来后转换地图坐标，检查地图/TF新鲜度及导航状态，沿机器人到目标的方向计算停靠点，失败才进入 FAULT。
4. `APPROACH_PLANNING`：发送 Nav2 目标，等待接受。接近阶段关闭 Frontier，只在 Nav2 接受后打开两级底盘运动门，进入 `APPROACHING`。沿途可以为绕障转向；机械臂不继续扫描。
5. Nav2 成功后先关闭底盘输出，进入 `APPROACH_SETTLING`。核对实际停靠点误差、距目标距离、朝向，并确认底盘连续静止 0.5秒，才报告 `ARRIVED`。如果目标原本就在停靠半径内，直接报告 `target_already_near_no_motion` / ARRIVED，保持原地，不绕行或后退到圆环上。
6. 导航拒绝/失败、超时、定位突变、地图/TF/相机/导航传感器陈旧、机械臂反馈断开或保持姿态明显改变，都会锁定 FAULT、请求取消导航并关闭底盘输出，不自动重试或恢复。

`approach_enabled=true` 表示本次选用了接近功能，不代表底盘正在动；看 `state` 和 `base_output_enabled`。`grasp_enabled` 始终为 false。ARRIVED 之后继续保持机械臂 SDK 通信。`vote_hits/vote_frames` 显示投票进度，`fault_reason` 显示整体故障原因。单帧漏检不触发 FAULT；真实 FAULT 锁定后，即使画面再次识别到 sponge 也不会自动继续行驶，需处理故障并受控重启。

## 外参与停靠参数

`config/wrist_search.json` 的 `approach` 段使用用户提供的安装变换，解释为机械臂底座坐标到 B2W **机身 `base_link`** 坐标的变换；不是地面投影 `base_footprint`：

```text
T_base_arm =
[1 0 0 0.078  ]
[0 1 0 0      ]
[0 0 1 0.09602]
[0 0 0 1      ]

p_map = T_map_base_link × T_base_arm × FK(q) × T_end_camera × p_camera
```

`T_end_camera` 复用 `unitree_z1/yolo/grasp_demo/config.json`；FK 使用当前 SDK 模型和确认时实测关节角。`T_map_base_link` 从 TF 历史读取，包括机身高度和姿态，避免把安装高度直接加到地面坐标。曝光时间采用本机采集起始时间估计，因此目标确认要求底盘和机械臂已停止；本阶段不在移动过程中重估目标。搜索投票允许机身与机械臂运动，投票通过后才停车并取一帧定位；不会将运动期间多个相机坐标直接平均。

- `standoff_m: 1.5`：**底盘中心**距离目标水平位置 1.5m，不是机身外沿间隙，也不是抓取距离。到位距离/位置容差默认 0.20m。
- `timeout_sec: 90`：整个接近阶段上限。底盘仍沿用专用 Nav2 配置的最大前进速度 0.10m/s。
- `depth_wait_timeout_sec: 10`：投票通过、停车后等待一帧有效深度的最长时间。投票本身不受深度影响。
- `tf_lookup_wait_sec: 0.3`：曝光时刻比最新 TF 略晚时，保持停车，最多等待0.3秒并处理 TF 回调，再查询同一个曝光时间。正常的几十毫秒延迟不再立即触发 FAULT；超时或实际数据陈旧仍报错。等待期间 Ctrl+C/stop 仍能中断，不用旧姿态替代历史变换。
- `map_timeout_sec: 3`，`tf_timeout_sec: 0.5`：地图接收和当前 TF 新鲜度上限；历史曝光 TF 允许覆盖有效曝光时长。ROS 节点应继续使用这套导航的统一时钟。

已按用户要求移除接近前的24方向候选搜索及1.30m空闲圆检查，不再因停靠点周围有未知/占据格而在本脚本提前拒绝。现在复用 `direct_approach_goal_xy()`，沿当前底盘中心到目标的方向，计算距目标1.5m的停靠点，朝向目标后发送 Nav2。机器人实际行驶路径由 Nav2 规划，可绕障；目标或路径不可达仍由 Nav2 报错并停车。地图/TF新鲜度监测、Nav2的1.80m轮廓、代价地图和避障继续生效。

## 停止与查看记录

终端 **Ctrl+C** 会取消导航、确认底盘停车、机械臂低速复位到 startFlat，成功后自动结束通信并退出。复位中再次 Ctrl+C 会中断并保持；复位失败保留 FAULT，不强制退出。

窗口按 `q` 或另一个终端请求停止，只停车停腕并保持通信：

```bash
cd /home/robot/unitree_ws/b2w_navigation
source setup.bash
ros2 service call /wrist_search/stop std_srvs/srv/Trigger '{}'
```

只停止后可回搜索终端按 Ctrl+C，复位并自动退出。若要直接手动结束通信，先将机械臂放稳/支撑，再调用：

```bash
ros2 service call /wrist_search/release_control std_srvs/srv/Trigger '{}'
```

- `/wrist_search/status`、`output/wrist_search/events.jsonl`：状态、停车请求、目标提交、接近和到位确认。
- `/wrist_search/target_pose`：冻结的海绵地图坐标，供 RViz 检查。
- `output/wrist_search/approach_target.json`：确认深度点、实测关节角、完整变换链、地图目标、停靠目标及距离配置。
- `output/wrist_search/nav2_result.json`：Nav2原始结果状态，区分 `SUCCEEDED=4`、`CANCELED=5`、`ABORTED=6`。当前Humble接口结果载荷只有Empty，规划失败的具体原因仍以planner/controller日志为准。
- `output/wrist_search/found.json`：本次确认结果；导航目标发布成功后更新地图目标信息。

第一次使用新增功能时重点核对记录中的地图目标方向、RViz 目标落点、实际停靠距离以及停止命令，再开展抓取阶段。

## 源码

| 功能 | 位置 |
| --- | --- |
| 搜索确认衔接、CLI、运动门和停止 | `scripts/search_with_wrist.py` |
| 深度定位、TF、停靠点、Nav2 与到位检查 | `scripts/wrist_approach.py` |
| 刚体变换与三维目标转换 | `src/b2w_perception/b2w_perception/wrist_approach_core.py` |
| 复用的直线方向停靠点计算 | `src/b2w_nav/b2w_nav/mission_supervisor_core.py`：`direct_approach_goal_xy()` |
| 外参与接近参数 | `config/wrist_search.json` 的 `approach` |
