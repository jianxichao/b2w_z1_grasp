# 第一步：机械臂相机搜索海绵

本页说明不带 `--approach` 的第一阶段。搜索后接近目标请看 [WRIST_APPROACH.md](WRIST_APPROACH.md)。

本阶段只做：**没找到 → 底盘探索与机械臂转腕同时进行；最近10帧有5帧检测到 sponge → 停底盘、停转腕并保持观察。** 不走到海绵旁边，不抓取，不开合夹爪，不自动返回基地。

## 启动时速度读数偏高

启动已改为只读位置窗口，不再凭一帧 dq 判断机械臂在运动。连接 SDK 后采样关节角，完整0.6秒内每个关节角极差 <0.003rad（约0.17°），且采样连续，就开始规划观察姿态。速度读数仍记录；持续明显位移、反馈/状态变化、非有限反馈或取消请求仍拒绝启动。最多等待3秒，尚未通过时不会请求关节保持或使能运动。

参数在 `wrist.startup_position_window_sec`、`startup_position_range_rad`、`startup_still_timeout_sec`。诊断文件为 `output/wrist_search/startup_status.json` 和 `startup_settle.jsonl`；它检查静止，不执行启动复位。Ctrl+C退出复位流程不变。

## 先预览机械臂相机

```bash
cd /home/robot/unitree_ws/b2w_navigation
./scripts/run_wrist_search.sh --view
```

已按用户要求在 `config/wrist_search.json` 的 `camera.password` 配置 `.164` SSH 密码，正常运行不再询问。环境变量 `B2W_CAMERA_PASSWORD` 可以覆盖配置。使用序列号 `244222076021` 的 RealSense，读取对齐的 RGB、深度和相机内参，模型使用工作区根目录的 `sponge_best.pt`。该命令不创建 Z1 SDK 客户端、不控制底盘。`q` 或 Ctrl+C 退出。

无图形界面时去掉 `--view`。短时取流检查可加 `--frames 2`。

## 机械臂姿态与控制器

带 `--execute` 时，启动直接从当前实测姿态规划到观察点，不复位。复位移到 Ctrl+C 退出流程；纯相机预览不复位、不使能机械臂。

联动探索要求 **B2W 已由遥控器站稳、处于正常可运动模式并保持静止**。不要让狗趴着运行 `--explore`，也不要在机械臂动作时才让狗站起；程序不会自动 StandUp。纯相机预览可以趴着。

启动顺序：

1. 联动模式先关闭底盘运动门，并通过反馈确认底盘静止。
2. 读取当前实测关节角，保持夹爪角度；不执行 startFlat 复位。
3. 直接从当前姿态，在机械臂底座坐标系的 SDK **末端位置** `(0.20, 0, 0.30) m` 附近搜索 IK，邻域半径最多 0.04 m，优先向下俯视 15°。
4. 验证 IK 的 FK、整段关节插值路径和到位后完整 J5 ±15° 扫描，再开始抬臂。夹爪角度不变。
5. 以不超过 0.10 rad/s 的关节速度平滑抬臂，等实测角度、速度稳定后才进入搜索；退出复位速度上限为 0.08 rad/s。

上述位置是 SDK 末端，不是相机光心；两者由已有安装关系区分。若当前姿态接近限位，不再直接从该姿态执行整段扫描。小于 0.02 rad 的初始反馈越界只允许沿路径改善，不能继续越界；明显越界、无可达观察点或路径检查失败时拒绝运动。

从 startFlat 合成起点的离线验证可在精确 `(0.20,0,0.30)` 找到解并完成 30° 光轴扫描；实际运行仍按实时反馈求解，不把离线角度写死。

需要本机 `z1_ctrl` 在 SDK 模式运行，且只能有一个 SDK 客户端。不要同时运行 `demo_self.py`（它会直接抓取）、`trial.py`、恢复/保持脚本或另一个搜索进程；本程序不会杀掉它们或自动接管它们的端口。

检查：

```bash
pgrep -af z1_ctrl
ss -lunp | grep -E '8071|8072'
```

仅当控制器未运行时，在单独终端启动：

```bash
cd /home/robot/unitree_ws/unitree_z1/z1_controller/build
./z1_ctrl
```

初始控制器状态可以是静止的 PASSIVE 或 JOINTCTRL。程序先检查复位路径，请求实测位置保持后执行低速复位，再按真实反馈规划抬臂路径；不调用电机校零、抓取或开合夹爪。如果正在运行其他 SDK 保持程序，应先按它自身的受控流程结束，不能强杀后直接抢占 UDP 8072。

默认扫描 J5，以准备完成后的观察姿态为中心 ±15°，关节速度不超过 0.03 rad/s。其余关节目标和夹爪角度保持不变。本地 SDK 模型验证：J5 改变镜头朝向，J6 在当前安装关系下主要使图像自转，因此不使用 J6 搜索。

## 先单独检查抬臂与转腕

先退出相机预览，再运行：

```bash
cd /home/robot/unitree_ws/b2w_navigation
./scripts/run_wrist_search.sh --execute --view
```

这条命令从当前姿态直接抬臂到观察点并使能机械臂扫描，不使能底盘。底盘应保持静止。确认图像随 J5 缓慢左右变化，机械臂及相机周围有足够空间。找到目标、超时或停止后，进程保持 SDK 通信，结束方式见下文。

## 同时探索与搜索

先结束旧导航栈，不能与原来的普通导航 launch 同时运行。在旧栈仍运行时先执行 `python scripts/control.py stop`，然后在旧导航终端 Ctrl+C。新版 launch 会检查已有导航节点并拒绝重复启动。

终端一，启动抬臂搜索专用导航：

```bash
cd /home/robot/unitree_ws/b2w_navigation
source setup.bash
ros2 launch b2w_nav arm_camera_search.launch.py start_rviz:=true
```

这一命令只启动节点，运动门仍关闭。它使用 1.80×1.80 m 的较保守轮廓覆盖展开的机械臂；局部规划使用多边形轮廓碰撞检查，前进速度上限 0.10 m/s。狭窄位置可能因没有可达探索点而等待，不会为了通过而缩小轮廓。

终端二，确认控制器就绪后（程序将自动准备观察姿态）：

```bash
cd /home/robot/unitree_ws/b2w_navigation
./scripts/run_wrist_search.sh --execute --explore --view
```

**这条命令会使能底盘与机械臂的真实动作。** 不需要再手动调用 `control.py enable` 或 Frontier enable。程序停止底盘、从当前姿态直接抬臂到观察姿态。脚本中的Nav2节点active/代价地图固定轮廓等前置校验已移除；若选择Nav2探索，规划器自身仍按配置执行。到位稳定后开始探索、扫描及逐帧投票。原任务监督器不启动，不调用机身 Spin 搜索动作；路径行驶本身仍可能需要转向。

首轮最多 3 次 Frontier 目标尝试、最多搜索 180 秒；其中一个上限到达就停止。相机失联/曝光过期、导航扫描/里程计过期、机械臂反馈异常或运动门被外部关闭也会锁定停止，不自动恢复。

若要从“单独转腕”换到“边走边搜索”，不要再启动第二个 SDK 客户端。先按下文受控结束单独转腕程序，再运行联动命令。

## 检测逻辑

- `SEARCHING`：Frontier 探索、J5 慢速扫描，同时对最近10个新检测帧做滚动投票。检测到 can（置信度 >=0.65）记1票，漏检记0票；有5票即通过，不必等满10帧。未满5票继续搜索，没有 VERIFYING 阶段，不因一帧候选而反复停车/恢复。
- 投票不要求连续命中、检测框 IoU 关联、关节速度连续低于阈值，也不要求深度；重复曝光、倒序帧和过期图像不计票。
- `FOUND`：达到票数后保存图像与关节角，停车停腕并保持。无深度也能完成本页的“发现目标”。带 `--approach` 时再进入 `TARGET_LOCALIZING`，用停车后的一帧有效深度定位，见 [WRIST_APPROACH.md](WRIST_APPROACH.md)。
- 单帧漏检或无深度不是 FAULT。`FAULT` 是相机/导航/机械臂/控制等故障锁定，后续即使相机再次识别到目标，也不会自动恢复投票或运动；相机识别线程可能仍继续更新窗口图像。
- `LIMIT` / `FAULT` / `STOPPED`：停止搜索，机械臂保持通信；不会自动重新使能，也不执行后续抓取。先处理故障，再按受控退出流程重新启动。
- 配置 `vote_window_frames=10`、`vote_required_frames=5`。心跳显示 `vote_hits`、`vote_frames`、`vote_required`、`vote_window`；`fault_reason` 是整体故障原因，`arm_error=null` 只代表没有机械臂错误。

状态与结果：

```bash
source setup.bash
ros2 topic echo /wrist_search/status
```

文件位于 `output/wrist_search/`：

- `latest.jpg`、`latest_depth.png`、`latest.json`：最新标注图、原始深度和元数据；
- `events.jsonl`：搜索/暂停/确认/故障事件；
- `found.jpg`、`found.json`：确认目标；点坐标仅在相机系，关节角是确认时反馈，**不是已同步标定的 map 系导航/抓取位姿**。

## 停止与结束通信

**正常退出按 Ctrl+C**：程序取消导航并请求底盘停车，确认底盘静止后，使用原 SDK 连接低速回到 `startFlat` 收拢关节角 `[0,0,-0.005,-0.074,0,0]rad`，保持夹爪角度；到位后自动结束相机和 SDK 通信并退出。这是关节姿态复位，不做电机零点标定。状态为 `EXIT_RESETTING` → `EXIT_READY`。

复位过程中再次 Ctrl+C（或调用 stop）会中断并保持通信。反馈异常、复位路径不通过、底盘未停车或复位超时也会保持 FAULT，不强制断开。排除故障后可再按 Ctrl+C 重试；退出复位不要求相机继续在线。

**只停车并保持、不复位退出**，按窗口 `q` 或执行：

```bash
ros2 service call /wrist_search/stop std_srvs/srv/Trigger '{}'
```

会停止底盘和转腕，但进程继续保持机械臂通信。这是保留的只停止功能；正常 Ctrl+C 退出则先复位再断开。SDK断开不提供持续承重保持。

也可用 `python scripts/control.py stop`，它现在会同时通知机械臂搜索程序停止。

**机械臂已放稳、支撑或可靠固定后**，再结束 SDK 通信：

```bash
ros2 service call /wrist_search/release_control std_srvs/srv/Trigger '{}'
```

活动搜索状态会拒绝直接释放，需要先 stop。找到目标也不能直接强杀保持进程。网络/电源/电机故障、进程崩溃或 SIGKILL 时，Python 不能保证机械臂承重保持；通信/状态异常会锁定 PASSIVE 请求，防止旧 JOINTCTRL 命令在连接恢复后自动重新使能；这时无法保证承重保持，需现场支撑。

## 配置与当前验证范围

搜索参数在 `config/wrist_search.json`，无需修改 Z1 抓取程序。扫描几何使用已有 `grasp_demo/config.json` 中的相机安装关系；本页第一阶段不依赖 B2W→Z1 安装外参，不发布导航目标；第二阶段采用已提供的安装变换，见 [WRIST_APPROACH.md](WRIST_APPROACH.md)。

已检查运动学、无目标继续搜索、10帧5票、漏检容忍、旧曝光拒绝、故障不恢复及释放控制门控；第一阶段已由用户反馈真机验证通过。新增接近阶段尚待真机验收；本次代理未执行真机动作。1.80 m 轮廓和采样运动学检查不能替代完整的机械臂自碰撞/环境碰撞模型。

## 看见 `Observation pose did not settle` 时

这表示程序未确认到位，不表示底盘应该开始走。复位和观察点都要求六个关节的最大角度误差 <0.08rad（约 4.6°），且稳定条件连续满足 0.3 秒。稳定依据是最大关节速度 <0.06rad/s，或完整 0.6秒位置窗口内每个关节的角度极差 <0.003rad（约 0.17°），且极差/窗口时长 <0.06rad/s。位置窗口要求至少 10个采样、相邻间隔不超过 0.15秒，并全部位于目标角度容差内；用于避免速度反馈尖峰导致已静止的机械臂反复确认失败。持续移动、较大摆动、位置超差和反馈中断仍拒绝通过。日志 `settle_basis=position_window` 表示采用位置窗口，`reported_speed` 表示采用速度。参数 `settle_position_window_sec` 和 `settle_position_range_rad` 位于配置的 `reset`、`observation` 段；窗口设为 0 可恢复单纯速度判定。观察点等待时间 10 秒，复位等待时间 15 秒。超时锁定 FAULT，随后即使机械臂晚些到位也不自动放行底盘。

阈值在 `config/wrist_search.json` 的 `reset` 和 `observation` 中；最新修改增加了位置窗口确认，仍保留关节限位、路径范围、跟踪误差和通信故障检查。下一次运行会输出最差关节、角度误差、速度和末端位置误差，记录到：

- `output/wrist_search/observation_status.json`：最新到位判定及实际/目标关节角、末端位置；
- `output/wrist_search/observation_settle.jsonl`：观察点到位检查过程；
- `output/wrist_search/exit_reset_plan.json`、`exit_reset_status.json`、`exit_reset_settle.jsonl`：退出复位的规划和到位记录。

状态字段含义：

| 字段 | 含义 |
| --- | --- |
| state | PREPARING 准备观察点；EXIT_RESETTING 退出复位；EXIT_READY 退出就绪；SEARCHING 探索扫描并投票；TARGET_LOCALIZING 停车后取深度；FOUND 已发现；FAULT 锁定故障；STOPPED 人工停止 |
| event=heartbeat | 周期状态输出，不是运动命令 |
| monotonic_s | 本机单调时钟秒数，用于日志排序，不是搜索已运行秒数 |
| explore | 启动时是否请求底盘探索，不代表底盘正在动 |
| base_output_enabled | 两级底盘运动门是否均已打开 |
| wrist_scanning | 是否正在执行转腕扫描 |
| arm_control_alive | SDK 通信仍运行且反馈健康；不等同于故障时的物理承重保证 |
| camera_alive | 新版表示最近曝光仍新鲜且相机线程未报错；旧版只检查相机线程是否报错 |
| arm_error | 锁定的机械臂错误原因；STOPPED 后仍保留原故障供排查 |
| approach_enabled / grasp_enabled | 本页搜索模式均为 false；加 `--approach` 后前者为 true，抓取始终关闭 |

Ctrl+C 正常完成退出复位后自动退出；窗口 q 或 stop 服务仅停止并保持通信，仍可用上述 release_control 手动结束。对正在运行的旧进程，文件修改不会热更新；不要同时再开第二个 SDK 客户端。


## 功能源码对应关系

| 功能 | 文件/函数 |
| --- | --- |
| 启动参数、状态机、底盘停止/放行、候选确认 | `scripts/search_with_wrist.py`：`main()`、`Coordinator` |
| 相机 SSH 与 RGB-D 协议 | `scripts/wrist_search_io.py`：`Camera`；远端 `scripts/wrist_camera_agent.py` |
| 直接抬臂、退出复位、到位检查、转腕和保持 | `scripts/wrist_search_io.py`：`Wrist.__init__()`、`move_to_observation()`、`wait_observation_settle()`、`reset_for_exit()`、`control()` |
| 复位路径、观察点 IK、扫描检查与到位判定 | `src/b2w_perception/b2w_perception/wrist_search_core.py`：`plan_startup_reset()`、`plan_observation()`、`validate_sweep()`、`SettleTracker` |
| 密码、观察点、复位关节角、容差、速度 | `config/wrist_search.json` |
| 专用 Nav2/Frontier 启动与大轮廓 | `src/b2w_nav/launch/arm_camera_search.launch.py`、`src/b2w_nav/config/nav2_arm_search.yaml` |

当前搜索程序不通过 `unitree_z1/yolo/grasp_demo/demo_self.py` 执行；该文件仍是原来的抓取实验程序。
