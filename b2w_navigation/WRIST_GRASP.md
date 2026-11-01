# 侧向接近海绵后单视角投票抓取与归位

在原侧向接近命令上增加 `--grasp`，才启用到位后的抓取。

```bash
# 终端一：底盘桥与里程计
cd /home/robot/unitree_ws/b2w_navigation
source setup.bash
ros2 launch b2w_hardware sport_position_bringup.launch.py
```

保持现有 `z1_ctrl` 运行。若未启动，在另一个终端执行：

```bash
cd /home/robot/unitree_ws/unitree_z1/z1_controller/build
./z1_ctrl
```

机器狗站稳、路径清空后启动：

```bash
cd /home/robot/unitree_ws/b2w_navigation
./scripts/run_wrist_search.sh --execute --approach --approach-backend sport --grasp --side left --view
```

`left` 表示停在海绵左侧，海绵在机器狗右侧，机械臂向右转；换成 `right` 后相反。不要另外运行 `demo_self.py`；抓取复用它的求解函数，始终只有一个 SDK 客户端和一个相机流。

流程：

1. 启动接管SDK后先给夹爪归0指令，六关节保持当前位置；归0指令发完等待0.5秒，不检查夹爪实测角度是否到位。之后准备观察姿态并原地转腕搜索海绵，确认后侧向接近，最终保持开始接近时的机身朝向。
2. `--grasp` 使用 `post_grasp.side_offset_m=1.0` 米侧向距离；不带该参数仍使用 `approach.side_offset_m=1.0` 米。当前接近在前后方向还要求误差≤0.06m、朝向误差≤0.10rad；配置的 `approach.stand_down_forward_compensation_m=0.07` 只在启用到位下蹲时使指令停车点前移7cm，以补偿最近四次下蹲约7–8cm的后移，横向1m不变。当前底盘配置仍在到位后执行一次 StandDown，发出 StandDown 后等待 3 秒且底盘静止即报告 ARRIVED，随后开始抓取阶段，不等待 API1005 成功回复。
3. 首次深度定位时保存 Sport 世界坐标中的目标位置；到位下蹲后，依据实时机身平移、转向、IMU 俯仰和高度换算回当前机械臂坐标。机械臂以 0.20 rad/s 的关节速度上限直接朝该估计位置观察；朝向与俯仰由目标位置计算，不再固定朝左/右 90° 横扫。转向到位角度容差 0.08 rad、速度容差 0.10 rad/s，连续稳定 0.2 秒；位置窗口为 0.3 秒、角度波动容差 0.005 rad。不再轮巡多个俯仰视角。
4. 在估计目标附近做 J5 ±6° 局部腕部扫描，速度参数 0.12 rad/s；首次寻找阶段的扫描速度参数为 0.09 rad/s（原 0.03）。沿用主搜索的最近 3 帧 2 票逻辑。漏检计 0 票，重复帧不计票；不要求连续命中，也不要求等满 3 帧。
5. 达到票数后停腕，获取一帧新的有效深度即可定位；不再要求连续 3 帧深度确认。复用 `demo_self.short_side_rotation`、`nearby_target` 求取沿框短边闭合的抓取姿态：先试检测定位中心和1cm网格近邻；近邻无解后，继续在全关节限位范围内以多起点数值优化寻找最近可用候选，取消4cm距离硬上限和强制深入量。扩大搜索时以位置接近为主，短边姿态为偏好，必要时调整抓取姿态；每个候选仍验证SDK逆解、直接接近与归位路径。优化结果是数值近似，不承诺全局最优，也不把远离海绵的可达点视为已抓稳。求解较长时在同一停腕视角刷新一帧深度，确认目标和机械臂未移动，不重新扫描或投票。定位中心来自检测框中心的深度，并非完整物体的几何体积中心。
6. 开爪后按`post_grasp.visual_servo`分段接近，默认每段TCP位移约3cm、单关节变化不超过0.2rad。段间停稳后用新图像、新深度和实测关节角更新海绵位置，单次位置修正不超过2cm。检测框仍在但无深度时，可用最后有效目标高度平面与相机射线交点修正水平位置。
7. 中途检测不到目标、没有可用的新定位或新点跳变超过关联范围时，立即锁定最后有效抓取关节目标，继续完成剩余接近、合爪、归位；不等待目标恢复，不重新投票。运动段内检测到丢失也不暂停该段，随后按缓存目标完成剩余路径。相机断流、控制器通信或跟踪异常仍中止。
8. 正常跟踪至最后3cm时锁定目标完成末段；到位后合爪，再保持合爪直接返回`reset.target_q_rad`，进入`GRASP_DONE`持续保持通信。返回角为`[0,0,-0.005,-0.074,0,0]`，不执行电机校零，不自动张爪。

腕部扫描、确认、运动和夹爪等待期间均处理 ROS、画面按键和取消请求。按 `q` 停止并保持；Ctrl+C 按原流程停车、合爪保持当前角度并复位退出。相机断流、反馈、跟踪或 IK 异常会进入 FAULT，不自动再次抓取；初次投票定位成功后，单纯检测/深度丢失按最后有效抓取目标继续。

开爪目标仍来自 grasp_config。已删除开爪角度误差、最小开度、稳定窗口及开爪未到位取消抓取的检查；按配置速度发完开爪指令，再等 `post_grasp.gripper_open_wait_sec=0.5` 秒即继续接近抓取，不由实测开度决定是否继续。实测角度和误差只记录到 `gripper_status.json` 与心跳，`open_command_completed` 只表示指令发送完成，不表示已确认实际张开。

夹爪受物体阻挡可能无法到达完全闭合角，此时沿用 demo 的处理：保持闭合目标并归位。`contact_inferred` 仅表示可能接触，`grasp_verified` 为 false；没有新增抓稳传感器，不能把流程完成当作已确认抓稳。

配置位于 `config/wrist_search.json` 的 `post_grasp` 段。抓取开合角、TCP 和手眼外参来自 `grasp_config`。1.0 米为用户校正后的停车距离参数，最终是否可达仍以实时 IK 和路径验证为准。

调试输出位于 `output/wrist_search/`：

- `grasp_search_hint.json`：根据首次定位和当前姿态换算的估计位置、相机瞄准角与局部扫描参数。
- `gripper_status.json`：开合爪实测角度、误差、持续时间与判定结果。
- `grasp_target.json`：本轮海绵定位中心、选中的TCP坐标、抓取关节目标和中心偏离量 `center_offset_m`。
- `grasp_result.json`：本轮 RUNNING / GRASP_DONE / FAULT 结果。
- `grasp_status.json`、`grasp_settle.jsonl`：关节到位稳定检查。
- `events.jsonl`、`/wrist_search/status`：GRASP_TURNING、GRASP_SEARCHING、GRASP_LOCALIZING、GRASP_PLANNING、GRASP_REFRESHING、GRASP_OPENING、GRASP_APPROACHING、GRASP_CLOSING、GRASP_RESETTING、GRASP_DONE。

验证使用无硬件状态机测试与真实 SDK 的离线运动学模型，不实例化 ArmInterface。目标坐标在移动/下蹲后的转换、中心优先最近可用逆解、前后停车对齐、单视角观察、间歇命中投票、重复帧过滤、单帧深度定位、直接抓取路径和归位已有离线检查；自动抓取整段仍待真机验收。路径验证是现有采样运动学包络检查，不是机械臂与周边物体的完整碰撞模型。


通信异常处理：SDK Python 绑定提供 feedbackStatus（UDP状态、控制器状态和逐电机错误）与 suspendTracking。UDP或电机连接异常时暂停跟踪并发送无状态切换的INVALID命令，保留SDK通信，不因单纯断链主动请求PASSIVE，也不在恢复后自动续跑。明确过流、过热等电机错误仍保留保护；硬件自身的断链保护无法由应用保证不卸力。最近故障写入arm_fault_status.json。

夹爪运动阶段同时发送目标角度及与渐变一致的速度，到位保持时速度为0；gripper_status.json记录sdk_command，可比较缓冲目标和真实发送包。启动回0是角度命令，不是编码器零点校准；实际夹爪不动时仍需结合控制器/电机反馈排查。开爪角度达标检查保持关闭。

抓取和归位的关节到位角度容差使用post_grasp.joint_position_tolerance_rad=0.08rad，避免已接近目标仍因原0.025rad限制阻塞合爪。正常流程仍为接近到位、合爪、合爪归位、GRASP_DONE后持续保持。

观察位姿 IK 无解时，不再以 4cm 邻域作为终止边界：初始观察和到位后的目标指向观察都先做局部 IK，再在关节工作范围内多起点优化附近可达位置。到位后优先保持镜头指向海绵（误差不超过5°），在有效候选中选择距离期望观察点最近的一项；无位置偏移硬上限，不保证数值搜索能取得全局最近解。候选仍须通过SDK逆解复核、完整腕部扫描和移动路径约束，不会为了凑逆解越过限位。计算过程中持续响应取消，最终仅执行一次选定的观察动作，投票仍为3帧2票。

`grasp_observation_plan.json` 保存选定观察位姿、路径和搜索方式；`grasp_search_hint.json` 保存实际观察点和偏移。若局部及工作范围搜索均无有效候选，仍停止并报告原因。抓取点继续使用已有的无4cm硬上限最近可达点搜索；复位采用配置关节角，不经过Cartesian IK。

观察规划已删除“相机距海绵至少10cm”的拒绝条件，以及对应的最小距离优化惩罚。附近可达点即使距目标不足10cm也可选用；不会为了满足相机距离而额外后退。只有镜头与目标完全重合时无法定义朝向，计算中避免除零。关节限位、移动路径、完整腕部扫描和相机朝向约束仍沿用。

首次转向前，重新接收停车下蹲后的Sport机身反馈及相机帧，旧的缓存机身位姿即使距当前不足0.5秒也不能直接使用。如新帧已检测到海绵且深度有效，优先用该帧和当前实测关节角更新海绵位置及保存的Sport参考系目标；尚未看到时，用本次新接收的机身平移、姿态和高度重算目标方向。此帧仅用于朝向提示，抓取仍需随后3帧2票和新深度确认。`grasp_search_hint.json`记录target_update_source、T_sport_body、updated_target_base_xyz_m及反馈/相机时间，区分post_arrival_rgbd与post_arrival_sport_pose两种数据来源。

修正近距离观察时的反向转臂：原固定20cm观察半径可能越过估计仅7cm远的目标，镜头偏移修正和附近点优化会选中J1左转、手臂反折、镜头向左回看的姿态，即使目标在右侧。现在观察半径取配置值与目标水平距离一半中的较小值，优先从机身一侧观察；J1限定在目标方向附近60°且保持目标左右侧，镜头不允许朝目标方向的反面回看。局部IK和工作范围搜索均执行相同的方向约束。日志新增target_bearing_deg、selected_base_joint_deg、selected_camera_yaw_deg，分别表示目标方位、实际选中的基座关节角和镜头方位，避免把镜头朝向误认为J1角度。

2026-10-30 修正当前竖装相机的绕光轴方向：原始相机画面地面位于左侧，且初始实测J6约0.002rad，并非未回0导致的90°画面旋转。当前使用的`unitree_z1/yolo/grasp_demo/config.json`将光学X映射到SDK末端+Z、光学Y映射到末端-Y、光学Z映射到末端+X；安装平移[-0.04,0,-0.04]m保持原值。初次地面海绵记录回放，目标机身Z由-0.061m修正为-0.613m，Y由0.502m修正为0.064m，说明错误不仅影响抬头俯头，也影响侧向停车。此为依据实际安装图像和日志的轴向修正，尚非完整手眼标定。备份及回放结果在output/sponge_validation/camera_portrait_previous_config.json和camera_portrait_replay.json。需要重启搜索并重新开始检测、接近，不能沿用旧坐标和旧停车目标。


## 分段视觉更新参数

配置入口为`config/wrist_search.json`的`post_grasp.visual_servo`，默认启用，重启搜索程序读取。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| enabled | true | 开启分段视觉更新；false沿用直接接近 |
| step_m | 0.03 | 每个运动段的TCP位移上限，m |
| max_joint_step_rad | 0.2 | 每段关节变化上限，rad |
| max_correction_m | 0.02 | 每次目标位置修正上限，m |
| association_distance_m | 0.08 | 新定位与原目标的关联范围；超出则沿用缓存目标 |
| final_blind_distance_m | 0.03 | 正常跟踪时末段锁定距离；中途丢失也允许按缓存目标完成更长的剩余路径 |
| max_steps | 60 | 最多更新段数 |
| max_duration_sec | 180 | 分段更新阶段时间预算，秒 |
| use_ground_projection | true | 检测框有效但无深度时，投影到最后有效目标高度平面 |

更新只使用停稳后采集的帧，避免拿运动期间的图像配当前关节角。单纯丢目标没有等待超时参数或恢复投票。`grasp_visual_trace.jsonl`记录每次新测量、修正量和`cached_target`续抓事件；`grasp_target.json`随更新保存当前点与关节目标。初次寻找仍使用3帧2票。

抓取路径和抓取后的正常/退出归位使用Sport机身反馈与安装变换推导的地面平面：相机离地至少3cm、采样连杆点离地至少1cm。观察阶段仍使用原有抬臂包络；关节限位、最大伸展和路径采样保留。此修正避免把机械臂基座平面误当作地面而将抓取点抬高。离线SDK回放中，原先与海绵中心相差11.3cm的样本现可直接求解中心。该包络不是完整机身/环境网格碰撞检测；流程完成继续记录grasp_verified=false。
