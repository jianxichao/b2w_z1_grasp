# B2W → Z1 安装外参：只读检查结果与静态测量方案

检查日期：2026-10-25。没有连接机器人、启动 SDK、采集实时反馈、
执行回 Home、切换模式、停止现有程序或发布 TF。新增文件仅在本目录。

## 1. 本机证据

- SDK：`../../unitree_z1/z1_sdk`，Git remote 为
  `https://github.com/unitreerobotics/z1_sdk.git`。
- 控制器：`../../unitree_z1/z1_controller`，Git remote 为
  `https://github.com/unitreerobotics/z1_controller.git`。
- 实际反馈：C++ `UNITREE_ARM::LowlevelState::getQ()`；Python
  `arm.lowstate.getQ()`。定义在 SDK 的
  `include/unitree_arm_sdk/message/LowlevelState.h`，Python 绑定在
  `examples_py/arm_python_interface.cpp`。返回六个 SDK 反馈关节角，单位 rad，
  数组下标 0..5 对应 J1..J6；按官方模型名称关联为 joint1..joint6。
- `arm.q` 是命令参数；`lowcmd` / `SendCmd` 是命令数据，不能充当实际反馈。
- `unitreeArm.h` 和 `ctrlComponents.h` 明确说明 `sendRecv()` 会发送命令包再接收反馈。
  `loopOn()`、`sendRecvThread->start()` 都启动收发线程。
  因此不能把“没有调用 MoveJ”理解成“没有发送控制数据”。
- `examples/read_z1_state.cpp` 虽然 main 只打印状态，但启动了收发线程；
  `yolo/grasp_demo/demo.py --state-only` 经 `Arm.read()` 调用 `sendRecv()`。
  两者均不满足本次严格的完全被动读取要求。
- SDK 有本地修改及未跟踪文件，不能直接当作原版官方 SDK。
- 检查时存在 `z1_ctrl` PID 26137 和 `recover_home.py` PID 30843；
  UDP 8071 被前者占用，8072 被后者占用。进程名不能证明机器人当前静止、
  已到 Home 或通信健康。这只是检查时快照，未干预这些程序。
- 本地 `main.cpp` 的 SDK 通道配置为控制器 127.0.0.1:8071 → SDK
  127.0.0.1:8072。控制器配置的下位机目标为 192.168.123.110:8881。
  本方案不直接连接下位机。

### Home 与角度约定

本地 `config/savedArmStates.csv`：

| 配置标签 | 六关节配置角 rad | degree（约） |
| --- | --- | --- |
| startFlat | 0, 0, -0.005, -0.074, 0, 0 | 0, 0, -0.2865, -4.2399, 0, 0 |
| forward | 0, 1.5, -1, -0.54, 0, 0 | 0, 85.9437, -57.2958, -30.9397, 0, 0 |

`State_BackToStart.h` 只有声明，实际实现封装在共享库。
对本地 x86_64 库做静态反汇编可见 `State_BackToStart::enter()`
调用 `JointSpaceTraj::setJointTraj(..., "startFlat", ...)`。
因此这版实现的 Home 目标引用 startFlat，而不是 forward；不等于六关节全零。
本次另以 `/proc/26137/maps` 确认该进程加载本工作区的同一份库，
其工作目录为 `/tmp/z1_can_trial_f9xgk305/build`；对应临时 config 的
startFlat 也与上表一致。不能把磁盘配置与当前实际到达角混为一谈。

已有 `yolo/grasp_demo/recovery_output/motion.jsonl` 中有历史 `sdk_home_complete`
反馈，约为 `[-0.00109, 0.00445, -0.00867, -0.07185, -0.00381, 0.00149] rad`，
与 startFlat 接近。但同一日志后续记录 `control_fault_latched`，原因为
状态缺失/断连/电机故障（日志没有区分具体哪一种），且
`physical_hold_guaranteed=false`。这些是历史记录，不代表当前状态；
不能据此宣称真机当前健康、静止或可安全测量。

`getQ()` 的静态反汇编显示它复制反馈 q 数组，不在 getter 中做符号/零偏转换。
**这不证明下位机、控制器处理和 URDF 之间没有转换。**
本地没有足够源码/标定记录来确认所有 SDK→URDF 映射。
先保留 `q_urdf[i] = s[i] * q_sdk[i] + b[i]`，其中 s 和 b 待核实；
不要把电机编码器原始角与关节空间反馈混用。
需要匹配当前机器人版本的官方零位/方向说明、现用控制器版本和零位标定记录；
也可先离线比较 SDK 模型与官方 URDF 在多个合成姿态的 FK，核对模型约定。
但模型一致仍不能代替真机零位正确性的确认。

## 2. 最小被动读取脚本（已准备，未连接验证）

`read_existing_feedback.py` 仅用 Python 标准库。它在 Linux `lo` 上通过
AF_PACKET 复制已有的 8071→8072 UDP 反馈，不占用这两个 UDP 端口、不发包、
不导入 SDK、不使能、不改变模式，不修改控制程序。
依据本地 `message/arm_common.h` 解码 208 字节的 `RecvState`，输出六个
`jointState[i].Pos` 的 rad/degree、关节编号和名称及电机状态。
目标角和 Home 配置角均输出 null，避免与实际反馈混淆。
电机 connection_state=0 为正常；error 的 0x01/0x02/0x04/0x20 位需关注。
该协议检查不能替代驱动层通信诊断；即使收到新 UDP 包，里面的电机反馈也可能陈旧。

连接/采集前提：

1. 用户确认允许采集，且确认机器人和现有程序处于可接受的稳定状态。
2. 已有控制器持续向本机 SDK 发送匹配此版本的反馈；端口/协议必须一致。
3. Linux root 或 CAP_NET_RAW 权限。权限不足则退出，不自动申请或修改权限。
4. 若已有反馈不存在，不为读取而启动控制器、SDK、HTTP 服务或切换模式。
   可改用现有程序已经导出的、带时间戳的实际 q 快照。

仅供之后确认后手工执行，本次没有执行：

```bash
sudo python3 /home/robot/unitree_ws/diagnostics/z1_mount/read_existing_feedback.py --capture-existing
```

默认不带参数会直接拒绝采集。超时也只退出。此脚本尚未通过真机包验证，
不将“已准备脚本”表述为“已读到当前六关节角”。

## 3. 官方 URDF 的 joint2→joint3 几何

检查范围内未发现独立的官方 Z1 URDF。本次在线核对的官方来源：
https://github.com/unitreerobotics/unitree_ros/blob/master/robots/z1_description/xacro/z1.urdf

本说明针对该 URDF，实际硬件型号应匹配。joint1 的 origin=(0,0,0.0585)，
axis=(0,0,1)；joint2 的 origin=(0,0,0.045)，axis=(0,1,0)；
joint3 的 origin=(-0.35,0,0)，axis=(0,1,0)；这三处 origin rpy 均为零。

以下 q1、q2 必须是已核实的 **URDF 实际反馈角**：

```text
p_joint2 = [0, 0, 0.1035]                     (link00，m)
d23 = Rz(q1) Ry(q2) [-0.35, 0, 0]
    = [-0.35 cos(q2) cos(q1),
       -0.35 cos(q2) sin(q1),
        0.35 sin(q2)]
p_joint3 = p_joint2 + d23
d_xy = [d23.x, d23.y]
L_xy = 0.35 |cos(q2)|
alpha = atan2(d23.y, d23.x)
```

joint3 自身转角不改变 joint3 轴心位置；q3..q6 不影响这个方向。
切勿无条件使用 alpha=q1+π：当 cos(q2)<0 时投影方向会翻转。
L_xy 接近零时 alpha 不可靠。脚本默认 L_xy<0.05 m 拒绝计算 Yaw，
这是保守工程阈值，不是官方阈值。可按测量误差调整：横向端点误差 ε
大致产生 ε/L_xy rad 的角度误差，应该优先选择长投影。

`mount_geometry.py` 完全离线，显式要求六个已确认的 URDF 角；不自动把
SDK 角当作 URDF 角。未知实测方向时不生成数值 R/T。
用法：`python3 mount_geometry.py --q-urdf q1 q2 q3 q4 q5 q6`，
其中 q1..q6 要替换为实际数值（rad）。之后可加 `--beta-degree 实测角度`。
其数值矩阵仍以 Roll/Pitch=0、实测方向可信为前提。

### 不移动机器人的最简单测量

1. 保持当前稳定姿态，不要求机器人去 Home。记录同一时刻的六关节实际反馈、
   时间戳，以及现有程序导出的速度/健康状态；用相邻快照确认姿态稳定。
2. 在车体上确定与 B2W base_link +X 平行的前向基准线，左侧是 +Y。
3. 从上方测量 **joint2 轴心→joint3 轴心** 的有向投影，不能使用末端/夹爪朝向
   代替。可用铅垂线投影轴心到共同平面，或用带前向基准、尽量正视的俯视照片；
   注意相机透视/倾斜和高差。关节外壳边缘不是轴心，需注明测量点依据。
4. 得到该有向线相对 B2W +X 的角 beta：前=0°，左=+90°，后=±180°，右=-90°。
   也可提供两个投影点在车体平面内的 (x,y)，由 atan2(Δy,Δx) 求 beta。
   必须标出哪个是 joint2、哪个是 joint3，否则存在 180° 歧义。
5. 在 Z 轴平行且 Roll/Pitch 可忽略的前提下，
   `installation_yaw = wrap_to_pi(beta - alpha)`。
   测量第二个已知几何方向可交叉验证；无需主动改变机器人姿态。

所需实物信息：有向 beta（或两轴心投影坐标/带前向基准的俯视照片）、
拍摄/读取时间对应关系、底座水平/平行证据和 SDK 零位方向的可信记录。
手机水平仪可检查安装面的相对倾角；车体整体相对地面倾斜不等于两坐标系相对倾斜。
如果相对 Roll/Pitch 不可忽略，单一平面方向不足以求完整 3D 旋转，需测量
至少两个不共线的空间方向，不能沿用纯 Yaw 公式。

Home 是否适合：如果映射核实后 q2≈0，startFlat 的投影约 0.35 m，几何上适合；
但必须实际静止、轴心可辨认且不受遮挡。无需为了标定执行 Home。
forward 的 q2=1.5 时投影仅约 0.02476 m，按默认阈值不适合。
上述都是配置角的条件分析，不是当前实际姿态的计算结果。

## 4. 尚未确定 Yaw 时的参数化外参

定义 `p_base = R_base_link00 p_link00 + t_base_link00`，即 link00 原点/坐标轴
在 base_link 下的位姿。不是其逆变换。t 仍为用户给出的初估值，未标定。
若相对 Roll/Pitch 可忽略，令未知安装 Yaw 为 ψ：

```text
R = [ cosψ  -sinψ  0 ]
    [ sinψ   cosψ  0 ]
    [ 0      0     1 ]

T = [ cosψ  -sinψ  0   0.078   ]
    [ sinψ   cosψ  0   0       ]
    [ 0      0     1   0.09602 ]
    [ 0      0     0   1       ]
```

若 Roll/Pitch 未核实，应保留一般 `R=Rz(ψ) Ry(θ) Rx(φ)`，不能认定纯 Yaw。

ROS2 Humble 静态 TF 示例，仅在 ψ 验证、Roll/Pitch 可忽略之后运行。
YAW_RAD 必须先填写验证值；以下保护避免未知角自动变成零：

```bash
: "${YAW_RAD:?请先填入已验证的安装Yaw，单位rad}"
ros2 run tf2_ros static_transform_publisher \
  --x 0.078 --y 0 --z 0.09602 \
  --yaw "$YAW_RAD" --pitch 0 --roll 0 \
  --frame-id base_link --child-frame-id link00
```

发布前应检查 TF 树没有另一个 link00 父变换；官方独立 URDF 自带的
world→link00 固定关节也应在整合时处理，避免重复父节点。
本次没有修改任何 URDF/TF 配置或运行发布命令。

## 5. 离线验证

已用合成字节包检查反馈解码、UDP 方向过滤、rad/degree 换算、错误包拒绝；
用合成 URDF 角检查零位 FK、近竖直退化、cos(q2) 变号时的方向翻转、
后四关节不影响该方向、Yaw 差值与齐次矩阵平移。全部通过。
没有运行 SDK、实时抓包或任何真机操作。协议与真机状态新鲜度仍需之后验证。
