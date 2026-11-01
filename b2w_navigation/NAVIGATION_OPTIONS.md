# 2026-10-29：B2W自动避障接口核对与导航改造建议

本轮为接口核对、只读服务查询和方案设计，未修改当前运动控制链路，未切换避障开关或发送运动命令。

## 已确认的接口与证据

- 本机与宇树公开C++ SDK中，独立 ObstaclesAvoidClient 位于 go2/obstacles_avoid，提供 SwitchGet、SwitchSet、Move、UseRemoteCommandFromApi 等接口；不能仅凭目录中的声明推断B2W固件兼容。
- B2 SportClient声明 Move、MoveToPos、TrajectoryFollow、SwitchMoveMode、VisionWalk；这些声明没有承诺MoveToPos带雷达避障。官方B2W Sport示例复用b2命名空间并演示Move/StopMove，没有演示独立自动避障服务。MoveToPos是否适用于当前B2W、坐标约定和避障语义仍未实机验证。
- 之前DDS话题清单出现过 /api/obstacles_avoid/request、response，只表示话题曾被发现。
- 本轮只查询API版本（内部ID1）与SwitchGet（ID1002），当前发现请求订阅者0、响应发布者0，两个查询均无响应。因此目前不能以“官方自动避障已可用”为基础重写控制。查询结果：output/obstacle_avoid_readonly_probe.json。此结果不能证明所有B2W固件都不支持，只说明本次没有发现工作的服务。
- 宇树机载SLAM导航是另一条路线：需要对应服务、定位和地图就绪。目前工作链路为本机slam_toolbox/Nav2，尚未确认机载服务可用，不能和Sport速度控制或Go2避障接口混为一谈。

官方来源：

- https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/robot/go2/obstacles_avoid/obstacles_avoid_client.hpp
- https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/robot/b2/sport/sport_client.hpp
- https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/b2w/b2w_sport_client.cpp
- https://github.com/unitreerobotics/unitree_slam/blob/main/unitree_slam_example/demo_b2.cpp

## 建议的任务与控制分层

上层只选目标，底层做局部避障和到点停车：

1. 目标管理器：接收易拉罐停靠目标、人工探索点或Frontier选出的探索点，管理停止/取消与目标优先级。
2. 接近目标在odom下冻结，使用已确认的腕部相机深度、机械臂FK及安装外参，减少近距离接近对全局SLAM地图的依赖。SLAM可继续用于可视化和探索目标选择。
3. 目标到机器人之间的直线只作为参考方向；前方有障碍时允许局部绕行，没有可行通道则停车，不能同时要求严格直线和绕过障碍。
4. 局部控制器根据真实机身/当前臂姿占用范围、实时雷达和里程计，持续产生前进/转向速度；复用现有Sport桥、底盘运动门、停止超时、Ctrl+C复位退出及腕部搜索。
5. 探索点生成与执行分离，复用FrontierExplorer核心选点逻辑，而不让原frontier_explorer_node直接拥有NavigateToPose。

实现优先级建议：先保留Nav2 Controller Server/DWB的局部控制能力，提交指向目标的参考路径到FollowPath，移除当前全局PlannerServer/全局静态地图与NavigateToPose行为树的依赖。局部代价地图以odom为参考，使用实时扫描，不叠加历史静态SLAM地图；按实际姿态设置轮廓。这样复用现有局部跟踪、避障和反馈机制，避免立即从零实现整套控制器。

如果要求完全移除Nav2，也可另写局部轨迹采样控制器，但需额外完成动力学、刹停距离、碰撞判断、窄通道与陷阱等验证。局部避障不保证绕过长墙、走出凹形障碍或跨房间到达目标；此时需要中间探索目标或全局规划。

Nav2官方局部控制说明：

- https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/core_servers/controller_server/
- https://docs.nav2.org/rolling/configuration_and_development/first_time_robot_setup_guide/navigation_plugins/setup_navigation_plugins/

## 仍要解决的数据问题

雷达自体点云过滤需要放在建图/局部避障共同输入之前；否则更换控制框架后，机身/机械臂回波仍会被当作障碍。不能用“删除机器人周围整片环境点”代替真实自体过滤。旧SLAM地图残留需在输入修正后通过适当的重建/更新流程处理。

建议先验收短距离目标跟踪与遇障停车，再验收局部绕障，最后接探索目标。没有因为本轮调研而启用任何上述新控制流程。

后续实施：用户已明确确认MoveToPos使用机器人局部坐标，并指定到易拉罐左右0.5m。本地SDK点位控制模式已接入，见WRIST_SPORT_APPROACH.md；未自动开启避障、未执行真机动作。

后续更新：当前B2W固件对API1036返回3203（API未实现），已按用户要求改为Move(1008)+Sport里程计的软件点位控制。接近阶段无Nav2依赖，未添加自动绕障；具体运行以WRIST_SPORT_APPROACH.md为准。
