#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launch_file="$root_dir/src/rl2_nav/launch/autonomous_mapping_sport_bringup.launch.py"
safety_file="$root_dir/src/rl2_nav/rl2_nav/cmd_vel_safety_node.py"
bridge_file="$root_dir/src/rl2_hardware/rl2_hardware/go2w_sport_cmd_bridge.py"
bridge_yaml="$root_dir/src/rl2_hardware/config/go2w_sport_bridge.yaml"

pass_count=0
fail_count=0

check_file() {
  local label="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    printf '[PASS] %-42s %s\n' "$label" "$path"
    pass_count=$((pass_count + 1))
  else
    printf '[FAIL] %-42s %s\n' "$label" "$path"
    fail_count=$((fail_count + 1))
  fi
}

check_text() {
  local label="$1"
  local text="$2"
  local path="$3"
  if grep -Fq -- "$text" "$path"; then
    printf '[PASS] %s\n' "$label"
    pass_count=$((pass_count + 1))
  else
    printf '[FAIL] %s\n' "$label"
    fail_count=$((fail_count + 1))
  fi
}

cat <<'EOF'
===============================================================================
RL2 双门控安全架构：离线部署审计（OFFLINE DEPLOYMENT AUDIT）
机器人无需开机；本报告读取本地源码和安装产物，不代表节点当前在线。
===============================================================================

控制链路：

Nav2 controller / velocity_smoother
          |
          |  /cmd_vel_nav
          v
+-----------------------------------------+
| Gate 1: /cmd_vel_safety_node            |
| enable: /cmd_vel_safety_node/enable     |
| 限速 + 指令超时 + 感知/里程计门控       |
+-----------------------------------------+
          |
          |  /cmd_vel_sport
          v
+-----------------------------------------+
| Gate 2: /go2w_sport_cmd_bridge          |
| enable: /go2w_sport_cmd_bridge/enable   |
| 新鲜度检查 + Sport API 封装 + 停车心跳  |
+-----------------------------------------+
          |
          |  /api/sport/request (1008/1003)
          v
Unitree GO2-W internal sport controller

安全不变量：两个 Gate 必须分别显式 enable；任意一个未启用都不能形成运动输出。

=== Source and installed artifacts ===
EOF

check_file "Gate 1 source" "$safety_file"
check_file "Gate 2 source" "$bridge_file"
check_file "integrated launch" "$launch_file"
check_file "Gate 2 parameter file" "$bridge_yaml"
check_file "installed Gate 1 executable" "$root_dir/install/rl2_nav/lib/rl2_nav/cmd_vel_safety_node"
check_file "installed Gate 2 executable" "$root_dir/install/rl2_hardware/lib/rl2_hardware/go2w_sport_cmd_bridge"

cat <<'EOF'

=== Gate 1 deployment checks ===
EOF
check_text "Gate 1 subscribes to /cmd_vel_nav" '"input_topic": "/cmd_vel_nav"' "$launch_file"
check_text "Gate 1 exclusively outputs /cmd_vel_sport" '"output_topic": "/cmd_vel_sport"' "$launch_file"
check_text "Gate 1 starts DISARMED" '"enabled_on_start": False' "$launch_file"
check_text "Gate 1 exposes a SetBool enable service" 'self.create_service(SetBool, "~/enable", self._on_enable)' "$safety_file"
check_text "Gate 1 rejects stale Nav2 commands" 'input_timeout_sec' "$safety_file"
check_text "Gate 1 requires fresh scan input" 'require_scan' "$safety_file"
check_text "Gate 1 requires fresh odometry input" 'require_odom' "$safety_file"
check_text "Gate 1 includes odometry-jump disarm" 'odom pose jump; output disarmed' "$safety_file"

cat <<'EOF'

=== Gate 2 deployment checks ===
EOF
check_text "Gate 2 subscribes to /cmd_vel_sport" 'input_topic: /cmd_vel_sport' "$bridge_yaml"
check_text "Gate 2 publishes only /api/sport/request" 'request_topic: /api/sport/request' "$bridge_yaml"
check_text "Gate 2 starts DISARMED" 'enabled_on_start: false' "$bridge_yaml"
check_text "Gate 2 exposes a SetBool enable service" 'self.create_service(SetBool, "~/enable", self._on_enable)' "$bridge_file"
check_text "Gate 2 has a 0.30s command timeout" 'input_timeout_sec: 0.30' "$bridge_yaml"
check_text "Gate 2 checks fresh robot sport state" 'sport_state_timeout_sec: 0.50' "$bridge_yaml"
check_text "Gate 2 exposes explicit StopMove" 'self.create_service(Trigger, "~/stop_move", self._on_stop_move)' "$bridge_file"
check_text "Gate 2 repeats stop commands" 'stop_repeat_count: 3' "$bridge_yaml"

cat <<EOF

=== Audit summary ===
PASS: $pass_count
FAIL: $fail_count
EOF

if (( fail_count == 0 )); then
  cat <<'EOF'
RESULT: PASS — 双门控节点、话题边界、独立使能服务和默认撤防配置均已部署。
NOTE: 此结果证明软件部署结构；真机在线状态需另用 ros2 node/topic/service 检查。
EOF
else
  cat <<'EOF'
RESULT: FAIL — 当前工作树缺少一个或多个双门控部署要素，请先修复再提交。
EOF
  exit 1
fi
