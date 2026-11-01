#!/usr/bin/env bash
# Read-only preflight for navigation -> ARX X5 automatic grasp integration.
# This script never publishes a topic, calls a service, or opens the CAN bus.

set -uo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly UNITREE_SETUP="${HOME}/learning/unitree_ros2/setup_go2w_humble.sh"
readonly WORKSPACE_SETUP="${PROJECT_ROOT}/install/setup.bash"
readonly MODE="${1:-real}"

if [[ "${MODE}" != "real" && "${MODE}" != "dry-run" ]]; then
  echo "usage: $0 [real|dry-run]" >&2
  exit 64
fi

failures=0
warnings=0

pass() { echo "[PASS] $*"; }
warn() { echo "[WARN] $*"; warnings=$((warnings + 1)); }
fail() { echo "[FAIL] $*"; failures=$((failures + 1)); }

echo '=== Environment ==='
set +u
if [[ -f "${UNITREE_SETUP}" ]]; then
  # shellcheck disable=SC1090
  source "${UNITREE_SETUP}"
else
  fail "missing Unitree setup: ${UNITREE_SETUP}"
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi
if [[ -f "${WORKSPACE_SETUP}" ]]; then
  # shellcheck disable=SC1090
  source "${WORKSPACE_SETUP}"
else
  fail "workspace is not built: ${WORKSPACE_SETUP}"
fi
set -u

for package in rl2_interfaces rl2_nav rl2_manipulation robot_arm; do
  if ros2 pkg prefix "${package}" >/dev/null 2>&1; then
    pass "ROS package available: ${package}"
  else
    fail "ROS package missing: ${package}"
  fi
done

echo '=== Hardware visibility ==='
if lsusb 2>/dev/null | grep -Eqi '2bc5|Orbbec|Gemini'; then
  pass 'Gemini/Orbbec USB camera is visible'
else
  fail 'Gemini/Orbbec USB camera is not visible in lsusb'
fi
if compgen -G '/dev/arxcan*' >/dev/null; then
  pass "ARX USB-CAN device exists: $(printf '%s ' /dev/arxcan*)"
else
  [[ "${MODE}" == "real" ]] && fail 'no /dev/arxcan* device found' || \
    warn 'no /dev/arxcan* device found (allowed only in dry-run)'
fi
if ip -brief link show can1 >/dev/null 2>&1; then
  state="$(ip -brief link show can1 | awk '{print $2}')"
  if [[ "${state}" == "UP" ]]; then
    pass 'can1 is UP'
  else
    [[ "${MODE}" == "real" ]] && fail "can1 exists but is ${state}" || \
      warn "can1 exists but is ${state} (allowed only in dry-run)"
  fi
else
  [[ "${MODE}" == "real" ]] && fail 'can1 does not exist' || \
    warn 'can1 does not exist (allowed only in dry-run)'
fi

echo '=== Exclusive ownership ==='
conflicts="$(pgrep -af 'rl_real_go2|rl_sar|robot-arm|test_everang.py|test_grasp.py' || true)"
if [[ -n "${conflicts}" ]]; then
  fail 'conflicting motion/arm process detected:'
  printf '%s\n' "${conflicts}"
else
  pass 'no known conflicting motion/arm process'
fi

echo '=== Live ROS graph (read-only) ==='
if timeout 5s ros2 action list -t 2>/dev/null | \
    grep -Fq '/arm/grasp_target [rl2_interfaces/action/GraspTarget]'; then
  pass '/arm/grasp_target Action server is visible'
else
  fail '/arm/grasp_target Action server is not visible'
fi

arm_status_publishers="$(timeout 5s ros2 topic info /arm/status --no-daemon 2>/dev/null | \
  awk '/Publisher count:/ {print $3}' | tail -1)"
if [[ "${arm_status_publishers:-0}" -ge 1 ]]; then
  pass '/arm/status has a publisher'
else
  fail '/arm/status has no publisher'
fi

vision_publishers="$(timeout 5s ros2 topic info /robot_arm/target_pose --no-daemon 2>/dev/null | \
  awk '/Publisher count:/ {print $3}' | tail -1)"
if [[ "${vision_publishers:-0}" -ge 1 ]]; then
  pass '/robot_arm/target_pose has a publisher'
else
  fail '/robot_arm/target_pose has no publisher (arm vision is not ready)'
fi

if timeout 5s ros2 node list --no-daemon 2>/dev/null | grep -Fxq '/mission_supervisor_node'; then
  integration="$(timeout 5s ros2 param get /mission_supervisor_node arm_integration_enabled 2>/dev/null | tail -1)"
  posture="$(timeout 5s ros2 param get /mission_supervisor_node target_posture_enabled 2>/dev/null | tail -1)"
  [[ "${integration}" == 'Boolean value is: True' ]] && \
    pass 'mission supervisor arm integration is enabled' || \
    fail "arm integration is not enabled: ${integration:-unknown}"
  [[ "${posture}" == 'Boolean value is: False' ]] && \
    pass 'target crouch/posture is disabled' || \
    fail "target posture must be false for this test: ${posture:-unknown}"
else
  warn 'mission_supervisor_node is not running; live parameters were not checked'
fi

echo '=== Result ==='
echo "failures=${failures} warnings=${warnings}"
if (( failures > 0 )); then
  echo 'NOT READY: do not enable the bottom velocity gates or start a real grasp.'
  exit 2
fi
echo "READY FOR STAGED ${MODE} TEST. This result does not authorize unattended robot motion."
