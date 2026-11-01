#!/usr/bin/env bash
# Best-effort, bounded-time shutdown of every motion-producing layer.

set -uo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly UNITREE_SETUP="${HOME}/learning/unitree_ros2/setup_go2w_humble.sh"
readonly WORKSPACE_SETUP="${PROJECT_ROOT}/install/setup.bash"
readonly CALL_TIMEOUT_S="${RL2_STOP_TIMEOUT_S:-3}"

set +u
if [[ -f "${UNITREE_SETUP}" ]]; then
  # shellcheck disable=SC1090
  source "${UNITREE_SETUP}"
elif [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi
if [[ -f "${WORKSPACE_SETUP}" ]]; then
  # shellcheck disable=SC1090
  source "${WORKSPACE_SETUP}"
fi
set -u

if ! command -v ros2 >/dev/null 2>&1; then
  echo "[STOP][FATAL] ros2 is unavailable; no ROS stop request was sent." >&2
  exit 2
fi

call_set_bool_false() {
  local service_name="$1"
  local required="$2"
  echo "[STOP] disabling ${service_name}"
  if timeout "${CALL_TIMEOUT_S}s" ros2 service call \
    "${service_name}" std_srvs/srv/SetBool '{data: false}' >/tmp/rl2_stop_call.log 2>&1; then
    sed -n '/response:/,$p' /tmp/rl2_stop_call.log
    return 0
  fi
  echo "[STOP][WARN] ${service_name} unavailable or timed out." >&2
  [[ "${required}" == "optional" ]]
}

call_trigger() {
  local service_name="$1"
  echo "[STOP] requesting ${service_name}"
  if timeout "${CALL_TIMEOUT_S}s" ros2 service call \
    "${service_name}" std_srvs/srv/Trigger '{}' >/tmp/rl2_stop_call.log 2>&1; then
    sed -n '/response:/,$p' /tmp/rl2_stop_call.log
    return 0
  fi
  echo "[STOP][INFO] optional ${service_name} unavailable or timed out."
  return 0
}

critical_failures=0

# Stop upstream goal producers first, then both downstream velocity gates.
# Notify the box coordinator first so an operator-requested stop is recorded
# as a reusable ABORTED state rather than an apparent coordinator fault.
call_trigger /three_target_box_mission/abort
call_trigger /mission_supervisor_node/stop
call_set_bool_false /frontier_explorer_node/enable optional || true
call_set_bool_false /cmd_vel_safety_node/enable required || critical_failures=$((critical_failures + 1))
call_set_bool_false /go2w_sport_cmd_bridge/enable required || critical_failures=$((critical_failures + 1))

# A second bridge disable produces another bounded API 1003 StopMove and also
# covers a command that may have raced with the first pass.  This repeat is
# belt-and-suspenders: the first disable above already disarmed the bridge and
# sent StopMove, and the bridge service can be busy finishing that API call
# (observed 2026-09-30 during measurement rounds), so a timeout here is a
# warning, not a critical failure.
call_set_bool_false /go2w_sport_cmd_bridge/enable optional || true

if (( critical_failures > 0 )); then
  echo "[STOP][FATAL] ${critical_failures} critical stop request(s) failed." >&2
  echo "Use the remote-control lock immediately and inspect the ROS launch." >&2
  exit 2
fi

echo "[STOP][OK] Frontier/mission producers are stopped where present."
echo "[STOP][OK] Velocity gate is disabled and Sport bridge sent API 1003 StopMove."
