#!/usr/bin/env bash
# Single-entry operator tool for the user-PC navigation stack.
#
# Subcommands:
#   ps                     List this workspace's stack processes (PID/PPID/
#                          role) and flag orphans (matched processes whose
#                          parent launch is gone).  No ROS environment needed.
#   status [--watch [S]]   Process summary + one latched JSON line per state
#                          topic (pose-jump guard, slam_health, both motion
#                          gates, mission supervisor) + dock ping.
#                          --watch redrawes every 2s (optionally only S seconds).
#   kill [--dry-run]       stop_all_motion.sh -> SIGINT launch (8s) ->
#                          SIGTERM leftovers (3s) -> SIGKILL survivors, then
#                          re-checks that nothing matched remains.
#   start [args...]        Exec start_base_target_posture_stack.sh (foreground,
#                          Ctrl+C semantics preserved).  When no mission mode
#                          or target count is supplied, defaults to the
#                          three-target storage-box mission.
#   restart [args...]      kill, then start.
#
# Safety: NEVER touches dock (192.168.123.18) processes -- dock SLAM is managed
# by ensure_dock_slam_stack.sh -- and NEVER changes any motion-gate enable
# state.  Orphan sweeping here only kills processes that belong to this
# workspace's launch stack by cmdline pattern.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UNITREE_SETUP="${UNITREE_SETUP:-/home/kazuhalry/learning/unitree_ros2/setup_go2w_humble.sh}"
DOCK_IP="${RL2_DOCK_IP:-192.168.123.18}"

usage() {
  cat <<'EOF'
Usage: ./scripts/rl2_stack.sh <command> [options]

Commands:
  ps                    Show stack process table (launch, nodes, ORPHAN flags).
  status [--watch [S]]  Show process summary, latched status topics and dock
                        reachability.  --watch redrawes every 2s; give seconds
                        to stop after S (default: run until Ctrl+C).
  kill [--dry-run]      One-key stop of the whole user-PC stack with an orphan
                        sweep (SIGINT launch -> SIGTERM -> SIGKILL).  Dock
                        processes are never touched; the stage-1 software
                        stop disables both motion gates (re-open manually).
  start [args...]       Run start_base_target_posture_stack.sh with args.
                        Defaults to --three-target-box unless an explicit
                        mission mode or --target-count is supplied.
  restart [args...]     kill, then start.

Examples:
  ./scripts/rl2_stack.sh ps
  ./scripts/rl2_stack.sh status --watch 30
  ./scripts/rl2_stack.sh kill --dry-run
  ./scripts/rl2_stack.sh restart --skip-dock
EOF
}

# ---------------------------------------------------------------------------
# Process matcher (shared by ps/status/kill)
# ---------------------------------------------------------------------------

# Print the stack role name for one cmdline, or nothing if it is not ours.
_stack_role() {
  local args="$1"
  case "$args" in
    *"scripts/three_target_box_mission.py"*)
      printf 'three-target-coordinator'
      return 0
      ;;
  esac
  case "$args" in
    *"bin/ros2 launch rl2_nav "*)
      printf 'ros2-launch'
      return 0
      ;;
  esac
  case "$args" in
    # Anything executing from this workspace's install tree: python nodes,
    # the YOLO perception node, and RViz with the workspace config.
    *"$PROJECT_ROOT/install/rl2_"*)
      ;;
    # System ROS binaries launched with this launch's temp params files.
    *"/opt/ros/humble/lib/nav2_"*|*"/opt/ros/humble/lib/pointcloud_to_laserscan"*)
      case "$args" in
        *"--params-file /tmp/launch_params_"*) ;;
        *) return 1 ;;
      esac
      ;;
    *)
      return 1
      ;;
  esac
  local name="${args##*__node:=}"
  if [[ "$name" != "$args" && -n "$name" ]]; then
    printf '%s' "${name%% *}"
    return 0
  fi
  local exe
  exe="$(printf '%s' "$args" | awk '{print $2}')"
  printf '%s' "${exe##*/}"
}

# Print "pid|ppid|etimes|role" for every matched stack process.
_collect() {
  local pid ppid etimes args role
  while read -r pid ppid etimes args; do
    [[ -n "${pid:-}" ]] || continue
    role="$(_stack_role "$args")"
    [[ -n "$role" ]] || continue
    printf '%s|%s|%s|%s\n' "$pid" "$ppid" "$etimes" "$role"
  done < <(ps -eo pid=,ppid=,etimes=,args=)
}

_collect_pids() {
  _collect | awk -F'|' '{print $1}'
}

_launch_pids() {
  _collect | awk -F'|' '$4 == "ros2-launch" {print $1}'
}

# ---------------------------------------------------------------------------
# ps
# ---------------------------------------------------------------------------

cmd_ps() {
  local rows=() launch_pids=()
  local pid ppid etimes role
  while IFS='|' read -r pid ppid etimes role; do
    [[ -n "$pid" ]] || continue
    rows+=("$pid|$ppid|$etimes|$role")
    [[ "$role" == "ros2-launch" ]] && launch_pids+=("$pid")
  done < <(_collect)

  echo "Stack processes for $PROJECT_ROOT"
  if ((${#rows[@]} == 0)); then
    echo "  (none running)"
    return 0
  fi
  printf '%-8s %-8s %-9s %-7s %s\n' PID PPID AGE_S FLAGS ROLE
  local row flag orphans=0
  for row in "${rows[@]}"; do
    IFS='|' read -r pid ppid etimes role <<<"$row"
    flag=""
    if [[ "$role" != "ros2-launch" && "$role" != "three-target-coordinator" ]]; then
      if ! printf '%s\n' "${launch_pids[@]:-}" | grep -qx -- "$ppid"; then
        flag="ORPHAN"
        orphans=$((orphans + 1))
      fi
    fi
    printf '%-8s %-8s %-9s %-7s %s\n' "$pid" "$ppid" "$etimes" "$flag" "$role"
  done
  echo "--"
  echo "launch: ${#launch_pids[@]}  nodes: $(( ${#rows[@]} - ${#launch_pids[@]} ))  orphans: $orphans"
  if ((orphans > 0)); then
    echo "ORPHAN = stack node without a live parent launch (old stack leftover)."
    echo "Clean up with: ./scripts/rl2_stack.sh kill"
  fi
}

# ---------------------------------------------------------------------------
# ROS environment (status/kill only; ps and start never need it)
# ---------------------------------------------------------------------------

_load_env() {
  if [[ -n "${RL2_STACK_ENV_LOADED:-}" ]]; then
    return 0
  fi
  if [[ ! -f "$UNITREE_SETUP" ]]; then
    echo "[STACK][WARN] Unitree setup not found: $UNITREE_SETUP" >&2
    return 1
  fi
  set +u
  # shellcheck disable=SC1090
  source "$UNITREE_SETUP" >/dev/null
  if [[ -f "$PROJECT_ROOT/install/setup.bash" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/install/setup.bash"
  fi
  set -u
  if ! command -v ros2 >/dev/null 2>&1; then
    echo "[STACK][WARN] ros2 is unavailable after sourcing the setups." >&2
    return 1
  fi

  local dds_interface
  dds_interface="$(ip -o -4 addr show up \
    | awk '$4 ~ /^192\.168\.123\./ {print $2; exit}')"
  if [[ -n "$dds_interface" ]]; then
    local dds_config="/tmp/rl2_cyclonedds_${USER}.xml"
    cat >"$dds_config" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface name="$dds_interface" priority="default" multicast="default"/>
      </Interfaces>
      <AllowMulticast>spdp</AllowMulticast>
    </General>
  </Domain>
</CycloneDDS>
EOF
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export ROS_DOMAIN_ID=0
    export CYCLONEDDS_URI="file://$dds_config"
  else
    echo "[STACK][WARN] no 192.168.123.x interface; topic reads may time out." >&2
  fi
  unset SNAP SNAP_NAME SNAP_INSTANCE_NAME SNAP_REVISION SNAP_ARCH SNAP_VERSION
  unset SNAP_LIBRARY_PATH GTK_PATH GTK_EXE_PREFIX GTK_IM_MODULE_FILE
  RL2_STACK_ENV_LOADED=1
  return 0
}

# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

_read_latched() {
  # ros2 topic echo --once appends a '---' message separator after the
  # value; drop it so the captured string stays pure JSON.
  timeout 3s ros2 topic echo "$1" std_msgs/msg/String --field data \
    --once --no-daemon --spin-time 1 2>/dev/null | sed '/^---$/d' || true
}

_json_get() { # $1 = json string, $2 = key
  printf '%s' "$1" | python3 -c '
import json, sys
try:
    payload = json.loads(sys.stdin.read())
except (ValueError, json.JSONDecodeError):
    sys.exit(1)
value = payload.get(sys.argv[1])
if isinstance(value, (dict, list)):
    print(json.dumps(value, separators=(",", ":")))
else:
    print(value)
' "$2" 2>/dev/null
}

_status_once() {
  echo "===== rl2_stack status  $(date '+%F %T') ====="

  local launch_pids=() orphans=0 total=0 pid ppid etimes role
  local launch_pid_list
  launch_pid_list="$(_launch_pids)"
  while IFS='|' read -r pid ppid etimes role; do
    [[ -n "$pid" ]] || continue
    total=$((total + 1))
    if [[ "$role" == "ros2-launch" ]]; then
      launch_pids+=("$pid (${etimes}s)")
    elif ! grep -qx -- "$ppid" <<<"$launch_pid_list"; then
      orphans=$((orphans + 1))
    fi
  done < <(_collect)

  if ((total == 0)); then
    echo "[procs ] stack NOT running (no matched processes)"
  else
    echo "[procs ] launch=${#launch_pids[@]} [${launch_pids[*]:-}]  matched=$total  orphans=$orphans"
  fi

  local dock_state="unreachable"
  ping -c1 -W1 "$DOCK_IP" >/dev/null 2>&1 && dock_state="reachable"
  echo "[dock  ] $DOCK_IP $dock_state (ping only; no SSH)"

  if ((total == 0)); then
    echo "[topics] skipped: no stack process is running."
    return 0
  fi
  if ! _load_env; then
    echo "[topics] skipped: ROS environment unavailable."
    return 0
  fi

  local guard health gate1 gate2 supervisor coordinator storage_box
  guard="$(_read_latched /rl2_nav/slam_pose_jump_guard/status)"
  if [[ -n "$guard" ]]; then
    echo "[guard  ] $guard"
  else
    echo "[guard  ] no latched message (guard node not running?)"
  fi

  health="$(_read_latched /rl2_nav/slam_health)"
  if [[ -n "$health" ]]; then
    echo "[health ] $health"
  else
    echo "[health ] no latched message (slam_health unavailable)"
  fi

  gate1="$(_read_latched /cmd_vel_safety_node/status)"
  if [[ -n "$gate1" ]]; then
    echo "[gate-1 ] cmd_vel_safety enabled=$(_json_get "$gate1" enabled) \
odom_jump_monitoring=$(_json_get "$gate1" odom_jump_monitoring) reason=$(_json_get "$gate1" reason)"
    echo "[gate-1 ] $gate1"
  else
    echo "[gate-1 ] topic missing (node not running or old build without the status topic)"
  fi

  gate2="$(_read_latched /go2w_sport_cmd_bridge/status)"
  if [[ -n "$gate2" ]]; then
    local gate2_enabled
    gate2_enabled="$(_json_get "$gate2" enabled)"
    if [[ "$gate2_enabled" == "None" ]]; then
      gate2_enabled="UNKNOWN (old bridge build without enabled field)"
    fi
    echo "[gate-2 ] sport_bridge enabled=$gate2_enabled stage=$(_json_get "$gate2" stage)"
    echo "[gate-2 ] $gate2"
  else
    echo "[gate-2 ] topic missing (node not running or old build without the status topic)"
  fi

  supervisor="$(_read_latched /mission_supervisor_node/status)"
  if [[ -n "$supervisor" ]]; then
    echo "[super  ] state=$(_json_get "$supervisor" state) \
targets=$(_json_get "$supervisor" reached_targets)/$(_json_get "$supervisor" target_goal_count) \
goal=$(_json_get "$supervisor" active_goal) guard=$(_json_get "$supervisor" guard_state)"
    echo "[super  ] last_event=$(_json_get "$supervisor" last_event)"
  else
    echo "[super  ] topic missing (node not running or old build without the status topic)"
  fi

  coordinator="$(_read_latched /three_target_box_mission/status)"
  if [[ -n "$coordinator" ]]; then
    echo "[coord  ] phase=$(_json_get "$coordinator" phase) armed=$(_json_get "$coordinator" armed) return_confirmed=$(_json_get "$coordinator" return_confirmed)"
    echo "[coord  ] detail=$(_json_get "$coordinator" detail)"
  else
    echo "[coord  ] topic missing (three-target coordinator not running?)"
  fi

  storage_box="$(_read_latched /storage_box/status)"
  if [[ -n "$storage_box" ]]; then
    echo "[box    ] state=$(_json_get "$storage_box" state) commanded=$(_json_get "$storage_box" commanded_state)"
    echo "[box    ] detail=$(_json_get "$storage_box" detail)"
  else
    echo "[box    ] topic missing (storage-box bridge not running?)"
  fi
}

cmd_status() {
  local watch="$1" duration="$2" elapsed=0
  while :; do
    if [[ "$watch" == true ]]; then
      clear 2>/dev/null || printf '\033[2J\033[H'
    fi
    _status_once
    if [[ "$watch" != true ]]; then
      return 0
    fi
    if ((duration > 0)); then
      elapsed=$((elapsed + 2))
      ((elapsed >= duration)) && return 0
    fi
    sleep 2
  done
}

# ---------------------------------------------------------------------------
# kill
# ---------------------------------------------------------------------------

_wait_gone() { # $1 = seconds
  local deadline=$((SECONDS + $1)) remaining
  while ((SECONDS < deadline)); do
    remaining="$(_collect_pids | wc -l)"
    ((remaining == 0)) && return 0
    sleep 0.5
  done
  return 1
}

_signal_all() { # $1 = signal, $2 = wait seconds
  local sig="$1" wait_s="$2"
  local pids=()
  mapfile -t pids < <(_collect_pids)
  if ((${#pids[@]} == 0)); then
    return 0
  fi
  echo "[KILL]   $sig -> ${pids[*]}"
  # shellcheck disable=SC2086
  kill -"$sig" "${pids[@]}" 2>/dev/null || true
  _wait_gone "$wait_s" || true
}

cmd_kill() {
  local dry_run=false
  if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
  elif [[ -n "${1:-}" ]]; then
    echo "Unknown kill option: $1" >&2
    usage >&2
    exit 2
  fi

  if [[ "$dry_run" == true ]]; then
    echo "[KILL] dry run: processes that WOULD be stopped:"
    cmd_ps
    echo "[KILL] dry run: no signal was sent."
    return 0
  fi

  echo "[KILL] step 1/4: software stop of every motion layer (best-effort)..."
  if _load_env; then
    timeout 20s "$SCRIPT_DIR/stop_all_motion.sh" \
      || echo "[KILL][WARN] stop_all_motion.sh failed; continuing with signals."
  else
    echo "[KILL][WARN] ROS env unavailable; skipping the software stop."
  fi

  echo "[KILL] step 2/4: SIGINT to ros2 launch (graceful shutdown, up to 8s)..."
  local launch=()
  mapfile -t launch < <(_launch_pids)
  if ((${#launch[@]} > 0)); then
    echo "[KILL]   INT -> ${launch[*]}"
    kill -INT "${launch[@]}" 2>/dev/null || true
    _wait_gone 8 || true
  else
    echo "[KILL]   no live launch process."
  fi

  echo "[KILL] step 3/4: SIGTERM to leftovers (up to 3s)..."
  _signal_all TERM 3

  echo "[KILL] step 4/4: SIGKILL to survivors (up to 2s)..."
  _signal_all KILL 2

  local survivors
  survivors="$(_collect_pids | tr '\n' ' ')"
  if [[ -n "${survivors// /}" ]]; then
    echo "[KILL][FATAL] still alive after SIGKILL: $survivors" >&2
    echo "[KILL][FATAL] inspect them manually before restarting the stack." >&2
    return 1
  fi
  echo "[KILL][OK] stack fully stopped; no matched processes remain."
  echo "[KILL] dock ($DOCK_IP) processes were NOT touched. The software stop"
  echo "[KILL] in stage 1 disabled both motion gates; re-open them manually."
  return 0
}

# ---------------------------------------------------------------------------
# start / restart
# ---------------------------------------------------------------------------

cmd_start() {
  # Keep the operator entry point deterministic.  Enforcing the competition
  # default here prevents an unnoticed two-target launch while explicit
  # diagnostic modes remain intact.
  local explicit_mission=false
  local arg
  for arg in "$@"; do
    case "$arg" in
      --three-target-box|--single-target-posture|--single-target-arm|--target-count)
        explicit_mission=true
        break
        ;;
    esac
  done
  if [[ "$explicit_mission" == false ]]; then
    exec "$SCRIPT_DIR/start_base_target_posture_stack.sh" \
      "$@" --three-target-box
  fi
  exec "$SCRIPT_DIR/start_base_target_posture_stack.sh" "$@"
}

cmd_restart() {
  cmd_kill || {
    echo "[RESTART][FATAL] kill left survivors; refusing to start a second stack." >&2
    exit 1
  }
  echo "[RESTART] starting the stack..."
  cmd_start "$@"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

command="${1:-}"
shift || true
case "$command" in
  ps)
    cmd_ps "$@"
    ;;
  status)
    watch=false
    duration=0
    while (($#)); do
      case "$1" in
        --watch)
          watch=true
          if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
            duration="$2"
            shift
          fi
          ;;
        *)
          echo "Unknown status option: $1" >&2
          usage >&2
          exit 2
          ;;
      esac
      shift
    done
    cmd_status "$watch" "$duration"
    ;;
  kill)
    cmd_kill "$@"
    ;;
  start)
    cmd_start "$@"
    ;;
  restart)
    cmd_restart "$@"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown command: $command" >&2
    usage >&2
    exit 2
    ;;
esac
