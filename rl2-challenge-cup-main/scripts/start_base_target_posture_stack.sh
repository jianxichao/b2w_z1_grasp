#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unitree_setup="${UNITREE_SETUP:-/home/kazuhalry/learning/unitree_ros2/setup_go2w_humble.sh}"
start_rviz=true
restart_dock=false
skip_dock=false
build_first=false
mission_mode="two-target-posture"
target_goal_count=2
target_crouch_hold_s=3.0
target_posture_enabled=true
stop_after_target_interaction=false
arm_integration_enabled=false
start_box_coordinator=false

usage() {
  cat <<'EOF'
Usage: ./scripts/start_base_target_posture_stack.sh [options]

Options:
  --restart-dock  Restart dock SLAM, Mid360 and D435i before launch.
  --skip-dock     Do not SSH to the dock (use only when sensors are already up).
  --no-rviz       Start the stack without RViz.
  --build         Build rl2_hardware, rl2_nav and rl2_perception first.
  --target-count N
                  Number of targets before returning to base (default 2).
  --crouch-hold-s X
                  Seconds to stay crouched at each target (default 3.0).
  --three-target-box
                  Three targets, crouch 5s each, no arm; storage box closes
                  before mission start and opens only after base arrival and
                  a stable stop. Mission and gates still require explicit calls.
  --single-target-posture
                  One target: approach, crouch/stand, then remain idle.
  --single-target-arm
                  One target: approach, Gemini/ARX5 grasp, then return to base.
                  The arm runtime must be started separately on the dock.
  -h, --help      Show this help.

Safety: this script starts sensing/navigation nodes and API 1801 mapping, but
keeps both velocity gates disabled and keeps the mission stopped. It cannot
move the robot by itself.
EOF
}

while (($#)); do
  case "$1" in
    --restart-dock) restart_dock=true ;;
    --skip-dock) skip_dock=true ;;
    --no-rviz) start_rviz=false ;;
    --build) build_first=true ;;
    --target-count)
      (($# >= 2)) || { echo "--target-count requires a value" >&2; exit 2; }
      target_goal_count="$2"; shift ;;
    --crouch-hold-s)
      (($# >= 2)) || { echo "--crouch-hold-s requires a value" >&2; exit 2; }
      target_crouch_hold_s="$2"; shift ;;
    --three-target-box)
      mission_mode="three-target-box-posture"
      target_goal_count=3
      target_crouch_hold_s=5.0
      target_posture_enabled=true
      stop_after_target_interaction=false
      arm_integration_enabled=false
      start_box_coordinator=true
      ;;
    --single-target-posture)
      mission_mode="single-target-posture"
      target_goal_count=1
      target_posture_enabled=true
      stop_after_target_interaction=true
      arm_integration_enabled=false
      ;;
    --single-target-arm)
      mission_mode="single-target-arm-return"
      target_goal_count=1
      target_posture_enabled=false
      stop_after_target_interaction=false
      arm_integration_enabled=true
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

cd "$project_root"

if [[ ! -f "$unitree_setup" ]]; then
  echo "ERROR: Unitree ROS 2 setup not found: $unitree_setup" >&2
  exit 1
fi

if pgrep -af 'ros2 launch rl2_nav autonomous_mapping_sport_bringup.launch.py' \
    >/dev/null; then
  echo "ERROR: the base target/posture stack already appears to be running." >&2
  echo "Stop the old launch first with Ctrl+C and ./scripts/stop_all_motion.sh." >&2
  exit 1
fi

# Orphan pre-check: workspace stack nodes without a live launch are the
# double-stack hazard observed on 2026-09-26 (a crashed launch left 14
# orphaned nodes, so every node ran twice and two bridges consumed
# /cmd_vel_sport).  Refuse to start on top of them instead of killing
# silently.
orphan_list="$(
  { pgrep -af "$project_root/install/rl2_" || true;
    pgrep -af '/opt/ros/humble/lib/nav2_.*launch_params_' || true;
    pgrep -af '/opt/ros/humble/lib/pointcloud_to_laserscan.*launch_params_' || true;
  } || true
)"
if [[ -n "$orphan_list" ]]; then
  echo "ERROR: orphaned stack processes are still running (no live launch):" >&2
  printf '%s\n' "$orphan_list" >&2
  echo "Clean them up first with: ./scripts/rl2_stack.sh kill" >&2
  exit 1
fi

if [[ "$skip_dock" == false ]]; then
  dock_args=(--with-camera)
  if [[ "$restart_dock" == true ]]; then
    dock_args=(--restart --with-camera)
  fi
  "$project_root/scripts/ensure_dock_slam_stack.sh" "${dock_args[@]}"
fi

set +u
# shellcheck disable=SC1090
source "$unitree_setup"
if [[ -f "$project_root/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "$project_root/install/setup.bash"
fi
set -u

dds_interface="$(ip -o -4 addr show up | awk '$4 ~ /^192\.168\.123\./ {print $2; exit}')"
if [[ -z "$dds_interface" ]]; then
  echo "ERROR: no UP IPv4 interface in 192.168.123.0/24 was found." >&2
  echo "Reconnect the robot Ethernet cable, then run: ip -br addr" >&2
  exit 1
fi

dds_config="/tmp/rl2_cyclonedds_${USER}.xml"
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

# When launched from the Snap build of VS Code, child GUI processes can
# inherit core20 GTK/library hints. RViz is a host binary and must not load
# glibc/libpthread from that Snap runtime.
unset SNAP SNAP_NAME SNAP_INSTANCE_NAME SNAP_REVISION SNAP_ARCH SNAP_VERSION
unset SNAP_LIBRARY_PATH GTK_PATH GTK_EXE_PREFIX GTK_IM_MODULE_FILE

if [[ "$build_first" == true ]]; then
  echo "Building the base-only stack..."
  colcon build \
    --packages-select rl2_interfaces rl2_hardware rl2_nav rl2_perception \
    --symlink-install
  set +u
  # shellcheck disable=SC1091
  source "$project_root/install/setup.bash"
  set -u
fi

if [[ ! -f "$project_root/install/setup.bash" ]]; then
  echo "ERROR: install/setup.bash is absent. Re-run with --build." >&2
  exit 1
fi

run_stamp="$(date +%Y%m%d_%H%M%S)"
export ROS_LOG_DIR="/tmp/rl2_base_target_posture_${run_stamp}"
mkdir -p "$ROS_LOG_DIR"

coordinator_pid=""
if [[ "$start_box_coordinator" == true ]]; then
  "$project_root/scripts/start_storage_box_bridge.sh"
  if pgrep -af 'scripts/[t]hree_target_box_mission.py' >/dev/null; then
    echo "ERROR: a three-target coordinator from an earlier stack is still running." >&2
    echo "Refusing to reuse it because its state/parameters may not match this launch." >&2
    echo "Run ./scripts/rl2_stack.sh kill, then start again with --three-target-box." >&2
    exit 1
  else
    python3 "$project_root/scripts/three_target_box_mission.py" \
      --ros-args -p target_count:="$target_goal_count" &
    coordinator_pid=$!
    echo "Started three-target storage-box coordinator (PID $coordinator_pid)."
  fi
elif pgrep -af 'scripts/[t]hree_target_box_mission.py' >/dev/null; then
  echo "ERROR: a three-target coordinator is running, but this launch is not in" >&2
  echo "--three-target-box mode. This would route a start request to a" >&2
  echo "coordinator whose target count differs from the mission supervisor." >&2
  echo "Run ./scripts/rl2_stack.sh kill, then choose the intended mode explicitly." >&2
  exit 1
fi

readiness_check() {
  local attempt
  local readiness_failed=false
  echo "Waiting for local ROS 2 services..."
  for attempt in $(seq 1 45); do
    if ros2 service list --no-daemon 2>/dev/null | \
        grep -qx '/unitree_slam/start_mapping'; then
      break
    fi
    sleep 1
  done
  if ! ros2 service list --no-daemon 2>/dev/null | \
      grep -qx '/unitree_slam/start_mapping'; then
    echo "READINESS FAILED: /unitree_slam/start_mapping is unavailable." >&2
    return 1
  fi

  echo "Starting a fresh official mapping session (API 1801)..."
  ros2 service call /unitree_slam/start_mapping std_srvs/srv/Trigger '{}' \
    || return 1
  sleep 5

  echo "=== One-key initialization report ==="
  echo "DDS interface: $dds_interface"
  ros2 lifecycle get /planner_server 2>/dev/null || true
  ros2 lifecycle get /controller_server 2>/dev/null || true
  ros2 action info /navigate_to_pose 2>/dev/null || true

  if timeout 8s ros2 topic echo /slam/current_pose \
      geometry_msgs/msg/PoseStamped --once --no-daemon --spin-time 6 \
      >/dev/null 2>&1; then
    echo "PASS: fresh SLAM pose received"
  else
    echo "READINESS FAILED: no fresh /slam/current_pose within the check window"
    readiness_failed=true
  fi
  if timeout 8s ros2 topic echo /d435i/color/image_raw \
      sensor_msgs/msg/Image --once --no-daemon --spin-time 6 \
      >/dev/null 2>&1; then
    echo "PASS: D435i color image received"
  else
    echo "READINESS FAILED: no D435i color image within the check window"
    readiness_failed=true
  fi
  if timeout 8s ros2 topic echo /target_detection_status \
      std_msgs/msg/String --once --no-daemon --spin-time 6 \
      >/dev/null 2>&1; then
    echo "PASS: target perception status received"
  else
    echo "READINESS FAILED: no target perception status within the check window"
    readiness_failed=true
  fi
  if ! python3 "$project_root/scripts/stack_readiness_check.py" startup \
      --target-count "$target_goal_count" --timeout 10; then
    echo "READINESS FAILED: coordinated stack preflight did not pass" >&2
    readiness_failed=true
  fi
  if [[ "$readiness_failed" == true ]]; then
    echo "Initialization is incomplete. Motion gates remain disabled." >&2
    return 1
  fi
  echo "READY (motion disabled): required SLAM and perception inputs are healthy."
  echo "Log directory: $ROS_LOG_DIR"
}

launch_pid=""
cleanup() {
  trap - INT TERM EXIT
  if [[ -n "$launch_pid" ]]; then
    kill -INT "$launch_pid" 2>/dev/null || true
    wait "$launch_pid" 2>/dev/null || true
  fi
  if [[ -n "$coordinator_pid" ]]; then
    kill "$coordinator_pid" 2>/dev/null || true
  fi
  "$project_root/scripts/stop_all_motion.sh" >/dev/null 2>&1 || true
}
trap cleanup INT TERM EXIT

echo "Starting RL2 mission stack: mode=$mission_mode"
echo "Motion gates remain disabled and mission auto-start is disabled."
echo "Mission parameters: target_goal_count=$target_goal_count, target_crouch_hold_s=$target_crouch_hold_s, target_posture_enabled=$target_posture_enabled, stop_after_target_interaction=$stop_after_target_interaction, arm_integration_enabled=$arm_integration_enabled"
echo "Press Ctrl+C here to stop the stack safely."

ros2 launch rl2_nav autonomous_mapping_sport_bringup.launch.py \
  start_rviz:="$start_rviz" \
  start_perception:=true \
  start_supervisor:=true \
  mission_auto_start:=false \
  target_goal_count:="$target_goal_count" \
  target_crouch_hold_s:="$target_crouch_hold_s" \
  target_posture_enabled:="$target_posture_enabled" \
  stop_after_target_interaction:="$stop_after_target_interaction" \
  lock_visual_goal_after_stable_candidate:=true \
  locked_approach_min_confidence:=0.70 \
  arm_integration_enabled:="$arm_integration_enabled" \
  start_arm:=false &
launch_pid=$!

# Readiness is a hard startup transaction boundary.  Previously this check
# ran in an independent background job: it could print READINESS FAILED while
# the launch kept running, leaving a half-initialized stack that blocked the
# next start attempt. A failed check now exits this script; the EXIT trap stops
# the launch/coordinator and closes the motion gates.
if ! readiness_check; then
  echo "READINESS FAILED: cleaning up this incomplete stack start." >&2
  exit 1
fi

set +e
wait "$launch_pid"
launch_rc=$?
set -e
launch_pid=""
exit "$launch_rc"
