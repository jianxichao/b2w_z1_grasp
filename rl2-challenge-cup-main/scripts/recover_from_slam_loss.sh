#!/usr/bin/env bash
# One-key recovery after the official SLAM stream died mid-mission.
#
# What happened: the dock-side unitree_slam stopped publishing, the pose-jump
# guard latched RECOVERING with both motion gates closed, and the mission
# supervisor aborted (mission_slam_lost:*).  This script restarts the dock
# sensor stack, starts a fresh API 1801 mapping session, and waits for
# /rl2_nav/slam_health to go green again.
#
# Safety: this script NEVER enables a motion gate.  Re-enabling
# /cmd_vel_safety_node/enable and /go2w_sport_cmd_bridge/enable stays a
# manual, deliberate operator action (see the printed checklist at the end).

set -uo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unitree_setup="${UNITREE_SETUP:-/home/kazuhalry/learning/unitree_ros2/setup_go2w_humble.sh}"
skip_dock=false
auto_acknowledge=false

usage() {
  cat <<'EOF'
Usage: ./scripts/recover_from_slam_loss.sh [options]

Recovers from an official SLAM stream loss:
  1. best-effort mission abort + software stop (never enables motion)
  2. restart unitree_slam / Mid360 / D435i on the dock (asks for SSH password)
  3. start a fresh API 1801 mapping session
  4. wait until /rl2_nav/slam_health reports slam_pose_ok and >=1000 map cells
  5. print the manual re-arm checklist (acknowledge guard, then both gates)

Options:
  --skip-dock        Do not SSH to the dock (use only if you restarted it
                     yourself, e.g. via ./scripts/ensure_dock_slam_stack.sh).
  --auto-acknowledge Also call /slam_pose_jump_guard/acknowledge once the new
                     map is stable.  Motion gates still stay disabled.
  -h, --help         Show this help.

Requires the main launch (start_base_target_posture_stack.sh) to be running;
otherwise this script exits after the dock restart and tells you how to
start it.
EOF
}

while (($#)); do
  case "$1" in
    --skip-dock) skip_dock=true ;;
    --auto-acknowledge) auto_acknowledge=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

cd "$project_root"

if [[ ! -f "$unitree_setup" ]]; then
  echo "[RECOVER][FATAL] Unitree ROS 2 setup not found: $unitree_setup" >&2
  exit 1
fi

set +u
# shellcheck disable=SC1090
source "$unitree_setup"
if [[ -f "$project_root/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "$project_root/install/setup.bash"
fi
set -u

if ! command -v ros2 >/dev/null 2>&1; then
  echo "[RECOVER][FATAL] ros2 is unavailable after sourcing the setups." >&2
  exit 1
fi

dds_interface="$(ip -o -4 addr show up | awk '$4 ~ /^192\.168\.123\./ {print $2; exit}')"
if [[ -z "$dds_interface" ]]; then
  echo "[RECOVER][FATAL] no UP IPv4 interface in 192.168.123.0/24 was found." >&2
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

unset SNAP SNAP_NAME SNAP_INSTANCE_NAME SNAP_REVISION SNAP_ARCH SNAP_VERSION
unset SNAP_LIBRARY_PATH GTK_PATH GTK_EXE_PREFIX GTK_IM_MODULE_FILE

echo "[RECOVER] Step 1/5: aborting the mission (best-effort)..."
if timeout 5s ros2 service call /mission_supervisor_node/abort \
    std_srvs/srv/Trigger '{}' >/tmp/rl2_recover_abort.log 2>&1; then
  sed -n '/response:/,$p' /tmp/rl2_recover_abort.log
else
  echo "[RECOVER][INFO] /mission_supervisor_node/abort unavailable; skipping."
fi

echo "[RECOVER] Step 2/5: software stop of every motion layer..."
if ! "$project_root/scripts/stop_all_motion.sh"; then
  echo "[RECOVER][FATAL] software stop failed; refusing to restart sensors while motion state is unknown." >&2
  echo "Engage the remote-control lock, repair the stop path, and retry." >&2
  exit 2
fi

if [[ "$skip_dock" == true ]]; then
  echo "[RECOVER] Step 3/5: --skip-dock given; not touching the dock."
else
  echo "[RECOVER] Step 3/5: restarting unitree_slam/Mid360/D435i on the dock"
  echo "(SSH password prompt may appear)..."
  if ! "$project_root/scripts/ensure_dock_slam_stack.sh" --restart --with-camera; then
    echo "[RECOVER][FATAL] dock restart failed (dock unreachable or SSH denied)." >&2
    echo "Engage the remote-control lock, check 192.168.123.18 connectivity, then retry." >&2
    exit 2
  fi
fi

# Order matters: after a dock SLAM restart, mapping odometry typically only
# starts flowing once a fresh mapping session (API 1801) is active.  Waiting
# for odom before the service call dead-ends here when the 30-loop times out,
# and the recovery never reaches start_mapping.  So: service check first,
# then 1801, then wait for the data it produces.
if ! timeout 5s ros2 service list --no-daemon 2>/dev/null | \
    grep -qx '/unitree_slam/start_mapping'; then
  echo "[RECOVER][FATAL] /unitree_slam/start_mapping is unavailable." >&2
  echo "The main launch is not running. Start it without touching the dock again:" >&2
  echo "  ./scripts/start_base_target_posture_stack.sh --skip-dock" >&2
  exit 2
fi

echo "[RECOVER] Step 4/5: starting a fresh official mapping session (API 1801)..."
if ! timeout 15s ros2 service call /unitree_slam/start_mapping \
    std_srvs/srv/Trigger '{}'; then
  echo "[RECOVER][FATAL] /unitree_slam/start_mapping call failed." >&2
  exit 2
fi

echo "[RECOVER] Waiting for fresh dock odometry on /unitree/slam_mapping/odom..."
odom_ok=false
for attempt in $(seq 1 30); do
  if timeout 6s ros2 topic echo /unitree/slam_mapping/odom \
      nav_msgs/msg/Odometry --once --no-daemon --spin-time 5 \
      >/dev/null 2>&1; then
    odom_ok=true
    break
  fi
  sleep 1
done
if [[ "$odom_ok" != true ]]; then
  echo "[RECOVER][FATAL] no dock odometry after 30s." >&2
  echo "Check the dock processes: ssh unitree@192.168.123.18 'pgrep -af unitree_slam'" >&2
  exit 2
fi
echo "[RECOVER] PASS: fresh dock odometry received."

echo "[RECOVER] Step 5/5: waiting for /rl2_nav/slam_health to go green"
echo "(slam_pose_ok=true and >=1000 known map cells; up to 90s)..."
health_ok=false
for attempt in $(seq 1 18); do
  health_line="$(timeout 6s ros2 topic echo /rl2_nav/slam_health \
    std_msgs/msg/String --field data --once --no-daemon --spin-time 5 \
    2>/dev/null || true)"
  if [[ -n "$health_line" ]] && printf '%s' "$health_line" | python3 -c '
import json, sys
try:
    text = "\n".join(line for line in sys.stdin.read().splitlines()
                     if line.strip() and line.strip() != "---")
    payload = json.loads(text)
except (ValueError, json.JSONDecodeError):
    sys.exit(1)
sys.exit(0 if bool(payload.get("slam_pose_ok")) and
         int(payload.get("map_known_cells") or 0) >= 1000 else 1)
'; then
    health_ok=true
    echo "[RECOVER] PASS: $health_line"
    break
  fi
  sleep 4
done
if [[ "$health_ok" != true ]]; then
  echo "[RECOVER][FATAL] slam health did not recover within 90s." >&2
  echo "Inspect: ros2 topic echo /rl2_nav/slam_health --field data --no-daemon" >&2
  echo "The lidar may need more time to accumulate the map; re-run this check." >&2
  exit 2
fi

guard_line="$(timeout 6s ros2 topic echo /rl2_nav/slam_pose_jump_guard/status \
  std_msgs/msg/String --field data --once --no-daemon --spin-time 5 \
  2>/dev/null || true)"
guard_state="$(printf '%s' "$guard_line" | python3 -c '
import json, sys
try:
    text = "\n".join(line for line in sys.stdin.read().splitlines()
                     if line.strip() and line.strip() != "---")
    print(json.loads(text).get("state", ""))
except (ValueError, json.JSONDecodeError):
    print("")
' 2>/dev/null || true)"

if [[ "$auto_acknowledge" == true ]]; then
  if [[ "$guard_state" == "READY_FOR_MANUAL_REARM" ]]; then
    echo "[RECOVER] Acknowledging the pose-jump guard (--auto-acknowledge)..."
    timeout 5s ros2 service call /slam_pose_jump_guard/acknowledge \
      std_srvs/srv/Trigger '{}' || \
      echo "[RECOVER][WARN] acknowledge call failed; do it manually."
  elif [[ -n "$guard_state" ]]; then
    echo "[RECOVER][INFO] guard state is ${guard_state}, not READY_FOR_MANUAL_REARM."
    echo "Wait for the new map to stabilize, then call acknowledge manually."
  fi
fi

echo ""
echo "[RECOVER] SLAM stream recovered. Guard state: ${guard_state:-unknown}"
echo ""
echo "Manual re-arm checklist (this script never does these):"
echo "  1. Verify the new map in RViz around the robot's actual position."
echo "  2. ros2 service call /slam_pose_jump_guard/acknowledge std_srvs/srv/Trigger '{}'"
echo "     (skip if --auto-acknowledge already did it and state is MONITORING)"
echo "  3. Only when ready to move again, enable both gates explicitly:"
echo "     ros2 service call /go2w_sport_cmd_bridge/enable std_srvs/srv/SetBool '{data: true}'"
echo "     ros2 service call /go2w_sport_cmd_bridge/balance_stand std_srvs/srv/Trigger '{}'"
echo "     ros2 service call /cmd_vel_safety_node/enable std_srvs/srv/SetBool '{data: true}'"
echo "  4. Restart the crouch mission per 蹲起流程复现.md section 4."
echo ""
echo "[RECOVER][OK] done."
