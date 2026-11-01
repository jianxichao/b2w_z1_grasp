#!/usr/bin/env bash
set -euo pipefail

restart=false
with_camera=false
dock_host="unitree@192.168.123.18"

usage() {
  cat <<'EOF'
Usage: ./scripts/ensure_dock_slam_stack.sh [--restart] [--with-camera] [user@host]

Ensures the official Unitree SLAM and Mid360 processes are running on the dock.
With --with-camera it also ensures the D435i ROS 2 driver is running.
--restart restarts only the selected processes. It never commands robot motion.
EOF
}

while (($#)); do
  case "$1" in
    --restart)
      restart=true
      ;;
    --with-camera)
      with_camera=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      dock_host="$1"
      ;;
  esac
  shift
done

echo "[1/3] Checking dock and Mid360 network..."
ping -c 1 -W 1 192.168.123.18 >/dev/null
ping -c 1 -W 1 192.168.123.20 >/dev/null

echo "[2/3] Ensuring dock sensor processes are running..."
ssh "$dock_host" "bash -s -- '$restart' '$with_camera'" <<'REMOTE'
set -euo pipefail
restart="$1"
with_camera="$2"
bin=/unitree/module/unitree_slam/bin
libs=/home/unitree/cyclonedds_ws/install/cyclonedds/lib:/usr/local/lib
camera_start=/home/unitree/realsense2_ros2_ws/start_d435i_ros2_on_dock.sh

if [[ "$restart" == true ]]; then
  echo "Restarting selected dock processes..."
  pkill -x unitree_slam 2>/dev/null || true
  pkill -x mid360_driver 2>/dev/null || true
  if [[ "$with_camera" == true ]]; then
    pkill -f realsense2_camera_node 2>/dev/null || true
    pkill -f start_d435i_ros2_on_dock.sh 2>/dev/null || true
  fi
  sleep 2
fi

if ! pgrep -x unitree_slam >/dev/null; then
  cd "$bin"
  LD_LIBRARY_PATH="$libs" nohup ./unitree_slam >/tmp/unitree_slam.log 2>&1 &
  sleep 4
fi

if ! pgrep -x mid360_driver >/dev/null; then
  cd "$bin"
  LD_LIBRARY_PATH="$libs" nohup ./mid360_driver >/tmp/mid360_driver.log 2>&1 &
  sleep 5
fi

if [[ "$with_camera" == true ]] && ! pgrep -f realsense2_camera_node >/dev/null; then
  if [[ ! -x "$camera_start" ]]; then
    echo "ERROR: D435i start script is missing or not executable: $camera_start" >&2
    exit 1
  fi
  nohup "$camera_start" >/tmp/d435i_ros2.log 2>&1 &
  sleep 10
fi

echo "--- dock processes ---"
pgrep -af 'unitree_slam|mid360_driver|realsense2_camera_node|start_d435i_ros2_on_dock' || true
test -s /home/unitree/rl2_site_map_v1.pcd && \
  ls -lh /home/unitree/rl2_site_map_v1.pcd || true

test "$(pgrep -xc unitree_slam)" -eq 1
test "$(pgrep -xc mid360_driver)" -eq 1
if [[ "$with_camera" == true ]]; then
  pgrep -f realsense2_camera_node >/dev/null
fi
REMOTE

echo "[3/3] Dock sensor stack is ready. This script did not enable robot motion."
echo "On the user PC, verify DDS endpoints after sourcing ROS 2:"
echo "  ros2 topic info /api/slam_operate/request --no-daemon --spin-time 3"
