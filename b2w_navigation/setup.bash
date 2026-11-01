# source this file in each terminal. Does not start hardware.
B2W_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
if [[ -f "$B2W_WS/.runtime/env.bash" ]]; then
    source "$B2W_WS/.runtime/env.bash"
fi
if [[ -x "$B2W_WS/../unitree_z1/yolo/.venv/bin/python" ]]; then
    export PATH="$B2W_WS/../unitree_z1/yolo/.venv/bin:$PATH"
    export PYTHONPATH="$B2W_WS/../unitree_z1/yolo/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
fi
if [[ -f "$B2W_WS/install/setup.bash" ]]; then
    source "$B2W_WS/install/setup.bash"
fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$B2W_WS/config/cyclonedds.xml"
