#!/usr/bin/env bash
set -e
B2W_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$B2W_ROOT/setup.bash"
cd "$B2W_ROOT"
/usr/bin/python3 -m compileall -q src scripts
# System Python is used by ROS code generation; venv Python runs PT perception.
/usr/bin/python3 /usr/bin/colcon build --base-paths src --symlink-install \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DBUILD_TESTING=OFF
