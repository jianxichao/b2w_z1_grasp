#!/usr/bin/env bash
set -euo pipefail

ROOT="${DOCK_ARM_ROOT:-/home/unitree/rl2_arm_runtime}"
INTERFACE="${ARX_CAN_INTERFACE:-can1}"

echo "=== Dock architecture / ROS ==="
uname -m
python3 --version
test -f /opt/ros/foxy/setup.bash

echo "=== ARX USB-CAN ==="
ls -l /dev/ttyACM* 2>/dev/null || true
udevadm info -q property -n /dev/ttyACM0 2>/dev/null | \
  grep -E 'ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL_SHORT' || true
ip -details link show "$INTERFACE" 2>/dev/null || echo "NOT_READY: run scripts/dock_arm_setup_can.sh"

echo "=== Compiled runtime ==="
test -f "$ROOT/install/setup.bash"
set +u
source /opt/ros/foxy/setup.bash
source "$ROOT/install/setup.bash"
set -u
ros2 pkg prefix rl2_interfaces
ros2 pkg prefix robot_arm
ros2 pkg prefix rl2_manipulation
python3 - <<'PY'
import arx5_interface
from rl2_interfaces.action import GraspTarget
from robot_arm.hardware import Arx5Backend
from rl2_manipulation.arm_action_server import ArmRuntimeNode
print("ARX5 SDK:", arx5_interface.__file__)
print("Python/ROS imports: OK")
PY

echo "PASS: software is ready. This check did not connect to or move the arm."
