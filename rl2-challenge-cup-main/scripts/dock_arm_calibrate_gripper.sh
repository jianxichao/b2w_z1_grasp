#!/usr/bin/env bash
# Interactive ARX5 gripper calibration for the dock.  It never moves joints
# and deliberately disables the controller's background sender while the
# operator sets the two mechanical endpoints by hand.
set -euo pipefail

ROOT="${DOCK_ARM_ROOT:-/home/unitree/rl2_arm_runtime}"
CAN_INTERFACE="${ARX_CAN_INTERFACE:-can1}"
CONFIG="$ROOT/src/robot_arm/config/hardware.yaml"

cd "$ROOT"
if pgrep -af 'arm_runtime_node|vision_target_node' >/dev/null; then
  echo "ABORT: stop dock_arm_initialize/dock_arm_start first; calibration needs exclusive CAN access." >&2
  exit 3
fi
ip link show "$CAN_INTERFACE" >/dev/null 2>&1 || {
  echo "ABORT: $CAN_INTERFACE is absent; run ./scripts/dock_arm_setup_can.sh first." >&2
  exit 3
}

echo "The arm must be stationary and unobstructed. This script sends zero torque to the gripper only."
echo "At prompt 1: physically close the jaws completely, then press Enter."
echo "At prompt 2: physically open the jaws completely, then press Enter."

CALIBRATION_PY="$(mktemp)"
trap 'rm -f "$CALIBRATION_PY"' EXIT
cat >"$CALIBRATION_PY" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
interface = sys.argv[2]
import arx5_interface as arx5

robot = arx5.RobotConfigFactory.get_instance().get_config("X5")
robot.urdf_path = str(root / "src/robot_arm/third_party/arx5_sdk/models/X5.urdf")
# Some official ARM64 SDK builds run the normal gripper-range sanity check
# during construction even when background feedback is disabled.  A negative
# bootstrap value is the SDK-documented way to enter its calibration routine
# from an invalid zero; it is intentionally not written to hardware.yaml.
robot.gripper_open_readout = -3.4
cfg = arx5.ControllerConfigFactory.get_instance().get_config("joint_controller", 6)
cfg.gravity_compensation = False
cfg.background_send_recv = False
controller = arx5.Arx5JointController(robot, cfg, interface)
controller.calibrate_gripper()
PY
set +e
python3 "$CALIBRATION_PY" "$ROOT" "$CAN_INTERFACE"
calibration_status=$?
set -e
if [[ "$calibration_status" -ne 0 ]]; then
  echo "NOTE: the dock SDK reported an exit warning after printing the readout."
  echo "      This is expected with the temporary -3.4 bootstrap; save the displayed readout below."
fi

echo
read -r -p "Paste the printed 'Fully-open joint position readout' value exactly: " value
python3 - "$CONFIG" "$value" <<'PY'
import math
import re
import sys

path, raw = sys.argv[1:]
value = float(raw)
if not math.isfinite(value) or abs(value) < 1e-6:
    raise SystemExit("ABORT: value must be finite and non-zero")
text = open(path, encoding="utf-8").read()
updated, count = re.subn(
    r"^gripper_open_readout:\s*[-+0-9.eE]+\s*$",
    f"gripper_open_readout: {value:.8f}",
    text,
    count=1,
    flags=re.M,
)
if count != 1:
    raise SystemExit("ABORT: gripper_open_readout was not found exactly once")
open(path, "w", encoding="utf-8").write(updated)
print(f"Saved gripper_open_readout={value:.8f} to {path}")
PY

echo "Calibration saved. Start the normal runtime with: ./scripts/dock_arm_initialize.sh real vision"
