#!/usr/bin/env bash
# Daily dock-side ARX5 + Gemini startup.  It never starts a grasp Action.
set -euo pipefail

ROOT="${DOCK_ARM_ROOT:-/home/unitree/rl2_arm_runtime}"
MODE="${1:-real}"
VISION="${2:-vision}"

case "$MODE" in
  real|dry-run) ;;
  *) echo "usage: $0 [real|dry-run] [vision|no-vision]" >&2; exit 2 ;;
esac
case "$VISION" in
  vision|no-vision) ;;
  *) echo "usage: $0 [real|dry-run] [vision|no-vision]" >&2; exit 2 ;;
esac

cd "$ROOT"

if pgrep -af 'arm_runtime_node|vision_target_node' >/dev/null; then
  echo "ABORT: an arm runtime process is already active; do not start a second CAN controller." >&2
  pgrep -af 'arm_runtime_node|vision_target_node' >&2 || true
  exit 3
fi

CONFIG="$ROOT/src/robot_arm/config/hardware.yaml"
INSTALLED_CONFIG="$ROOT/install/robot_arm/share/robot_arm/config/hardware.yaml"
python3 - "$CONFIG" <<'PY'
import math
import re
import sys

path = sys.argv[1]
try:
    text = open(path, encoding="utf-8").read()
except OSError as exc:
    raise SystemExit(f"ABORT: cannot read {path}: {exc}")
match = re.search(r"^gripper_open_readout:\s*([-+0-9.eE]+)\s*$", text, re.M)
if not match:
    raise SystemExit("ABORT: gripper_open_readout is missing from hardware.yaml")
value = float(match.group(1))
if not math.isfinite(value) or abs(value) < 1e-6:
    raise SystemExit("ABORT: gripper_open_readout must be a non-zero finite calibration value")
print(f"Using saved gripper_open_readout={value:.5f}")
PY

# The runtime imports robot_arm from the colcon install space.  Keep its
# package-data copy aligned with the source calibration; otherwise a freshly
# calibrated source file is silently ignored until a full rebuild.
if [[ ! -f "$INSTALLED_CONFIG" ]]; then
  echo "ABORT: installed hardware configuration is missing: $INSTALLED_CONFIG" >&2
  exit 1
fi
backup="$INSTALLED_CONFIG.pre_calibration_sync"
if [[ ! -e "$backup" ]]; then
  cp -p "$INSTALLED_CONFIG" "$backup"
fi
cp "$CONFIG" "$INSTALLED_CONFIG"
echo "Synchronized calibration into installed runtime config."

echo "=== 1/3 Establish SocketCAN (sudo may ask for the dock password) ==="
./scripts/dock_arm_setup_can.sh

echo "=== 2/3 Verify SDK, ROS packages, and CAN interface ==="
./scripts/dock_arm_preflight.sh

echo "=== 3/3 Start arm runtime in foreground ==="
echo "No grasp goal is sent by this script. Keep this terminal open while using the arm."
exec ./scripts/dock_arm_start.sh "$MODE" "$VISION"
