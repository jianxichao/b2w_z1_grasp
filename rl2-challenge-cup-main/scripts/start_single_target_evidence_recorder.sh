#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:-}"

if [[ -z "$RUN_ID" ]]; then
  echo "Usage: $0 <run-id>"
  echo "Example: $0 single_target_001"
  exit 2
fi

set +u
source /home/kazuhalry/learning/unitree_ros2/setup_go2w_humble.sh
source "$ROOT_DIR/install/setup.bash"
set -u

export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/rl2_single_target_evidence}"
mkdir -p "$ROS_LOG_DIR"

exec python3 "$ROOT_DIR/scripts/record_single_target_evidence.py" \
  --run-id "$RUN_ID" \
  --output-root "$ROOT_DIR/evidence/single_target_runs"
