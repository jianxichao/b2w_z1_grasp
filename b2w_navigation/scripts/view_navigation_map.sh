#!/usr/bin/env bash
set -e
B2W_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$B2W_ROOT/setup.bash"
# Visualization only: does not launch controllers, publish goals or enable motion.
exec rviz2 -d "$B2W_ROOT/src/b2w_nav/rviz/map_diagnosis.rviz" --ros-args -r __node:=b2w_map_diagnosis_rviz
