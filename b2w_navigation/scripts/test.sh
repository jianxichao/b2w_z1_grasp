#!/usr/bin/env bash
set -e
B2W_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$B2W_ROOT/setup.bash"
# Isolate integration mocks from the real robot's DDS domain 0.
export ROS_DOMAIN_ID=139
export ROS_LOCALHOST_ONLY=1
export CYCLONEDDS_URI="file://$B2W_ROOT/config/cyclonedds_test.xml"
python -m pytest -q "$B2W_ROOT/src/b2w_hardware/test" \
    "$B2W_ROOT/src/b2w_perception/test" "$B2W_ROOT/src/b2w_nav/test"
