#!/usr/bin/env bash
# Observation-only preflight for the one-target autonomous chain.
set -uo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source /home/kazuhalry/learning/unitree_ros2/setup_go2w_humble.sh >/dev/null
source "$project_root/install/setup.bash" >/dev/null
set -u

fail=0
warn=0
pass() { printf 'PASS  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; warn=$((warn + 1)); }
fail() { printf 'FAIL  %s\n' "$*"; fail=$((fail + 1)); }

services="$(timeout 8s ros2 service list --no-daemon 2>/dev/null || true)"
actions="$(timeout 8s ros2 action list -t 2>/dev/null || true)"
topics="$(timeout 8s ros2 topic list -t --no-daemon 2>/dev/null || true)"

check_service() {
  grep -Fxq "$1" <<<"$services" && pass "service $1" || fail "missing service $1"
}
check_action() {
  grep -Fq "$1 " <<<"$actions" && pass "action $1" || fail "missing action $1"
}
check_topic() {
  grep -Fq "$1 " <<<"$topics" && pass "topic $1" || fail "missing topic $1"
}

printf '=== One-target closed-loop preflight (NO MOTION) ===\n'
check_service /mission_supervisor_node/start
check_service /mission_supervisor_node/abort
check_service /cmd_vel_safety_node/enable
check_service /go2w_sport_cmd_bridge/enable
check_service /storage_box/open
check_service /storage_box/close
check_action /navigate_to_pose
check_action /arm/grasp_target
check_topic /slam/current_pose
check_topic /navigation_map
check_topic /target_detection_status
check_topic /robot_arm/target_pose
check_topic /arm/status
check_topic /storage_box/status
check_topic /task_event

if grep -Fq '/storage_box/sample_present ' <<<"$topics"; then
  pass 'independent in-bin sensor /storage_box/sample_present'
else
  warn 'no independent in-bin sensor; Action/controller success cannot prove sample is in the box'
fi

for item in \
  'target_goal_count:1' \
  'target_posture_enabled:False' \
  'arm_integration_enabled:True'; do
  name="${item%%:*}"
  expected="${item#*:}"
  actual="$(timeout 4s ros2 param get /mission_supervisor_node "$name" 2>/dev/null | tail -1)"
  if grep -Fq "$expected" <<<"$actual"; then
    pass "parameter $name = $expected"
  else
    fail "parameter $name expected $expected, got ${actual:-<unavailable>}"
  fi
done

printf '=== Result: %d FAIL, %d WARN ===\n' "$fail" "$warn"
((fail == 0))
