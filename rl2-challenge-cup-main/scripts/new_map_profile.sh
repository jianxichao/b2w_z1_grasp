#!/usr/bin/env bash
set -eo pipefail

name="${1:-}"
if [[ ! "$name" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Usage: $0 PROFILE_NAME (letters, digits, _ and - only)" >&2
  exit 2
fi

root="${HOME}/learning"
profile="$root/maps/$name"
template="$root/rl2-challenge-cup-week123_work_tree/src/rl2_nav/config/base_cave_waypoints.template.yaml"
if [[ -e "$profile" ]]; then
  echo "Refusing to overwrite existing profile: $profile" >&2
  exit 1
fi
mkdir -p "$profile"
cp "$template" "$profile/base_cave_waypoints.yaml"
echo "Created: $profile"
echo "Save the matching dock PCD as /home/unitree/${name}.pcd"
echo "Capture coordinates, then use waypoint_output_path=$profile/base_cave_waypoints.yaml"

