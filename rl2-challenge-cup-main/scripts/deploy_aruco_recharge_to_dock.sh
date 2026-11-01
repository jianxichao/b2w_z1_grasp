#!/usr/bin/env bash
# Deploy Unitree's supplied Go2-W ArUco recharge package without overwriting
# the dock SDK, unitree_slam, or an older recharge deployment.
# This script only copies files and validates library resolution.  It never
# starts aruco_recharge and cannot move the robot.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${ARUCO_SOURCE_DIR:-$HOME/learning/aruco_recharge}"
dock_host="${1:-unitree@192.168.123.18}"
release_name="aruco_recharge_20250801"
remote_dir="/home/unitree/${release_name}"

for item in aruco_recharge aruco_config.yaml aruco_id0.png; do
  if [[ ! -f "$source_dir/$item" ]]; then
    echo "ERROR: missing $source_dir/$item" >&2
    exit 1
  fi
done

if ! file "$source_dir/aruco_recharge" | grep -q 'ARM aarch64'; then
  echo 'ERROR: supplied aruco_recharge is not an ARM64 executable.' >&2
  exit 1
fi

echo '=== Source package ==='
printf 'source: %s\n' "$source_dir"
sha256sum "$source_dir/aruco_recharge" "$source_dir/aruco_config.yaml" "$source_dir/aruco_id0.png"

echo '=== Uploading into a new versioned dock directory ==='
remote_stage="/home/unitree/.${release_name}.staging.$(date +%s).$$"
ssh "$dock_host" "mkdir -p '$remote_stage'"
scp "$source_dir/aruco_recharge" "$source_dir/aruco_config.yaml" "$source_dir/aruco_id0.png" \
  "${dock_host}:${remote_stage}/"

echo '=== Verifying on dock; no program will be started ==='
ssh "$dock_host" "bash -s" <<REMOTE
set -euo pipefail
stage='$remote_stage'
target='$remote_dir'
echo '--- uploaded files ---'
sha256sum "\$stage/aruco_recharge" "\$stage/aruco_config.yaml" "\$stage/aruco_id0.png"
echo '--- required runtime dependencies ---'
ldd "\$stage/aruco_recharge" | grep -E 'libddsc|libddscxx|libyaml-cpp|opencv|not found' || true
if ldd "\$stage/aruco_recharge" | grep -q 'not found'; then
  echo 'ERROR: one or more dependencies are missing; retaining staging directory for diagnosis.' >&2
  exit 2
fi
if [ -e "\$target" ]; then
  echo "ERROR: target already exists: \$target (nothing overwritten)." >&2
  exit 3
fi
mv "\$stage" "\$target"
chmod 755 "\$target/aruco_recharge"
echo "DEPLOYED: \$target"
REMOTE

cat <<EOF

Deployment succeeded.  It did not start the recharge service.

When the robot, pile and surroundings satisfy the safety checklist, SSH to the
dock, choose no ROS environment, then run:

  cd $remote_dir
  ./aruco_recharge

EOF
