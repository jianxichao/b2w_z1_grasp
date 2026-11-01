#!/usr/bin/env bash
# Add a private, paired CycloneDDS C/C++ runtime for the official ArUco
# executable. It never changes /usr/local/lib or starts the executable.
set -euo pipefail

sdk_lib="${ARUCO_SDK_LIB_DIR:-$HOME/learning/unitree_sdk2/thirdparty/lib/aarch64}"
dock_host="${1:-unitree@192.168.123.18}"
remote_dir="/home/unitree/aruco_recharge_20250801"
remote_stage="${remote_dir}/.private_dds_staging.$(date +%s).$$"

for library in libddsc.so libddscxx.so; do
  if [[ ! -f "$sdk_lib/$library" ]]; then
    echo "ERROR: missing SDK library: $sdk_lib/$library" >&2
    exit 1
  fi
  if ! file "$sdk_lib/$library" | grep -q 'ARM aarch64'; then
    echo "ERROR: $sdk_lib/$library is not ARM64." >&2
    exit 1
  fi
done

echo '=== Private SDK runtime source ==='
sha256sum "$sdk_lib/libddsc.so" "$sdk_lib/libddscxx.so"

ssh "$dock_host" "test -x '$remote_dir/aruco_recharge'; mkdir -p '$remote_stage'"
scp "$sdk_lib/libddsc.so" "$sdk_lib/libddscxx.so" "${dock_host}:${remote_stage}/"

ssh "$dock_host" "bash -s" <<REMOTE
set -euo pipefail
stage='$remote_stage'
target='$remote_dir/lib'
if [ -e "\$target" ]; then
  echo "ERROR: \$target already exists; no files were overwritten." >&2
  exit 3
fi
ln -s libddsc.so "\$stage/libddsc.so.0"
ln -s libddscxx.so "\$stage/libddscxx.so.0"
LD_LIBRARY_PATH="\$stage" ldd '$remote_dir/aruco_recharge' | grep -E 'libddsc|libddscxx|not found'
mv "\$stage" "\$target"
echo "PRIVATE_DDS_DEPLOYED: \$target"
REMOTE

cat <<EOF

No process was started and no system library was changed.

To test only service startup on the dock (no aruco_start command):
  cd $remote_dir
  LD_LIBRARY_PATH="\$PWD/lib" ./aruco_recharge
EOF
