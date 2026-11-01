#!/usr/bin/env bash
set -euo pipefail

# Deploy and build a tiny SDK2-native /aruco_cmd publisher on the dock.
# This script never runs the publisher and therefore cannot move the robot.

workspace_dir="/home/kazuhalry/learning/rl2-challenge-cup-week123_work_tree"
sdk_dir="/home/kazuhalry/learning/unitree_sdk2"
source_file="$workspace_dir/tools/aruco_native_sender/aruco_native_sender.cpp"
runtime_tools_dir="$workspace_dir/tools/aruco_native_sender"
dock_host="unitree@192.168.123.18"
remote_dir="/home/unitree/aruco_native_sender"
bundle_file="/tmp/aruco_native_sender_sdk2_bundle.tar.gz"

test -f "$source_file"
test -f "$sdk_dir/lib/aarch64/libunitree_sdk2.a"
test -f "$sdk_dir/thirdparty/lib/aarch64/libddsc.so"
test -f "$sdk_dir/thirdparty/lib/aarch64/libddscxx.so"

tar -C "$sdk_dir" -czf "$bundle_file" \
  include \
  thirdparty/include \
  lib/aarch64/libunitree_sdk2.a \
  thirdparty/lib/aarch64/libddsc.so \
  thirdparty/lib/aarch64/libddscxx.so

echo "Uploading SDK2 headers/libraries and native sender source..."
ssh "$dock_host" "mkdir -p '$remote_dir'"
scp "$bundle_file" "$source_file" \
  "$runtime_tools_dir/start_aruco_recharge_runtime.sh" \
  "$runtime_tools_dir/send_aruco_start_native.sh" \
  "$runtime_tools_dir/stop_aruco_recharge_runtime.sh" \
  "$dock_host:$remote_dir/"

echo "Building on the dock (no DDS process will be started)..."
ssh "$dock_host" "
set -eu
cd '$remote_dir'
tar -xzf aruco_native_sender_sdk2_bundle.tar.gz
mkdir -p lib
cp thirdparty/lib/aarch64/libddsc.so lib/
cp thirdparty/lib/aarch64/libddscxx.so lib/
ln -sfn libddsc.so lib/libddsc.so.0
ln -sfn libddscxx.so lib/libddscxx.so.0
g++ -std=c++17 -O2 aruco_native_sender.cpp \
  -Iinclude \
  -Ithirdparty/include \
  -Ithirdparty/include/ddscxx \
  lib/aarch64/libunitree_sdk2.a \
  -Llib -Wl,-rpath,'\$ORIGIN/lib' \
  -lddscxx -lddsc -pthread \
  -o aruco_native_sender
chmod 755 aruco_native_sender \
  start_aruco_recharge_runtime.sh \
  send_aruco_start_native.sh \
  stop_aruco_recharge_runtime.sh
file aruco_native_sender
LD_LIBRARY_PATH=\"\$PWD/lib\" ldd aruco_native_sender | grep -E 'ddsc|ddscxx|not found'
"

echo
echo "DEPLOYED: $remote_dir/aruco_native_sender"
echo "RUNTIME TOOLS: $remote_dir/{start,send,stop}_aruco_*"
echo "The sender was built but NOT executed. No DDS command was published."
