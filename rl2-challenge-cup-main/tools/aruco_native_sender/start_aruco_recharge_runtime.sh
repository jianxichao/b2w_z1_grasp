#!/usr/bin/env bash
set -euo pipefail

program_dir="/home/unitree/aruco_recharge_20250801_eth1_test"
parent_iface="eth10"
runtime_iface="eth1"
runtime_ip="192.168.123.223"

test -x "$program_dir/aruco_recharge"
test -d "$program_dir/lib"
ip link show "$parent_iface" >/dev/null

if ! ip link show "$runtime_iface" >/dev/null 2>&1; then
  echo "Creating reversible $runtime_iface macvlan on $parent_iface (sudo required)..."
  sudo ip link add link "$parent_iface" name "$runtime_iface" type macvlan mode bridge
  sudo ip addr add "$runtime_ip/24" dev "$runtime_iface"
  sudo ip link set dev "$runtime_iface" allmulticast on
  sudo ip link set dev "$runtime_iface" up
fi

ip -br addr show dev "$parent_iface"
ip -br addr show dev "$runtime_iface"

if pgrep -x aruco_recharge >/dev/null; then
  echo "ERROR: aruco_recharge is already running:" >&2
  pgrep -af aruco_recharge >&2
  exit 3
fi

echo "Starting official ArUco recharge service in the foreground."
echo "This does not start robot motion until aruco_start is sent."
cd "$program_dir"
exec env LD_LIBRARY_PATH="$program_dir/lib" ./aruco_recharge
