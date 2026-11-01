#!/usr/bin/env bash
# Reversible eth1 test for a binary that hard-codes multicast-iface=eth0.
set -euo pipefail

dock_host="${1:-unitree@192.168.123.18}"

# -tt is required because the dock's sudo policy reads its password only from
# an interactive terminal.
ssh -tt "$dock_host" 'bash -s' <<'REMOTE'
set -euo pipefail
source_dir=/home/unitree/aruco_recharge_20250801
test_dir=/home/unitree/aruco_recharge_20250801_eth1_test
parent_iface=eth10
test_iface=eth1
test_ip=192.168.123.223

test -x "$source_dir/aruco_recharge"
if [ -e "$test_dir" ]; then
  echo "ERROR: $test_dir already exists; nothing changed." >&2
  exit 3
fi
if ip link show "$test_iface" >/dev/null 2>&1; then
  echo "ERROR: interface $test_iface already exists; nothing changed." >&2
  exit 4
fi
if ping -c 1 -W 1 "$test_ip" >/dev/null 2>&1; then
  echo "ERROR: proposed temporary address $test_ip replies to ping; choose another address." >&2
  exit 5
fi

echo '=== Creating private eth1 program copy ==='
cp -a "$source_dir" "$test_dir"

python3 - "$test_dir/aruco_recharge" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
data = path.read_bytes()
old = b"multicast-iface=eth0"
new = b"multicast-iface=eth1"
if data.count(old) != 1:
    raise SystemExit(f"expected exactly one video-interface string, found {data.count(old)}")
path.write_bytes(data.replace(old, new, 1))
print("patched only the video pipeline interface: eth0 -> eth1")
PY

sed -i '/^net_addr:/{n;s/^[[:space:]]*.*/  eth1/;}' "$test_dir/aruco_config.yaml"
grep -A1 '^net_addr:' "$test_dir/aruco_config.yaml"
grep -abo 'multicast-iface=eth1' "$test_dir/aruco_recharge"

echo '=== Creating temporary eth1 macvlan on eth10 ==='
echo "This needs sudo only for the new temporary interface $test_iface."
sudo ip link add link "$parent_iface" name "$test_iface" type macvlan mode bridge
sudo ip addr add "$test_ip/24" dev "$test_iface"
sudo ip link set dev "$test_iface" allmulticast on
sudo ip link set dev "$test_iface" up
ip -br addr show dev "$parent_iface"
ip -br addr show dev "$test_iface"

cat <<EOF
READY: original files are unchanged at $source_dir
TEST COPY: $test_dir
TEMPORARY INTERFACE: $test_iface ($test_ip)
To test only service startup (do not publish aruco_start):
  cd $test_dir
  LD_LIBRARY_PATH="\$PWD/lib" ./aruco_recharge
To remove only the temporary network interface later:
  sudo ip link delete $test_iface
EOF
REMOTE
