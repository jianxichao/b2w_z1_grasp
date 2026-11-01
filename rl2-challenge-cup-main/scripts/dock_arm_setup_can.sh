#!/usr/bin/env bash
set -euo pipefail

DEVICE="${ARX_CAN_DEVICE:-/dev/ttyACM0}"
INTERFACE="${ARX_CAN_INTERFACE:-can1}"

[[ -c "$DEVICE" ]] || { echo "MISSING: $DEVICE" >&2; exit 2; }
udevadm info -q property -n "$DEVICE" | grep -q '^ID_VENDOR_ID=16d0$'
udevadm info -q property -n "$DEVICE" | grep -q '^ID_MODEL_ID=117e$'

sudo pkill -x slcand 2>/dev/null || true
sleep 0.2
if ip link show "$INTERFACE" >/dev/null 2>&1; then
  sudo ip link set "$INTERFACE" down || true
  sudo ip link delete "$INTERFACE" || true
fi
sudo slcand -o -f -s8 "$DEVICE" "$INTERFACE"
for _attempt in 1 2 3 4 5 6 7 8 9 10; do
  ip link show "$INTERFACE" >/dev/null 2>&1 && break
  sleep 0.1
done
ip link show "$INTERFACE" >/dev/null 2>&1 || {
  echo "slcand did not create $INTERFACE" >&2
  exit 3
}
sudo ip link set "$INTERFACE" up

echo "CAN interface ready (this does not command arm motion):"
ip -details -statistics link show "$INTERFACE"
