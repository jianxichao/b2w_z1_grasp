#!/usr/bin/env bash
# Remove only the temporary eth1 macvlan created for the ArUco test.
set -euo pipefail

dock_host="${1:-unitree@192.168.123.18}"
# Keep a TTY so sudo can prompt on the dock when needed.
ssh -tt "$dock_host" 'bash -s' <<'REMOTE'
set -euo pipefail
if ip link show eth1 >/dev/null 2>&1; then
  sudo ip link delete eth1
  echo 'Removed temporary eth1 interface.'
else
  echo 'eth1 is already absent; nothing to remove.'
fi
REMOTE
