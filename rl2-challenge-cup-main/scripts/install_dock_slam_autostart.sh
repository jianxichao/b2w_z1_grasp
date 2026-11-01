#!/usr/bin/env bash
set -eo pipefail

dock_host="${1:-unitree@192.168.123.18}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

scp "$root/deploy/dock/systemd/unitree-slam.service" \
    "$root/deploy/dock/systemd/unitree-mid360-driver.service" \
    "$dock_host:/tmp/"

ssh -t "$dock_host" \
  'sudo install -m 0644 /tmp/unitree-slam.service /etc/systemd/system/unitree-slam.service &&
   sudo install -m 0644 /tmp/unitree-mid360-driver.service /etc/systemd/system/unitree-mid360-driver.service &&
   sudo systemctl daemon-reload &&
   sudo systemctl enable unitree-slam.service unitree-mid360-driver.service &&
   sudo systemctl restart unitree-slam.service &&
   sleep 5 &&
   sudo systemctl restart unitree-mid360-driver.service &&
   sudo systemctl --no-pager --full status unitree-slam.service unitree-mid360-driver.service'

