#!/usr/bin/env bash
set -euo pipefail
systemctl --user stop rl2-storage-box-bridge.service
systemctl --user --no-pager --full status rl2-storage-box-bridge.service 2>&1 | sed -n '1,8p' || true
