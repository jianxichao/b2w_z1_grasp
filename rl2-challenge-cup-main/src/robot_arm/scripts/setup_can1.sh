#!/usr/bin/env bash
# 启动 ARX X5 使用的 SLCAN 接口 can1（每次重新插拔 USB-CAN 后执行一次）。
set -euo pipefail

DEVICE="${ARX_CAN_DEVICE:-/dev/arxcan1}"
IFACE="${ARX_CAN_IFACE:-can1}"

if ! command -v slcand >/dev/null 2>&1; then
  echo "错误：找不到 slcand。请先安装 can-utils：sudo apt-get install can-utils" >&2
  exit 1
fi

if [[ ! -e "${DEVICE}" ]]; then
  echo "错误：${DEVICE} 不存在。请检查 USB-CAN 是否插入、udev 别名是否生效。" >&2
  echo "提示：ls -l /dev/ttyACM* ; readlink -f /dev/arxcan1" >&2
  exit 1
fi

echo "使用 ${DEVICE} -> ${IFACE}"
sudo ip link set "${IFACE}" down 2>/dev/null || true
sudo ip link delete "${IFACE}" 2>/dev/null || true
sudo slcand -o -f -s8 "${DEVICE}" "${IFACE}"
sleep 0.5
sudo ip link set "${IFACE}" up
ip -details -statistics link show "${IFACE}"
pgrep -af '^slcand ' || true
