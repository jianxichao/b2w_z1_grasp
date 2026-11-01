#!/usr/bin/env bash
set -e
B2W_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
B2W_PYTHON="$B2W_ROOT/../unitree_z1/yolo/.venv/bin/python"
if [[ ! -x "$B2W_PYTHON" ]]; then
    echo "YOLO Python environment not found: $B2W_PYTHON" >&2
    exit 1
fi
if [[ ! -x "$B2W_ROOT/build/native/capture_builtin" ]]; then
    "$B2W_ROOT/scripts/build_native.sh"
fi
exec "$B2W_PYTHON" "$B2W_ROOT/scripts/detect_builtin.py" "$@"
