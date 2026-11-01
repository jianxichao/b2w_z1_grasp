#!/usr/bin/env bash
set -e
B2W_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$B2W_ROOT/setup.bash"
exec python "$B2W_ROOT/scripts/search_with_wrist.py" "$@"
