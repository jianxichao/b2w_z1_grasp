#!/usr/bin/env bash
set -e
B2W_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cmake -S "$B2W_ROOT/native" -B "$B2W_ROOT/build/native"
cmake --build "$B2W_ROOT/build/native" --target capture_builtin -j2
