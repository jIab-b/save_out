#!/usr/bin/env bash
set -euo pipefail

# Simple AIO build script
# Usage: ./build.sh [build_dir]

BUILD_DIR=${1:-build}
JOBS=${JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)}

cmake -S . -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" -j "$JOBS"

echo "Built into: $BUILD_DIR"
