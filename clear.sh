#!/usr/bin/env bash
set -euo pipefail

# Clear build directory (default: ./build)
# Usage: ./clear.sh [build_dir]

BUILD_DIR=${1:-build}
if [[ -d "$BUILD_DIR" ]]; then
  rm -rf -- "$BUILD_DIR"
  echo "Removed: $BUILD_DIR"
else
  echo "No build dir to remove: $BUILD_DIR"
fi
