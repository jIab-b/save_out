#!/usr/bin/env bash
set -euo pipefail

# Simple runner for diffuse binary
# Defaults allow running with no args: ./run.sh

BUILD_DIR=${BUILD_DIR:-build}
BIN="$BUILD_DIR/diffuse"

# Defaults (can be overridden via env)
DEFAULT_MODEL=${DEFAULT_MODEL:-diffusion/sdxl.profile.json}
DEFAULT_PROMPT=${DEFAULT_PROMPT:-"a cat sitting on a chair"}
DEFAULT_STEPS=${DEFAULT_STEPS:-20}
DEFAULT_OUT=${DEFAULT_OUT:-out.png}

if [[ ! -x "$BIN" ]]; then
  echo "error: $BIN not found. Build first with ./build.sh" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  set -- "$DEFAULT_MODEL" --prompt "$DEFAULT_PROMPT" --steps "$DEFAULT_STEPS" --out "$DEFAULT_OUT"
fi

exec "$BIN" "$@"
