#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-scene.toml}"
UTC="${2:-2026-01-24T00:00:00Z}"

uv run python tools/check_layers_not_zero.py \
  --config "$CONFIG" \
  --utc "$UTC" \
  --require-earthlight-layers-nonzero
