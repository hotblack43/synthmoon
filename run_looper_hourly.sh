#!/usr/bin/env bash
set -euo pipefail

CONFIG="scene.toml"
OUTDIR="OUTPUT/hourly_layer5"
N_STEPS=48
STEP_HOURS=1
START_UTC="2026-01-24T00:00:00Z"   # change this

mkdir -p "$OUTDIR"

# Backup + restore scene.toml automatically
CONFIG_BAK="${CONFIG}.bak_run_$$"
cp -a "$CONFIG" "$CONFIG_BAK"
trap 'mv -f "$CONFIG_BAK" "$CONFIG"' EXIT

# Set a TOML key like: utc = "...."  or out_fits = "...."
set_toml_string () {
  local key="$1"
  local value="$2"   # without quotes
  sed -i -E "s|^(${key}[[:space:]]*=[[:space:]]*)\"[^\"]*\"|\\1\"${value}\"|" "$CONFIG"
}

for ((i=0; i<N_STEPS; i++)); do
  step_utc="$(date -u -d "$START_UTC + $((i*STEP_HOURS)) hours" +"%Y-%m-%dT%H:%M:%SZ")"
  idx="$(printf "%03d" "$i")"
  safe_ts="${step_utc//:/}"  # remove ":" for filenames
  out_fits="${OUTDIR}/synth_layer5_${idx}_${safe_ts}.fits"

  set_toml_string "utc" "$step_utc"
  set_toml_string "out_fits" "$out_fits"

  echo "[$idx] utc=$step_utc -> $out_fits"
  PYTHONPATH=$PWD uv run -m synthmoon.run_v0 --config "$CONFIG" 
done

echo "Done. FITS written to: $OUTDIR"

