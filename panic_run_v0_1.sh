#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./panic_run_v0_1.sh                # uses scene.toml + tag v0.1.0
#   ./panic_run_v0_1.sh scene.toml
#   ./panic_run_v0_1.sh scene.toml v0.1.0

CONFIG="${1:-scene.toml}"
TAG="${2:-v0.1.0}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "== SYNTHMOON PANIC RUNNER =="
echo "Repo:   $REPO_DIR"
echo "Config: $CONFIG"
echo "Tag:    $TAG"
echo

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: Not a git repo. cd into your synthmoon repo and try again."
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: Config not found: $CONFIG"
  exit 1
fi

# Stash local changes so checkout can't fail
if [[ -n "$(git status --porcelain)" ]]; then
  STASH_MSG="panic-stash $(date -u +%Y%m%dT%H%M%SZ)"
  echo "Local changes detected -> stashing: $STASH_MSG"
  git stash push -u -m "$STASH_MSG" >/dev/null
  echo
fi

echo "Fetching tags..."
git fetch --tags >/dev/null 2>&1 || true

if ! git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  echo "ERROR: Tag not found: $TAG"
  echo "Fix (once): git tag -a $TAG -m \"Known-good\" && git push --tags"
  exit 1
fi

echo "Checking out $TAG ..."
git checkout -q "$TAG"
echo "Now at: $(git describe --tags --always --dirty)"
echo

echo "uv sync..."
uv sync >/dev/null
echo "OK"
echo

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="OUTPUT/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

echo "Run folder: $RUN_DIR"
cp -av "$CONFIG" "$RUN_DIR/scene.toml" >/dev/null
git describe --tags --always --dirty > "$RUN_DIR/git_version.txt"

OUT_PATH="$(uv run python - <<'PY'
import sys, tomllib
cfg_path = sys.argv[1]
with open(cfg_path, "rb") as f:
    d = tomllib.load(f)
print(d.get("paths", {}).get("out_fits", "OUTPUT/synth_moon_v0.fits"))
PY
"$CONFIG")"

echo "Expected output: $OUT_PATH"
echo

LOG="$RUN_DIR/run.log"
echo "Running renderer (logging to $LOG) ..."
PYTHONPATH="$REPO_DIR" uv run python -m synthmoon.run_v0 --config "$CONFIG" 2>&1 | tee "$LOG"
echo

if [[ -f "$OUT_PATH" ]]; then
  case "$OUT_PATH" in
    "$RUN_DIR"/*) echo "Output already in run folder." ;;
    *) echo "Copying output into run folder..." ; cp -av "$OUT_PATH" "$RUN_DIR/" >/dev/null ;;
  esac
else
  echo "WARNING: Output not found at: $OUT_PATH"
  echo "Check $LOG for the output path printed by the program."
fi

echo
echo "DONE. Run saved in: $RUN_DIR"
echo "Open in DS9:"
echo "  ds9 \"$RUN_DIR/$(basename "$OUT_PATH")\" &"
