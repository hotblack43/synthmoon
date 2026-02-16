#!/usr/bin/env bash
set -euo pipefail

UTC="${1:-2026-01-24T00:00:00Z}"
TMPDIR="$(mktemp -d /tmp/synthmoon_brdf_XXXXXX)"
HAPKE_CFG="$(mktemp ./scene_hapke_XXXXXX.toml)"
trap 'rm -rf "$TMPDIR"; rm -f "$HAPKE_CFG"' EXIT

LAMBERT_OUT="$TMPDIR/lambert.fits"
HAPKE_OUT="$TMPDIR/hapke.fits"

uv run -m synthmoon.run_v0 --config scene.toml --utc "$UTC" --out "$LAMBERT_OUT"

python - <<'PY' scene.toml "$HAPKE_CFG"
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
out = []
in_moon = False
set_brdf = False
for line in src:
    s = line.strip()
    if s.startswith("[") and s.endswith("]"):
        in_moon = (s == "[moon]")
    if in_moon and s.startswith("brdf"):
        out.append('brdf = "hapke"')
        set_brdf = True
    else:
        out.append(line)
if not set_brdf:
    raise SystemExit("Could not find [moon] brdf line in scene.toml")
Path(sys.argv[2]).write_text("\n".join(out) + "\n", encoding="utf-8")
PY

uv run -m synthmoon.run_v0 --config "$HAPKE_CFG" --utc "$UTC" --out "$HAPKE_OUT"

python - <<'PY' "$LAMBERT_OUT" "$HAPKE_OUT"
import sys
import numpy as np
from astropy.io import fits

a_path, b_path = sys.argv[1], sys.argv[2]
with fits.open(a_path, memmap=False) as ha, fits.open(b_path, memmap=False) as hb:
    a = np.asarray(ha[0].data)
    b = np.asarray(hb[0].data)
    h1 = ha[0].header
    h2 = hb[0].header

if a.shape != b.shape:
    raise SystemExit(f"FAIL: shape mismatch {a.shape} vs {b.shape}")

n = int(h1.get("NLAYERS", a.shape[0] if a.ndim == 3 else 1))
for i in range(1, n + 1):
    k = f"LAY{i}"
    if str(h1.get(k, "")) != str(h2.get(k, "")):
        raise SystemExit(f"FAIL: layer naming mismatch at {k}: {h1.get(k)} vs {h2.get(k)}")

if np.array_equal(a, b):
    raise SystemExit("FAIL: Lambert and Hapke outputs are identical")

print("PASS: Lambert vs Hapke differ; layer naming and shape are unchanged.")
PY
