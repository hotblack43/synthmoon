#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/go_single_image.sh --lon DEG --lat DEG --alt-m M (--utc ISO8601Z | --jd JD) [options]

Required:
  --lon DEG           Observer longitude in degrees (east-positive)
  --lat DEG           Observer latitude in degrees
  --alt-m M           Observer altitude in meters
  --utc ISO8601Z      UTC timestamp, e.g. 2006-02-17T06:18:45Z
  --jd JD             Julian Day (UTC scale), e.g. 2453783.7623264

Options:
  --config PATH       Base config template (default: scene.toml)
  --out PATH          Output FITS path (default: OUTPUT/synth_moon_single_<UTC>.fits)
  -h, --help          Show this help

Notes:
  - This script always runs with uv and forces the advanced model:
    Moon Hapke BRDF + DEM geometry + DEM Sun shadowing + extended Sun.
  - If --jd is provided, it is converted to UTC via astropy.
EOF
}

CONFIG="scene.toml"
OUT=""
UTC=""
JD=""
LON=""
LAT=""
ALT_M=""

while (($#)); do
  case "$1" in
    --config) CONFIG="${2:?}"; shift 2 ;;
    --out) OUT="${2:?}"; shift 2 ;;
    --utc) UTC="${2:?}"; shift 2 ;;
    --jd) JD="${2:?}"; shift 2 ;;
    --lon) LON="${2:?}"; shift 2 ;;
    --lat) LAT="${2:?}"; shift 2 ;;
    --alt-m) ALT_M="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$LON" || -z "$LAT" || -z "$ALT_M" ]]; then
  echo "Error: --lon, --lat, and --alt-m are required." >&2
  usage >&2
  exit 2
fi

if [[ -n "$UTC" && -n "$JD" ]]; then
  echo "Error: use either --utc or --jd (not both)." >&2
  usage >&2
  exit 2
fi
if [[ -z "$UTC" && -z "$JD" ]]; then
  echo "Error: one of --utc or --jd is required." >&2
  usage >&2
  exit 2
fi

if [[ -n "$JD" ]]; then
  UTC="$(uv run python - "$JD" <<'PY'
import sys
from astropy.time import Time
jd = float(sys.argv[1])
print(Time(jd, format="jd", scale="utc").isot + "Z")
PY
)"
fi

if [[ -z "$OUT" ]]; then
  stamp="${UTC//:/}"
  stamp="${stamp//-}"
  OUT="OUTPUT/synth_moon_single_${stamp}.fits"
fi

TMP_CFG="$(mktemp /tmp/synthmoon_single_cfg_XXXX.toml)"
cleanup() {
  rm -f "$TMP_CFG"
}
trap cleanup EXIT

uv run python - "$CONFIG" "$TMP_CFG" "$LON" "$LAT" "$ALT_M" <<'PY'
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
lon = float(sys.argv[3])
lat = float(sys.argv[4])
alt_m = float(sys.argv[5])

txt = src.read_text(encoding="utf-8")
lines = txt.splitlines(keepends=True)

section = ""
seen = {
    "observer.lon_deg": False,
    "observer.lat_deg": False,
    "observer.height_m": False,
}

def set_kv(key: str, val: str, line: str) -> str:
    m = re.match(r"^(\s*" + re.escape(key) + r"\s*=\s*)(.*?)(\s*(#.*)?)\n?$", line)
    if not m:
        return line
    newline = "\n" if line.endswith("\n") else ""
    return f"{m.group(1)}{val}{m.group(3)}{newline}"

out = []
for line in lines:
    sm = re.match(r"^\s*\[([^\]]+)\]\s*$", line.strip())
    if sm:
        section = sm.group(1).strip().lower()
        out.append(line)
        continue

    if section == "observer":
        n = line
        nn = set_kv("mode", '"earth_site"', n)
        if nn != n:
            line = nn
        n = line
        nn = set_kv("lon_deg", f"{lon:.10f}", n)
        if nn != n:
            seen["observer.lon_deg"] = True
            line = nn
        n = line
        nn = set_kv("lat_deg", f"{lat:.10f}", n)
        if nn != n:
            seen["observer.lat_deg"] = True
            line = nn
        n = line
        nn = set_kv("height_m", f"{alt_m:.3f}", n)
        if nn != n:
            seen["observer.height_m"] = True
            line = nn

    elif section == "moon":
        n = line
        nn = set_kv("brdf", '"hapke"', n)
        if nn != n:
            line = nn
        n = line
        nn = set_kv("dem_refine_iter", "3", n)
        if nn != n:
            line = nn

    elif section == "sun":
        n = line
        nn = set_kv("extended_disk", "true", n)
        if nn != n:
            line = nn

    elif section == "shadows":
        n = line
        nn = set_kv("mode", '"dem"', n)
        if nn != n:
            line = nn
        n = line
        nn = set_kv("sun", '"dem"', n)
        if nn != n:
            line = nn

    out.append(line)

missing = [k for k, v in seen.items() if not v]
if missing:
    raise SystemExit(
        "Config template missing required observer keys/sections: "
        + ", ".join(missing)
        + f". Please start from a full scene.toml (got: {src})."
    )

dst.write_text("".join(out), encoding="utf-8")
PY

mkdir -p "$(dirname "$OUT")"

echo "UTC: ${UTC}"
echo "Observer: lon=${LON} lat=${LAT} alt_m=${ALT_M}"
echo "Output: ${OUT}"
echo "Config: ${TMP_CFG} (temporary)"

uv run python -m synthmoon.run_v0 \
  --config "$TMP_CFG" \
  --utc "$UTC" \
  --out "$OUT"

echo "Done: $OUT"
