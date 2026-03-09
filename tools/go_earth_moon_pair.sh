#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/go_earth_moon_pair.sh --lon DEG --lat DEG --alt-m M (--utc ISO8601Z | --jd JD) [options]

Required:
  --lon DEG           Observer longitude in degrees (east-positive)
  --lat DEG           Observer latitude in degrees
  --alt-m M           Observer altitude in meters
  --utc ISO8601Z      UTC timestamp, e.g. 2011-07-06T06:21:47Z
  --jd JD             Julian Day (UTC scale), e.g. 2455748.7651276

Options:
  --config PATH       Base config template (default: scene.toml)
  --out-dir DIR       Output directory (default: OUTPUT)
  --earth-out PATH    Explicit Earth output FITS path
  --moon-out PATH     Explicit Moon output FITS path
  -h, --help          Show this help

Behavior:
  - Uses the EO Earth workflow:
    MODIS daily clouds + NSIDC daily sea ice + MODIS static land ice.
  - If any required EO file is missing, the script stops and prints the
    exact uv-run commands needed to fetch/extract it.
EOF
}

CONFIG="scene.toml"
OUT_DIR="OUTPUT"
EARTH_OUT=""
MOON_OUT=""
UTC=""
JD=""
LON=""
LAT=""
ALT_M=""

while (($#)); do
  case "$1" in
    --config) CONFIG="${2:?}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:?}"; shift 2 ;;
    --earth-out) EARTH_OUT="${2:?}"; shift 2 ;;
    --moon-out) MOON_OUT="${2:?}"; shift 2 ;;
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
  UTC="$(UV_CACHE_DIR=/tmp/uvcache uv run python - "$JD" <<'PY'
import sys
from astropy.time import Time
jd = float(sys.argv[1])
print(Time(jd, format="jd", scale="utc").isot + "Z")
PY
)"
fi

readarray -t DATEINFO < <(UV_CACHE_DIR=/tmp/uvcache uv run python - "$UTC" <<'PY'
import sys
import datetime as dt
t = dt.datetime.fromisoformat(sys.argv[1].replace("Z","+00:00")).astimezone(dt.timezone.utc)
print(t.strftime("%Y"))
print(t.strftime("%Y%m%d"))
print(t.strftime("%j"))
print(t.strftime("%Y-%m-%dT%H:%M:%SZ"))
PY
)
YEAR="${DATEINFO[0]}"
DATE8="${DATEINFO[1]}"
DOY="${DATEINFO[2]}"
UTC_CANON="${DATEINFO[3]}"

MODIS_CF_GLOB="DATA/MODIS/mod08_d3.a${YEAR}${DOY}.*_cloud_fraction.fits"
MODIS_TAU_GLOB="DATA/MODIS/mod08_d3.a${YEAR}${DOY}.*_cloud_tau.fits"
NSIDC_ICE="DATA/NSIDC/ice_fraction_${DATE8}.fits"
LAND_ICE="DATA/MODIS/land_ice_mask_${YEAR}.fits"

first_glob() {
  shopt -s nullglob
  local matches=($1)
  shopt -u nullglob
  if ((${#matches[@]} == 0)); then
    return 1
  fi
  printf '%s\n' "${matches[0]}"
}

MISSING=0
MODIS_CF=""
MODIS_TAU=""
if ! MODIS_CF="$(first_glob "$MODIS_CF_GLOB")"; then
  MISSING=1
fi
if ! MODIS_TAU="$(first_glob "$MODIS_TAU_GLOB")"; then
  MISSING=1
fi
if [[ ! -f "$NSIDC_ICE" ]]; then
  MISSING=1
fi
if [[ ! -f "$LAND_ICE" ]]; then
  MISSING=1
fi

if ((MISSING)); then
  echo "Required EO input files are missing for UTC ${UTC_CANON}." >&2
  echo >&2
  echo "Run these commands, then rerun this script:" >&2
  echo >&2
  if [[ -z "$MODIS_CF" || -z "$MODIS_TAU" ]]; then
    echo "# MODIS daily clouds" >&2
    echo "uv run python tools/download_modis_cloud_granule.py \\" >&2
    echo "  --utc ${UTC_CANON} \\" >&2
    echo "  --product MOD08_D3 \\" >&2
    echo "  --out-dir DATA/MODIS" >&2
    echo >&2
    echo "uv run python tools/extract_modis_l3_cloud_maps.py \\" >&2
    echo "  --in-hdf 'DATA/MODIS/MOD08_D3.A${YEAR}${DOY}*.hdf' \\" >&2
    echo "  --out-dir DATA/MODIS" >&2
    echo >&2
  fi
  if [[ ! -f "$NSIDC_ICE" ]]; then
    echo "# NSIDC daily sea ice" >&2
    echo "uv run python tools/download_nsidc_g02202_daily.py \\" >&2
    echo "  --utc ${UTC_CANON} \\" >&2
    echo "  --hemisphere both \\" >&2
    echo "  --out-dir DATA/NSIDC" >&2
    echo >&2
    echo "uv run python tools/extract_nsidc_g02202_ice_map.py \\" >&2
    echo "  --north-nc DATA/NSIDC/sic_psn25_${DATE8}_*.nc \\" >&2
    echo "  --south-nc DATA/NSIDC/sic_pss25_${DATE8}_*.nc \\" >&2
    echo "  --out-fits ${NSIDC_ICE}" >&2
    echo >&2
  fi
  if [[ ! -f "$LAND_ICE" ]]; then
    echo "# MODIS static land ice" >&2
    echo "uv run python tools/download_modis_landcover_file.py \\" >&2
    echo "  --year ${YEAR} \\" >&2
    echo "  --product MCD12C1 \\" >&2
    echo "  --out-dir DATA/MODIS" >&2
    echo >&2
    echo "uv run python tools/extract_modis_landice_mask.py \\" >&2
    echo "  --in-hdf 'DATA/MODIS/MCD12C1.A${YEAR}001*.hdf' \\" >&2
    echo "  --out-fits ${LAND_ICE}" >&2
    echo >&2
  fi
  exit 3
fi

STAMP="${UTC_CANON//:/}"
STAMP="${STAMP//-/}"
mkdir -p "$OUT_DIR"
if [[ -z "$EARTH_OUT" ]]; then
  EARTH_OUT="${OUT_DIR}/earth_pair_${STAMP}.fits"
fi
if [[ -z "$MOON_OUT" ]]; then
  MOON_OUT="${OUT_DIR}/moon_pair_${STAMP}.fits"
fi

TMP_CFG="$(mktemp /tmp/synthmoon_pair_cfg_XXXX.toml)"
cleanup() {
  rm -f "$TMP_CFG"
}
trap cleanup EXIT

UV_CACHE_DIR=/tmp/uvcache uv run python - "$CONFIG" "$TMP_CFG" "$LON" "$LAT" "$ALT_M" "$MODIS_CF" "$MODIS_TAU" "$NSIDC_ICE" "$LAND_ICE" <<'PY'
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
lon = float(sys.argv[3])
lat = float(sys.argv[4])
alt_m = float(sys.argv[5])
modis_cf = sys.argv[6]
modis_tau = sys.argv[7]
nsidc_ice = sys.argv[8]
land_ice = sys.argv[9]

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
        for key, val in [
            ("mode", '"earth_site"'),
            ("lon_deg", f"{lon:.10f}"),
            ("lat_deg", f"{lat:.10f}"),
            ("height_m", f"{alt_m:.3f}"),
        ]:
            new_line = set_kv(key, val, line)
            if new_line != line:
                line = new_line
                if key in ("lon_deg", "lat_deg", "height_m"):
                    seen[f"observer.{key}"] = True

    elif section == "moon":
        for key, val in [("brdf", '"hapke"'), ("dem_refine_iter", "3")]:
            line = set_kv(key, val, line)

    elif section == "sun":
        line = set_kv("extended_disk", "true", line)

    elif section == "shadows":
        for key, val in [("mode", '"dem"'), ("sun", '"dem"')]:
            line = set_kv(key, val, line)

    elif section == "earth":
        for key, val in [
            ("class_map_fits", '""'),
            ("cloud_fraction_map_fits", f'"{modis_cf}"'),
            ("cloud_tau_map_fits", f'"{modis_tau}"'),
            ("cloud_map_lon_mode", '"-180_180"'),
            ("ice_fraction_map_fits", f'"{nsidc_ice}"'),
            ("ice_map_lon_mode", '"-180_180"'),
            ("ice_fraction_blend", "true"),
            ("land_ice_mask_fits", f'"{land_ice}"'),
            ("land_ice_mask_lon_mode", '"-180_180"'),
            ("land_ice_mask_blend", "true"),
            ("seasonal_ice_enable", "false"),
        ]:
            line = set_kv(key, val, line)

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

echo "UTC: ${UTC_CANON}"
echo "Observer: lon=${LON} lat=${LAT} alt_m=${ALT_M}"
echo "Cloud map: ${MODIS_CF}"
echo "Cloud tau: ${MODIS_TAU}"
echo "Sea ice: ${NSIDC_ICE}"
echo "Land ice: ${LAND_ICE}"
echo "Earth out: ${EARTH_OUT}"
echo "Moon out: ${MOON_OUT}"
echo "Config: ${TMP_CFG} (temporary)"

UV_CACHE_DIR=/tmp/uvcache uv run python tools/render_earth_fits.py \
  --config "$TMP_CFG" \
  --utc "$UTC_CANON" \
  --out "$EARTH_OUT"

UV_CACHE_DIR=/tmp/uvcache uv run python -m synthmoon.run_v0 \
  --config "$TMP_CFG" \
  --utc "$UTC_CANON" \
  --out "$MOON_OUT"

echo "Done:"
echo "  $EARTH_OUT"
echo "  $MOON_OUT"
