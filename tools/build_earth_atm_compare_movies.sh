#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/build_earth_atm_compare_movies.sh --start-jd JD --end-jd JD --step-hours H [options]

Required:
  --start-jd JD         Start Julian Day (UTC)
  --end-jd JD           End Julian Day (UTC)
  --step-hours H        Frame spacing in hours

Options:
  --config PATH         Base config template (default: scene.toml)
  --nx N                Output width (default: 1024)
  --ny N                Output height (default: 1024)
  --fps N               Output video fps (default: 12)
  --crf N               ffmpeg CRF (default: 18)
  --pad-frac X          Padding fraction around disk (default: 0.10)
  --workdir DIR         Working directory root (default: /tmp/synthmoon_earth_atm_compare)
  --atm-on-mp4 PATH     Atmosphere-on MP4 (default: OUTPUT/earth_movie_3d_20min_atm_on.mp4)
  --atm-off-mp4 PATH    Atmosphere-off MP4 (default: OUTPUT/earth_movie_3d_20min_atm_off.mp4)
  --keep-frames         Keep frame/FITS workdirs instead of deleting them
  -h, --help            Show this help
EOF
}

CONFIG="scene.toml"
START_JD=""
END_JD=""
STEP_HOURS=""
NX="1024"
NY="1024"
FPS="12"
CRF="18"
PAD_FRAC="0.10"
WORKDIR="/tmp/synthmoon_earth_atm_compare"
ATM_ON_MP4="OUTPUT/earth_movie_3d_20min_atm_on.mp4"
ATM_OFF_MP4="OUTPUT/earth_movie_3d_20min_atm_off.mp4"
KEEP_FRAMES=0

while (($#)); do
  case "$1" in
    --config) CONFIG="${2:?}"; shift 2 ;;
    --start-jd) START_JD="${2:?}"; shift 2 ;;
    --end-jd) END_JD="${2:?}"; shift 2 ;;
    --step-hours) STEP_HOURS="${2:?}"; shift 2 ;;
    --nx) NX="${2:?}"; shift 2 ;;
    --ny) NY="${2:?}"; shift 2 ;;
    --fps) FPS="${2:?}"; shift 2 ;;
    --crf) CRF="${2:?}"; shift 2 ;;
    --pad-frac) PAD_FRAC="${2:?}"; shift 2 ;;
    --workdir) WORKDIR="${2:?}"; shift 2 ;;
    --atm-on-mp4) ATM_ON_MP4="${2:?}"; shift 2 ;;
    --atm-off-mp4) ATM_OFF_MP4="${2:?}"; shift 2 ;;
    --keep-frames) KEEP_FRAMES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$START_JD" || -z "$END_JD" || -z "$STEP_HOURS" ]]; then
  echo "Error: --start-jd, --end-jd, and --step-hours are required." >&2
  usage >&2
  exit 2
fi

mkdir -p "$(dirname "$ATM_ON_MP4")" "$(dirname "$ATM_OFF_MP4")" "$WORKDIR"

ATM_ON_CFG="${WORKDIR}/scene_atm_on.toml"
ATM_OFF_CFG="${WORKDIR}/scene_atm_off.toml"
cp "$CONFIG" "$ATM_ON_CFG"
cp "$CONFIG" "$ATM_OFF_CFG"
perl -0pi -e 's/atmosphere_enable = false/atmosphere_enable = true/g; s/atmosphere_enable = true/atmosphere_enable = true/g' "$ATM_ON_CFG"
perl -0pi -e 's/atmosphere_enable = true/atmosphere_enable = false/g' "$ATM_OFF_CFG"

echo "Atmosphere-on config:  $ATM_ON_CFG"
echo "Atmosphere-off config: $ATM_OFF_CFG"

CMD_COMMON=(
  uv run python tools/build_earth_color_movie_jd.py
  --start-jd "$START_JD"
  --end-jd "$END_JD"
  --step-hours "$STEP_HOURS"
  --nx "$NX"
  --ny "$NY"
  --fps "$FPS"
  --crf "$CRF"
  --pad-frac "$PAD_FRAC"
)

if ((KEEP_FRAMES)); then
  CMD_COMMON+=(--keep-frames)
fi

echo "Building atmosphere-on movie..."
"${CMD_COMMON[@]}" \
  --config "$ATM_ON_CFG" \
  --workdir "${WORKDIR}/atm_on" \
  --out-mp4 "$ATM_ON_MP4"

echo "Building atmosphere-off movie..."
"${CMD_COMMON[@]}" \
  --config "$ATM_OFF_CFG" \
  --workdir "${WORKDIR}/atm_off" \
  --out-mp4 "$ATM_OFF_MP4"

echo "Done:"
echo "  $ATM_ON_MP4"
echo "  $ATM_OFF_MP4"
