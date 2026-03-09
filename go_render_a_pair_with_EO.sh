#!/usr/bin/env bash
set -euo pipefail

# JD-only wrapper invocation.
# Edit JD and site coordinates here as needed, then run:
#   ./go_render_a_pair_with_EO.sh

JD="2455748.7651276"
LON="-155.5763"
LAT="19.5362"
ALT_M="3397"

tools/go_earth_moon_pair.sh \
  --lon "${LON}" \
  --lat "${LAT}" \
  --alt-m "${ALT_M}" \
  --jd "${JD}" \
  --earth-out "OUTPUT/earth_pair_jd_${JD}.fits" \
  --moon-out "OUTPUT/moon_pair_jd_${JD}.fits"
