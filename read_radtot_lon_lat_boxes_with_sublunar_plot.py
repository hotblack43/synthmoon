#!/usr/bin/env python3
"""
read_radtot_lon_lat_boxes_with_sublunar_plot.py  (v4)

You are right: for Earth-based viewing, on-disc *visible* selenographic longitudes
should stay near ±90 (plus a few degrees). If SELON on-disc reaches ±150..±180, the
most likely cause is that the forward model wrote the *wrong intersection* of the ray
with the lunar sphere/ellipsoid for some pixels (the far-side root instead of the near-side root).

This analysis script cannot fix the renderer, but it *can* protect your box extraction
by enforcing a "visible-hemisphere" sanity mask:

  abs(SELON) <= VISIBLE_LON_LIMIT_DEG   (default 100 deg)

This removes any mistakenly-tagged far-side pixels that still lie inside the image disc.

Everything else from v3 remains:
  - layer selection by LAYn names: RADTOT, SELON, SELAT
  - disc mask from IFTOTAL/SCALED/RADTOT > DISC_THRESHOLD
  - proper terrestrial sublunar lon/lat via SPICE subpnt()
  - optional auto-expansion to avoid empty boxes

Run:
  uv run python read_radtot_lon_lat_boxes_with_sublunar_plot.py
"""

from __future__ import annotations

import csv
import glob
import os
import re
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
import matplotlib.pyplot as plt
import spiceypy as sp


# ----------------------------
# HARDWIRED CONFIG
# ----------------------------

FITS_PATTERN = "OUTPUT/hourly_layer5/synth_*.fits"
HDU_INDEX = 0

RAD_LAYER_NAME = "RADTOT"
LON_LAYER_NAME = "SELON"
LAT_LAYER_NAME = "SELAT"
IFTOTAL_LAYER_NAME = "IFTOTAL"
SCALED_LAYER_NAME = "SCALED"

# Fallback only if names aren't present:
RADTOT_LAYER_1BASED_FALLBACK = 5
SELON_LAYER_1BASED_FALLBACK = 11
SELAT_LAYER_1BASED_FALLBACK = 12
IFTOTAL_LAYER_1BASED_FALLBACK = 2
SCALED_LAYER_1BASED_FALLBACK = 1

# Your boxes (signed lon assumed)
BOX_A = dict(name="A", lon_min=-68.5, lon_max=-68.4, lat_min=-5.2, lat_max=-5.0)
BOX_B = dict(name="B", lon_min=59.0,  lon_max=59.2,  lat_min=16.9, lat_max=17.1)

OUT_CSV = "OUTPUT/radtot_box_means_with_sublunar.csv"
PLOT_PREFIX = "OUTPUT/radtot_box_ratio"

MK_PATH = None  # or "DATA/generic.tm"

BOX_STAT = "mean"  # "mean" or "median"
PLOT_MEANS_TOO = True

# Disc masking: pixels with IFTOTAL/SCALED/RADTOT > threshold are considered on-disc
DISC_THRESHOLD = 0.0

# Visible-hemisphere sanity constraint for SELON (deg).
# For Earth-based viewing this should be close to ±90; set slightly larger for safety.
VISIBLE_LON_LIMIT_DEG = 100.0

# Diagnostics
DEBUG_PRINT_PER_FILE = True

# Auto-expand boxes until at least MIN_PIXELS are selected (or we hit MAX_EXPAND_DEG).
AUTO_EXPAND_BOXES = True
MIN_PIXELS_PER_BOX = 10
EXPAND_STEP_DEG = 0.2
MAX_EXPAND_DEG = 10.0


# ----------------------------
# Helpers
# ----------------------------

def extract_time_from_name(path: str) -> str:
    m = re.search(r"_(\d{4}-\d{2}-\d{2}T\d{6}Z)\.fits$", os.path.basename(path))
    return m.group(1) if m else ""


def resolve_mk_from_header(hdr) -> str | None:
    mk = (hdr.get("MKFILE", "") or hdr.get("MKNAME", "")).strip()
    if not mk:
        return None
    candidates = [
        Path(mk),
        Path("DATA") / mk,
        Path("KERNELS") / mk,
        Path("kernels") / mk,
        Path("spice") / mk,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return mk


def jdobs_to_et(jd_obs: str) -> float:
    jd = float(jd_obs)
    t = Time(jd, format="jd", scale="utc")
    utc = t.isot + "Z"
    return float(sp.utc2et(utc))


def parse_et_from_header(hdr) -> float:
    date_obs = hdr.get("DATE-OBS", None)
    if date_obs is not None:
        s = str(date_obs).strip()
        if not s.endswith("Z"):
            s = s + "Z"
        return float(sp.utc2et(s))
    jd_obs = hdr.get("JD-OBS", None)
    if jd_obs is None:
        raise ValueError("Missing DATE-OBS and JD-OBS in header")
    return jdobs_to_et(str(jd_obs))


def sublunar_lonlat_deg(et: float) -> tuple[float, float]:
    spoint, _trgepc, _srfvec = sp.subpnt(
        "Near point: ellipsoid",
        "EARTH",
        et,
        "IAU_EARTH",
        "LT+S",
        "MOON",
    )
    _r, lon_rad, lat_rad = sp.reclat(spoint)
    return float(np.rad2deg(lon_rad)), float(np.rad2deg(lat_rad))


def load_cube(path: str, hdu_index: int) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[hdu_index].data
        hdr = hdul[hdu_index].header
        if data is None:
            raise ValueError(f"No data in HDU {hdu_index} for {path}")
        arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D FITS cube, got shape {arr.shape} in {path}")
    return arr, hdr


def layer_1based_to_0based(layer_1based: int) -> int:
    if layer_1based < 1:
        raise ValueError("Layer index must be 1-based (>= 1)")
    return layer_1based - 1


def find_layer_index_by_name(hdr, wanted: str) -> int | None:
    wanted_norm = wanted.strip().upper()
    nlay = int(hdr.get("NLAYERS", hdr.get("NAXIS3", 0)) or 0)
    if nlay <= 0:
        return None
    any_lay_cards = False
    for n in range(1, nlay + 1):
        key = f"LAY{n}"
        if key in hdr:
            any_lay_cards = True
            name = str(hdr[key]).strip().upper()
            if name == wanted_norm:
                return n - 1
    if not any_lay_cards:
        return None
    return None


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    m = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def detect_signed_lon_domain(lon_deg: np.ndarray) -> bool:
    v = lon_deg[np.isfinite(lon_deg)]
    if v.size == 0:
        return True
    lo = float(np.nanmin(v))
    hi = float(np.nanmax(v))
    has_neg = np.any(v < 0)
    has_pos = np.any(v > 0)
    if lo >= -180.1 and hi <= 180.1 and has_neg and has_pos:
        return True
    return False


def lon_in_box_signed(lon_deg: np.ndarray, lon_min: float, lon_max: float) -> np.ndarray:
    if lon_min <= lon_max:
        return (lon_deg >= lon_min) & (lon_deg <= lon_max)
    else:
        return (lon_deg >= lon_min) | (lon_deg <= lon_max)


def lon_in_box_360(lon_deg: np.ndarray, lon_min: float, lon_max: float) -> np.ndarray:
    lon360 = np.mod(lon_deg, 360.0)
    a = lon_min % 360.0
    b = lon_max % 360.0
    if a <= b:
        return (lon360 >= a) & (lon360 <= b)
    else:
        return (lon360 >= a) | (lon360 <= b)


def compute_box_stat(values: np.ndarray, stat: str) -> float:
    if values.size == 0:
        return float("nan")
    if stat == "mean":
        return float(np.nanmean(values))
    if stat == "median":
        return float(np.nanmedian(values))
    raise ValueError(f"Unknown BOX_STAT={stat!r}")


def safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b == 0.0:
        return float("nan")
    return a / b


def make_box_mask(lon: np.ndarray, lat: np.ndarray, good: np.ndarray,
                  lon_min: float, lon_max: float, lat_min: float, lat_max: float,
                  lon_is_signed: bool) -> np.ndarray:
    m_lon = lon_in_box_signed(lon, lon_min, lon_max) if lon_is_signed else lon_in_box_360(lon, lon_min, lon_max)
    return good & m_lon & (lat >= lat_min) & (lat <= lat_max)


def expanded_mask_until_minpix(lon: np.ndarray, lat: np.ndarray, good: np.ndarray,
                               box: dict, lon_is_signed: bool) -> tuple[np.ndarray, float]:
    expand = 0.0
    while True:
        lon_min = box["lon_min"] - expand
        lon_max = box["lon_max"] + expand
        lat_min = box["lat_min"] - expand
        lat_max = box["lat_max"] + expand
        m = make_box_mask(lon, lat, good, lon_min, lon_max, lat_min, lat_max, lon_is_signed)
        if (not AUTO_EXPAND_BOXES) or (np.sum(m) >= MIN_PIXELS_PER_BOX) or (expand >= MAX_EXPAND_DEG):
            return m, expand
        expand += EXPAND_STEP_DEG


def disc_mask_from_layers(hdr, cube: np.ndarray) -> tuple[np.ndarray, str, int]:
    candidates = [
        (IFTOTAL_LAYER_NAME, IFTOTAL_LAYER_1BASED_FALLBACK),
        (SCALED_LAYER_NAME, SCALED_LAYER_1BASED_FALLBACK),
        (RAD_LAYER_NAME, RADTOT_LAYER_1BASED_FALLBACK),
    ]
    for nm, fallback_1b in candidates:
        idx = find_layer_index_by_name(hdr, nm)
        if idx is None:
            idx = layer_1based_to_0based(fallback_1b)
        if 0 <= idx < cube.shape[0]:
            layer = np.asarray(cube[idx, :, :], dtype=float)
            m = np.isfinite(layer) & (layer > DISC_THRESHOLD)
            if np.any(m):
                return m, nm, int(idx)
    return np.ones((cube.shape[1], cube.shape[2]), dtype=bool), "NONE", -1


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    files = sorted(glob.glob(FITS_PATTERN))
    if not files:
        raise SystemExit(f"No files matched pattern: {FITS_PATTERN}")

    _, hdr0 = load_cube(files[0], HDU_INDEX)
    mk_path = MK_PATH or resolve_mk_from_header(hdr0)
    if not mk_path:
        raise SystemExit("Could not determine meta-kernel path. Set MK_PATH or add MKFILE/MKNAME to FITS headers.")

    sp.kclear()
    sp.furnsh(mk_path)

    rows = []
    xs = []
    meanA_list = []
    meanB_list = []
    ratio_list = []
    sublons = []
    sublats = []

    for i, f in enumerate(files):
        cube, hdr = load_cube(f, HDU_INDEX)

        rad_idx = find_layer_index_by_name(hdr, RAD_LAYER_NAME)
        lon_idx = find_layer_index_by_name(hdr, LON_LAYER_NAME)
        lat_idx = find_layer_index_by_name(hdr, LAT_LAYER_NAME)

        used_names = True
        if rad_idx is None or lon_idx is None or lat_idx is None:
            used_names = False
            rad_idx = layer_1based_to_0based(RADTOT_LAYER_1BASED_FALLBACK)
            lon_idx = layer_1based_to_0based(SELON_LAYER_1BASED_FALLBACK)
            lat_idx = layer_1based_to_0based(SELAT_LAYER_1BASED_FALLBACK)

        rad = np.asarray(cube[rad_idx, :, :], dtype=float)
        lon = np.asarray(cube[lon_idx, :, :], dtype=float)
        lat = np.asarray(cube[lat_idx, :, :], dtype=float)

        disc_mask, disc_layer_used, disc_idx0 = disc_mask_from_layers(hdr, cube)

        # Visible hemisphere sanity mask (protect against far-side root mistakes)
        visible_lon_mask = np.isfinite(lon) & (np.abs(lon) <= VISIBLE_LON_LIMIT_DEG)

        good = finite_mask(rad, lon, lat) & disc_mask & visible_lon_mask
        good_frac = float(np.sum(good)) / float(good.size)

        lon_is_signed = detect_signed_lon_domain(lon)

        mA, expA = expanded_mask_until_minpix(lon, lat, good, BOX_A, lon_is_signed)
        mB, expB = expanded_mask_until_minpix(lon, lat, good, BOX_B, lon_is_signed)

        meanA = compute_box_stat(rad[mA], BOX_STAT)
        meanB = compute_box_stat(rad[mB], BOX_STAT)

        # Raw ratio (A/B)
        ratio_raw = safe_ratio(meanA, meanB)

        # Symmetric ratio for plotting: in (0,1] by inverting if > 1
        if np.isfinite(ratio_raw) and ratio_raw > 0.0 and ratio_raw > 1.0:
            ratio = 1.0 / ratio_raw
        else:
            ratio = ratio_raw
        et = parse_et_from_header(hdr)
        sublon_deg, sublat_deg = sublunar_lonlat_deg(et)

        if DEBUG_PRINT_PER_FILE:
            lon_valid = lon[finite_mask(lon, lat) & disc_mask]
            lat_valid = lat[finite_mask(lon, lat) & disc_mask]
            # Pre-visibility mask range
            lon_minv = float(np.nanmin(lon_valid)) if lon_valid.size else float("nan")
            lon_maxv = float(np.nanmax(lon_valid)) if lon_valid.size else float("nan")
            # Post-visibility mask range
            lon_vis = lon[good]
            lon_vis_min = float(np.nanmin(lon_vis)) if lon_vis.size else float("nan")
            lon_vis_max = float(np.nanmax(lon_vis)) if lon_vis.size else float("nan")
            lat_minv = float(np.nanmin(lat_valid)) if lat_valid.size else float("nan")
            lat_maxv = float(np.nanmax(lat_valid)) if lat_valid.size else float("nan")

            dom = "signed(-180..180]" if lon_is_signed else "0..360"
            out_frac = 1.0 - (float(np.sum(visible_lon_mask & disc_mask)) / float(np.sum(disc_mask)) if np.sum(disc_mask) else 0.0)
            print(
                f"[{i+1:04d}/{len(files):04d}] {os.path.basename(f)}  "
                f"disc={disc_layer_used}[{disc_idx0}]  lon_dom={dom}  "
                f"good_frac={good_frac:.3f} far_lon_rejected_frac~{out_frac:.3f}  "
                f"npixA={int(np.sum(mA))} expA={expA:.1f}  npixB={int(np.sum(mB))} expB={expB:.1f}  "
                f"meanA={meanA:.6g} meanB={meanB:.6g} ratio_raw={ratio_raw:.6g} ratio_sym={ratio:.6g}  "
                f"SELON_range_disc=({lon_minv:.3f},{lon_maxv:.3f})  "
                f"SELON_range_visible=({lon_vis_min:.3f},{lon_vis_max:.3f})  "
                f"SELAT_range_disc=({lat_minv:.3f},{lat_maxv:.3f})  "
                f"sublon={sublon_deg:.3f} sublat={sublat_deg:.3f}  used_names={1 if used_names else 0}"
            )

        x = i + 1
        xs.append(x)
        meanA_list.append(meanA)
        meanB_list.append(meanB)
        ratio_list.append(ratio)
        sublons.append(sublon_deg)
        sublats.append(sublat_deg)

        rows.append({
            "index_1based": x,
            "file": f,
            "time_utc_from_name": extract_time_from_name(f),
            "DATE_OBS": str(hdr.get("DATE-OBS", "")),
            "JD_OBS": str(hdr.get("JD-OBS", "")),

            "rad_layer_name": RAD_LAYER_NAME,
            "lon_layer_name": LON_LAYER_NAME,
            "lat_layer_name": LAT_LAYER_NAME,
            "rad_layer_index0": int(rad_idx),
            "lon_layer_index0": int(lon_idx),
            "lat_layer_index0": int(lat_idx),
            "used_layer_names": 1 if used_names else 0,

            "disc_layer_used": disc_layer_used,
            "disc_layer_index0": disc_idx0,
            "disc_threshold": DISC_THRESHOLD,

            "visible_lon_limit_deg": VISIBLE_LON_LIMIT_DEG,
            "lon_domain_signed": 1 if lon_is_signed else 0,
            "good_frac": good_frac,

            "boxA_lon_min": BOX_A["lon_min"],
            "boxA_lon_max": BOX_A["lon_max"],
            "boxA_lat_min": BOX_A["lat_min"],
            "boxA_lat_max": BOX_A["lat_max"],
            "boxA_stat": BOX_STAT,
            "boxA_value": meanA,
            "boxA_npix": int(np.sum(mA)),
            "boxA_expand_deg": expA,

            "boxB_lon_min": BOX_B["lon_min"],
            "boxB_lon_max": BOX_B["lon_max"],
            "boxB_lat_min": BOX_B["lat_min"],
            "boxB_lat_max": BOX_B["lat_max"],
            "boxB_stat": BOX_STAT,
            "boxB_value": meanB,
            "boxB_npix": int(np.sum(mB)),
            "boxB_expand_deg": expB,

            "ratio_A_over_B": ratio_raw,
            "ratio_sym_0_1": ratio,  # inverted if raw>1


            "sublon_deg": float(sublon_deg),
            "sublon_360": float(sublon_deg % 360.0),
            "sublat_deg": float(sublat_deg),
        })

    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote: {OUT_CSV}")

    os.makedirs(os.path.dirname(PLOT_PREFIX) or ".", exist_ok=True)

    plt.figure()
    plt.plot(xs, ratio_list, marker="o")
    plt.xlabel("Index (1..N, sorted by filename)")
    plt.ylabel(f"{BOX_STAT}(RADTOT in BoxA) / {BOX_STAT}(RADTOT in BoxB)")
    plt.title("RADTOT box ratio")
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.tight_layout()
    out1 = f"{PLOT_PREFIX}_vs_index.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"Wrote: {out1}")

    lon360 = [row["sublon_360"] for row in rows]
    pairs = sorted(zip(lon360, ratio_list), key=lambda t: t[0])
    lon_sorted = [p[0] for p in pairs]
    ratio_sorted = [p[1] for p in pairs]

    plt.figure()
    plt.plot(lon_sorted, ratio_sorted, marker="o")
    plt.xlabel("Sub-lunar longitude (deg, 0..360, IAU_EARTH) [sorted]")
    plt.ylabel(f"{BOX_STAT}(RADTOT BoxA) / {BOX_STAT}(RADTOT BoxB)")
    plt.title("RADTOT box ratio vs sub-lunar longitude (sorted)")
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.tight_layout()
    out2 = f"{PLOT_PREFIX}_vs_sublon_sorted.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Wrote: {out2}")

    plt.figure()
    plt.plot(sublats, ratio_list, marker="o")
    plt.xlabel("Sub-lunar latitude (deg, IAU_EARTH)")
    plt.ylabel(f"{BOX_STAT}(RADTOT BoxA) / {BOX_STAT}(RADTOT BoxB)")
    plt.title("RADTOT box ratio vs sub-lunar latitude")
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.tight_layout()
    out3 = f"{PLOT_PREFIX}_vs_sublat.png"
    plt.savefig(out3, dpi=150)
    plt.close()
    print(f"Wrote: {out3}")

    if PLOT_MEANS_TOO:
        plt.figure()
        plt.plot(xs, meanA_list, marker="o")
        plt.xlabel("Index (1..N, sorted by filename)")
        plt.ylabel(f"{BOX_STAT}(RADTOT in BoxA)")
        plt.title("RADTOT BoxA statistic vs index")
        plt.grid(True)
        plt.ylim(bottom=0)
        plt.tight_layout()
        out4 = f"{PLOT_PREFIX}_boxA_vs_index.png"
        plt.savefig(out4, dpi=150)
        plt.close()
        print(f"Wrote: {out4}")

        plt.figure()
        plt.plot(xs, meanB_list, marker="o")
        plt.xlabel("Index (1..N, sorted by filename)")
        plt.ylabel(f"{BOX_STAT}(RADTOT in BoxB)")
        plt.title("RADTOT BoxB statistic vs index")
        plt.grid(True)
        plt.ylim(bottom=0)
        plt.tight_layout()
        out5 = f"{PLOT_PREFIX}_boxB_vs_index.png"
        plt.savefig(out5, dpi=150)
        plt.close()
        print(f"Wrote: {out5}")

    sp.kclear()


if __name__ == "__main__":
    main()
