#!/usr/bin/env python3
"""
read_layer5_ratios_with_sublunar_plot.py

For each FITS in OUTPUT/hourly_layer5:
- read image (2D)
- extract I0=img[y0,x0], I1=img[y1,x1], ratio=I0/I1
- read JD-OBS from header
- compute sub-lunar lon/lat on Earth using SPICE (IAU_EARTH) at that time
- write CSV and make plots:
    1) ratio vs index (1..N)
    2) ratio vs sub-lunar longitude
    3) ratio vs sub-lunar latitude

Run inside your uv env (so spiceypy is available):
  PYTHONPATH=$PWD uv run python read_layer5_ratios_with_sublunar_plot.py --x0 10 --y0 10 --x1 20 --y1 20
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pixel ratios + sub-lunar lon/lat from JD-OBS using SPICE.")
    p.add_argument("--pattern", default="OUTPUT/hourly_layer5/synth_layer5_*.fits",
                   help="Glob pattern for FITS files.")
    p.add_argument("--x0", type=int, required=True)
    p.add_argument("--y0", type=int, required=True)
    p.add_argument("--x1", type=int, required=True)
    p.add_argument("--y1", type=int, required=True)
    p.add_argument("--ext", type=int, default=0, help="FITS HDU index (default 0 = primary).")
    p.add_argument("--mk", default=None,
                   help=("Path to SPICE meta-kernel to load. "
                         "If omitted, will try to use FITS header MKFILE/MKNAME (resolved relative to repo root)."))
    p.add_argument("--csv", default="OUTPUT/hourly_layer5_pixel_ratios_with_sublunar.csv")
    p.add_argument("--plot_prefix", default="OUTPUT/hourly_layer5_ratio",
                   help="Prefix for output plots (PNG).")
    return p.parse_args()


def load_2d_image(path: str, ext: int = 0) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[ext].data
        if data is None:
            raise ValueError(f"No data in HDU {ext} for {path}")
        arr = np.asarray(data)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0, :, :]
        if arr.shape[-1] == 1:
            return arr[:, :, 0]
    raise ValueError(f"Unexpected FITS data shape {arr.shape} in {path}")


def get_pix(img: np.ndarray, x: int, y: int) -> float:
    ny, nx = img.shape
    if not (0 <= x < nx and 0 <= y < ny):
        raise IndexError(f"Pixel (x={x}, y={y}) out of bounds for image shape (ny={ny}, nx={nx})")
    return float(img[y, x])


def safe_ratio(a: float, b: float) -> float:
    if b == 0.0 or not np.isfinite(b):
        return float("nan")
    if not np.isfinite(a):
        return float("nan")
    return a / b


def extract_time_from_name(path: str) -> str:
    m = re.search(r"_(\d{4}-\d{2}-\d{2}T\d{6}Z)\.fits$", os.path.basename(path))
    return m.group(1) if m else ""


def jdobs_to_et(jd_obs: str) -> float:
    # FITS header stores JD-OBS as string like "2461084.5000000"
    jd = float(jd_obs)
    # Astropy Time in UTC, then to ISO Z, then SPICE et
    t = Time(jd, format="jd", scale="utc")
    utc = t.isot + "Z"  # e.g. 2026-02-13T00:00:00.000Z
    # SPICE accepts many UTC formats; strip fractional if you like, but not required.
    et = sp.utc2et(utc)
    return float(et)


def sublunar_lonlat_deg(et: float) -> tuple[float, float]:
    """
    Sub-lunar point on Earth (planetocentric lon/lat, IAU_EARTH).
    Vector from Earth->Moon in Earth-fixed frame, then reclat.
    """
    pos_me_ecef_km, _lt = sp.spkpos("MOON", et, "IAU_EARTH", "LT+S", "EARTH")  # km
    _r, lon_rad, lat_rad = sp.reclat(pos_me_ecef_km)
    lon_deg = float(np.rad2deg(lon_rad))
    lat_deg = float(np.rad2deg(lat_rad))
    # keep lon in [-180,180] as returned; also handy to store 0..360
    return lon_deg, lat_deg


def resolve_mk_from_header(hdr) -> str | None:
    mk = hdr.get("MKFILE", "") or hdr.get("MKNAME", "")
    mk = mk.strip()
    if not mk:
        return None
    # In your header it's "generic.tm". In your repo it may live in DATA/ or KERNELS/ etc.
    # We'll try a few common places relative to current working directory.
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
    # fallback: return the raw name and let furnsh fail with a clear error
    return mk


def main() -> None:
    args = parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    # Load meta-kernel once (from args.mk, else from first FITS)
    with fits.open(files[0], memmap=False) as hdul0:
        hdr0 = hdul0[0].header

    mk_path = args.mk or resolve_mk_from_header(hdr0)
    if not mk_path:
        raise SystemExit("Could not determine meta-kernel path. Pass --mk /path/to/meta-kernel.tm")

    sp.kclear()
    sp.furnsh(mk_path)

    rows = []
    xs = []
    ratios = []
    sublons = []
    sublats = []

    for i, f in enumerate(files):
        with fits.open(f, memmap=False) as hdul:
            hdr = hdul[0].header
            jd_obs = hdr.get("JD-OBS", None)
            if jd_obs is None:
                raise ValueError(f"Missing JD-OBS in {f}")
            img = np.asarray(hdul[args.ext].data)

        img2 = load_2d_image(f, ext=args.ext)
        i0 = get_pix(img2, args.x0, args.y0)
        i1 = get_pix(img2, args.x1, args.y1)
        r = safe_ratio(i0, i1)

        et = jdobs_to_et(str(jd_obs))
        lon_deg, lat_deg = sublunar_lonlat_deg(et)

        x = i + 1
        xs.append(x)
        ratios.append(r)
        sublons.append(lon_deg)
        sublats.append(lat_deg)

        rows.append({
            "index_1based": x,
            "file": f,
            "time_utc_from_name": extract_time_from_name(f),
            "JD_OBS": str(jd_obs),
            "x0": args.x0, "y0": args.y0, "i0": i0,
            "x1": args.x1, "y1": args.y1, "i1": i1,
            "ratio_i0_over_i1": r,
            "sublon_deg": lon_deg,
            "sublon_360": lon_deg % 360.0,
            "sublat_deg": lat_deg,
        })

    # Write CSV
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote: {args.csv}")

    # Plot 1: ratio vs index
    prefix = args.plot_prefix
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    plt.figure()
    plt.plot(xs, ratios, marker="o")
    plt.xlabel("Index (1..N, sorted by filename)")
    plt.ylabel("Ratio I(x0,y0) / I(x1,y1)")
    plt.title("Hourly layer5 pixel ratio")
    plt.grid(True)
    plt.tight_layout()
    out1 = f"{prefix}_vs_index.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"Wrote: {out1}")

        # Plot 2: ratio vs sublunar longitude (sorted by longitude to avoid 0/360 wrap jump)
    lon360 = [row["sublon_360"] for row in rows]
    pairs = sorted(zip(lon360, ratios), key=lambda t: t[0])
    lon_sorted = [p[0] for p in pairs]
    ratio_sorted = [p[1] for p in pairs]

    plt.figure()
    plt.plot(lon_sorted, ratio_sorted, marker="o")
    plt.xlabel("Sub-lunar longitude (deg, 0..360, IAU_EARTH) [sorted]")
    plt.ylabel("Ratio I(x0,y0) / I(x1,y1)")
    plt.title("Ratio vs sub-lunar longitude (sorted)")
    plt.grid(True)
    plt.tight_layout()
    out2 = f"{prefix}_vs_sublon_sorted.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Wrote: {out2}")

    # Plot 3: ratio vs sublunar latitude
    plt.figure()
    plt.plot(sublats, ratios, marker="o")
    plt.xlabel("Sub-lunar latitude (deg, IAU_EARTH)")
    plt.ylabel("Ratio I(x0,y0) / I(x1,y1)")
    plt.title("Ratio vs sub-lunar latitude")
    plt.grid(True)
    plt.tight_layout()
    out3 = f"{prefix}_vs_sublat.png"
    plt.savefig(out3, dpi=150)
    plt.close()
    print(f"Wrote: {out3}")

    sp.kclear()


if __name__ == "__main__":
    main()

