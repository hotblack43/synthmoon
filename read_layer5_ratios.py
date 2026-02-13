#!/usr/bin/env python3
"""
read_layer5_ratios_plot.py

Reads FITS files like:
  OUTPUT/hourly_layer5/synth_layer5_###_YYYY-mm-ddTHHMMSSZ.fits

Extracts intensity at (x0,y0) and (x1,y1), computes ratio = I0/I1,
writes a CSV, AND makes a plot with x = 1..N (file order) and y = ratio.

Run (recommended inside your uv env):
  PYTHONPATH=$PWD uv run python read_layer5_ratios_plot.py --x0 123 --y0 45 --x1 200 --y1 210
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract pixel ratios from layer5-only FITS files + plot.")
    p.add_argument("--pattern", default="OUTPUT/hourly_layer5/synth_layer5_*.fits",
                   help="Glob pattern for FITS files.")
    p.add_argument("--x0", type=int, required=True)
    p.add_argument("--y0", type=int, required=True)
    p.add_argument("--x1", type=int, required=True)
    p.add_argument("--y1", type=int, required=True)
    p.add_argument("--ext", type=int, default=0, help="FITS HDU index to read (default 0 = primary).")
    p.add_argument("--csv", default="OUTPUT/hourly_layer5_pixel_ratios.csv",
                   help="Output CSV filename.")
    p.add_argument("--plot", default="OUTPUT/hourly_layer5_pixel_ratios.png",
                   help="Output plot filename (PNG).")
    return p.parse_args()


def extract_time_from_name(path: str) -> str:
    m = re.search(r"_(\d{4}-\d{2}-\d{2}T\d{6}Z)\.fits$", os.path.basename(path))
    return m.group(1) if m else ""


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


def main() -> None:
    args = parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    rows = []
    ratios = []
    xs = []

    for idx0, f in enumerate(files):
        img = load_2d_image(f, ext=args.ext)
        i0 = get_pix(img, args.x0, args.y0)
        i1 = get_pix(img, args.x1, args.y1)
        r = safe_ratio(i0, i1)

        # x-axis: 1..N
        x = idx0 + 1
        xs.append(x)
        ratios.append(r)

        rows.append({
            "index_1based": x,
            "file": f,
            "time_utc": extract_time_from_name(f),
            "x0": args.x0, "y0": args.y0, "i0": i0,
            "x1": args.x1, "y1": args.y1, "i1": i1,
            "ratio_i0_over_i1": r,
        })

    # Write CSV
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote: {args.csv}")

    # Plot
    os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
    plt.figure()
    plt.plot(xs, ratios, marker="o")
    plt.xlabel("Index (1..N, sorted by filename)")
    plt.ylabel("Ratio I(x0,y0) / I(x1,y1)")
    plt.title("Hourly layer5 pixel ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    plt.close()
    print(f"Wrote: {args.plot}")


if __name__ == "__main__":
    main()

