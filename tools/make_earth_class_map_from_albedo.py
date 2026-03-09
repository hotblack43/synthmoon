#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a simple Earth class-map FITS from an albedo map.")
    ap.add_argument("--in-fits", required=True, help="Input Earth albedo FITS (2D equirectangular)")
    ap.add_argument("--out-fits", required=True, help="Output class-map FITS (2D)")
    ap.add_argument("--ocean-threshold", type=float, default=0.12, help="A < threshold -> ocean class")
    ap.add_argument("--ice-lat-deg", type=float, default=70.0, help="|lat| >= this -> ice class")
    ap.add_argument("--ocean-class", type=int, default=0)
    ap.add_argument("--land-class", type=int, default=1)
    ap.add_argument("--ice-class", type=int, default=2)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    p_in = Path(args.in_fits)
    arr = fits.getdata(p_in).astype(np.float64)
    if arr.ndim != 2:
        raise SystemExit(f"Expected 2D FITS input, got shape={arr.shape}")

    ny, nx = arr.shape
    lat = 90.0 - (np.arange(ny, dtype=np.float64) + 0.5) * (180.0 / ny)
    lat2d = np.repeat(lat[:, None], nx, axis=1)

    cls = np.full(arr.shape, int(args.land_class), dtype=np.int16)
    ocean = np.isfinite(arr) & (arr < float(args.ocean_threshold))
    cls[ocean] = int(args.ocean_class)
    ice = np.abs(lat2d) >= float(args.ice_lat_deg)
    cls[ice] = int(args.ice_class)

    h = fits.Header()
    h["CLASSDEF"] = ("0=ocean,1=land,2=ice", "Default class meaning")
    h["OCTHRES"] = (float(args.ocean_threshold), "Ocean albedo threshold")
    h["ICELAT"] = (float(args.ice_lat_deg), "Ice |lat| threshold deg")
    fits.PrimaryHDU(data=cls, header=h).writeto(args.out_fits, overwrite=True, output_verify="silentfix")
    print(f"Wrote class map: {args.out_fits}  shape={cls.shape} dtype={cls.dtype}")


if __name__ == "__main__":
    main()
