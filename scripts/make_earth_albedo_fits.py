#!/usr/bin/env python3
"""
cd ~/WORKSHOP/SYNTHMOON
PYTHONPATH=$PWD uv run python scripts/make_earth_albedo_fits.py \
  --out DATA/earth_albedo.fits \
  --nlon 1440 --nlat 720 \
  --lon-mode 0_360 \
  --land 0.20 --ocean 0.06

"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from global_land_mask import globe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("DATA/earth_albedo.fits"))
    ap.add_argument("--nlon", type=int, default=1440, help="Number of longitude pixels (e.g. 1440 = 0.25 deg)")
    ap.add_argument("--nlat", type=int, default=720, help="Number of latitude pixels (e.g. 720 = 0.25 deg)")
    ap.add_argument("--lon-mode", choices=["0_360", "-180_180"], default="0_360")
    ap.add_argument("--land", type=float, default=0.20)
    ap.add_argument("--ocean", type=float, default=0.06)
    args = ap.parse_args()

    nlon, nlat = args.nlon, args.nlat

    # Lat grid: row0 = +90, last = -90
    lat = np.linspace(90.0, -90.0, nlat, dtype=np.float64)

    # Lon grid: either 0..360 or -180..180
    if args.lon_mode == "0_360":
        lon = np.linspace(0.0, 360.0, nlon, endpoint=False, dtype=np.float64)
        lon_for_mask = ((lon + 180.0) % 360.0) - 180.0  # convert to -180..180 for globe.is_land
    else:
        lon = np.linspace(-180.0, 180.0, nlon, endpoint=False, dtype=np.float64)
        lon_for_mask = lon

    # Build mesh (lat, lon) with shape (nlat,nlon)
    LON, LAT = np.meshgrid(lon_for_mask, lat)

    landmask = globe.is_land(LAT, LON)  # bool, True on land

    alb = np.where(landmask, float(args.land), float(args.ocean)).astype(np.float32)
    alb = np.clip(alb, 0.0, 1.0)

    hdu = fits.PrimaryHDU(alb)
    hdr = hdu.header
    hdr["EXTNAME"] = "EARTHALB"
    hdr["BUNIT"] = "ALBEDO"
    hdr["LONMODE"] = args.lon_mode
    hdr["LATORD"] = "N2S"   # row 0 north -> south
    hdr["Nlon"] = nlon
    hdr["Nlat"] = nlat
    hdr["ALBLAND"] = float(args.land)
    hdr["ALBOCEAN"] = float(args.ocean)
    hdr["COMMENT"] = "Equirectangular Earth albedo map. Row0=+90deg, last=-90deg."
    hdr["COMMENT"] = "Longitudes increase eastward across columns."
    hdr["COMMENT"] = "Values are float32 in 0..1."

    args.out.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(args.out, overwrite=True)

    print(f"Wrote: {args.out}")
    print(f"  shape={alb.shape} dtype={alb.dtype} min={alb.min():.6f} max={alb.max():.6f} mean={alb.mean():.6f}")


if __name__ == "__main__":
    main()

