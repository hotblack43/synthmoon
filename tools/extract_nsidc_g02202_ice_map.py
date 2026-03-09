#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _subdataset_ref(nc_path: Path) -> str:
    return f'NETCDF:"{nc_path}":cdr_seaice_conc'


def _warp_to_fits(nc_path: Path, out_fits: Path, nlon: int, nlat: int, resample: str) -> None:
    with tempfile.TemporaryDirectory(prefix="nsidc_ice_") as td:
        td = Path(td)
        unscaled_tif = td / "unscaled.tif"
        warped_tif = td / "warped_ll.tif"

        _run(
            [
                "gdal_translate",
                "-unscale",
                "-ot",
                "Float32",
                _subdataset_ref(nc_path),
                str(unscaled_tif),
            ]
        )
        _run(
            [
                "gdalwarp",
                "-overwrite",
                "-t_srs",
                "EPSG:4326",
                "-te",
                "-180",
                "-90",
                "180",
                "90",
                "-ts",
                str(nlon),
                str(nlat),
                "-r",
                str(resample),
                "-dstnodata",
                "nan",
                str(unscaled_tif),
                str(warped_tif),
            ]
        )
        _run(["gdal_translate", "-of", "FITS", str(warped_tif), str(out_fits)])


def _load_clean(path: Path) -> np.ndarray:
    x = fits.getdata(path).astype(np.float32)
    x = np.flipud(x)
    x[(~np.isfinite(x)) | (x < 0.0) | (x > 1.0)] = np.nan
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract a global ice-fraction lon/lat FITS from NSIDC G02202 north/south daily files.")
    ap.add_argument("--north-nc", required=True, help="North daily NetCDF, e.g. sic_psn25_20110706_F17_v05r00.nc")
    ap.add_argument("--south-nc", required=True, help="South daily NetCDF, e.g. sic_pss25_20110706_F17_v05r00.nc")
    ap.add_argument("--out-fits", required=True, help="Output 2D FITS file (global lon/lat)")
    ap.add_argument("--nlon", type=int, default=1440, help="Output longitude samples")
    ap.add_argument("--nlat", type=int, default=720, help="Output latitude samples")
    ap.add_argument("--resample", default="bilinear", choices=("near", "bilinear", "cubic"), help="GDAL warp resampling")
    args = ap.parse_args()

    north_nc = Path(args.north_nc)
    south_nc = Path(args.south_nc)
    out_fits = Path(args.out_fits)
    out_fits.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nsidc_ice_map_") as td:
        td = Path(td)
        north_fits = td / "north_ll.fits"
        south_fits = td / "south_ll.fits"
        _warp_to_fits(north_nc, north_fits, args.nlon, args.nlat, args.resample)
        _warp_to_fits(south_nc, south_fits, args.nlon, args.nlat, args.resample)

        north = _load_clean(north_fits)
        south = _load_clean(south_fits)

    lat_centers = 90.0 - (np.arange(args.nlat, dtype=np.float32) + 0.5) * (180.0 / float(args.nlat))
    north_rows = lat_centers >= 0.0
    south_rows = ~north_rows

    out = np.zeros((args.nlat, args.nlon), dtype=np.float32)
    if np.any(north_rows):
        x = north[north_rows]
        out[north_rows] = np.where(np.isfinite(x), x, 0.0)
    if np.any(south_rows):
        x = south[south_rows]
        out[south_rows] = np.where(np.isfinite(x), x, 0.0)

    hdr = fits.Header()
    hdr["BUNIT"] = ("1", "Sea ice area fraction")
    hdr["LONMODE"] = ("-180_180", "Longitude convention for equirectangular map")
    hdr["NSRCN"] = (north_nc.name, "NSIDC north daily source")
    hdr["NSRCS"] = (south_nc.name, "NSIDC south daily source")
    hdr["RESAMP"] = (str(args.resample), "GDAL warp resampling")
    hdr["COMMENT"] = "Rows run north-to-south; columns run -180..180 eastward."
    fits.PrimaryHDU(data=out.astype(np.float32), header=hdr).writeto(out_fits, overwrite=True, output_verify="silentfix")
    print(f"saved: {out_fits}")


if __name__ == "__main__":
    main()
