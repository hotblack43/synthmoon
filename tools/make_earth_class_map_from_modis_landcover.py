#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits


LC_CANDIDATES = (
    "Majority_Land_Cover_Type_1",
    "Land_Cover_Type_1",
    "LC_Type1",
)


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return p.stdout


def _subdatasets(hdf_path: Path) -> dict[str, str]:
    txt = _run(["gdalinfo", str(hdf_path)])
    out: dict[str, str] = {}
    pat = re.compile(r"^SUBDATASET_\d+_NAME=(.+)$")
    for line in txt.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        sds = m.group(1)
        name = sds.rsplit(":", 1)[-1]
        out[name] = sds
    return out


def _pick_sds(sds_map: dict[str, str], user_name: str | None) -> tuple[str, str]:
    if user_name:
        if user_name not in sds_map:
            raise SystemExit(f"Requested SDS '{user_name}' not found. Available: {sorted(sds_map)}")
        return user_name, sds_map[user_name]
    for c in LC_CANDIDATES:
        if c in sds_map:
            return c, sds_map[c]
    raise SystemExit(f"Could not find a land-cover SDS. Tried: {', '.join(LC_CANDIDATES)}")


def _source_info(sds_ref: str) -> tuple[int, int, bool]:
    txt = _run(["gdalinfo", sds_ref])
    m = re.search(r"Size is (\d+), (\d+)", txt)
    if not m:
        raise SystemExit("Could not determine source raster size from gdalinfo.")
    nx = int(m.group(1))
    ny = int(m.group(2))
    is_global_ll = "Origin = (-180.000000000000000,90.000000000000000)" in txt
    return nx, ny, is_global_ll


def _translate_or_warp_to_fits(sds_ref: str, out_path: Path, nlon: int | None, nlat: int | None, resample: str) -> None:
    with tempfile.TemporaryDirectory(prefix="modis_classmap_") as td:
        td = Path(td)
        tmp_tif = td / "landcover.tif"
        src_nx, src_ny, is_global_ll = _source_info(sds_ref)
        subprocess.run(["gdal_translate", "-ot", "Float32", sds_ref, str(tmp_tif)], check=True)
        if nlon is None:
            nlon = src_nx
        if nlat is None:
            nlat = src_ny
        if is_global_ll and nlon == src_nx and nlat == src_ny:
            subprocess.run(["gdal_translate", "-of", "FITS", str(tmp_tif), str(out_path)], check=True)
            return
        tmp_ll = td / "landcover_ll.tif"
        subprocess.run(
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
                str(tmp_tif),
                str(tmp_ll),
            ],
            check=True,
        )
        subprocess.run(["gdal_translate", "-of", "FITS", str(tmp_ll), str(out_path)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an EO-derived Earth class-map FITS from MODIS land cover.")
    ap.add_argument("--in-hdf", required=True, help="Input MCD12C1/MCD12Q1 HDF file (glob accepted if unique)")
    ap.add_argument("--out-fits", required=True, help="Output class-map FITS (2D)")
    ap.add_argument("--sds", default=None, help="Optional exact SDS name; otherwise a known candidate is used")
    ap.add_argument("--water-class-value", type=float, default=0.0, help="MODIS LC_Type1 value treated as water/ocean")
    ap.add_argument("--snow-ice-class-value", type=float, default=15.0, help="MODIS LC_Type1 value treated as permanent snow/ice")
    ap.add_argument("--ocean-class", type=int, default=0, help="Output class value for ocean")
    ap.add_argument("--land-class", type=int, default=1, help="Output class value for land")
    ap.add_argument("--ice-class", type=int, default=2, help="Output class value for ice")
    ap.add_argument("--nlon", type=int, default=None, help="Output longitude samples; default keeps source resolution when possible")
    ap.add_argument("--nlat", type=int, default=None, help="Output latitude samples; default keeps source resolution when possible")
    ap.add_argument("--resample", default="near", choices=("near", "bilinear", "cubic"), help="GDAL warp resampling")
    args = ap.parse_args()

    matches = sorted(glob.glob(args.in_hdf))
    if matches:
        if len(matches) > 1:
            raise SystemExit(f"--in-hdf matched multiple files; be specific or pass one file only: {matches}")
        in_hdf = Path(matches[0])
    else:
        in_hdf = Path(args.in_hdf)

    sds_map = _subdatasets(in_hdf)
    if not sds_map:
        raise SystemExit(f"No GDAL subdatasets found in {in_hdf}")
    sds_name, sds_ref = _pick_sds(sds_map, args.sds)

    tmp_out = Path(args.out_fits).with_suffix(".tmp.fits")
    _translate_or_warp_to_fits(sds_ref, tmp_out, args.nlon, args.nlat, args.resample)

    lc = np.flipud(fits.getdata(tmp_out).astype(np.float32))
    tmp_out.unlink(missing_ok=True)
    lc[~np.isfinite(lc)] = np.nan

    cls = np.full(lc.shape, int(args.land_class), dtype=np.int16)
    water = np.abs(lc - float(args.water_class_value)) <= 0.5
    ice = np.abs(lc - float(args.snow_ice_class_value)) <= 0.5
    cls[water] = int(args.ocean_class)
    cls[ice] = int(args.ice_class)

    hdr = fits.Header()
    hdr["CLASSDEF"] = (f"{int(args.ocean_class)}=ocean,{int(args.land_class)}=land,{int(args.ice_class)}=ice", "Output class definitions")
    hdr["LCFILE"] = (in_hdf.name, "MODIS land-cover source")
    hdr["LCSDS"] = (sds_name[:68], "Source SDS")
    hdr["LCWATER"] = (float(args.water_class_value), "MODIS class treated as ocean")
    hdr["LCICE"] = (float(args.snow_ice_class_value), "MODIS class treated as permanent ice")
    hdr["LONMODE"] = ("-180_180", "Longitude convention for equirectangular map")
    hdr["COMMENT"] = "Rows run north-to-south; columns run -180..180 eastward."
    Path(args.out_fits).parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=cls, header=hdr).writeto(args.out_fits, overwrite=True, output_verify="silentfix")
    print(f"{sds_name} -> {args.out_fits}")


if __name__ == "__main__":
    main()
