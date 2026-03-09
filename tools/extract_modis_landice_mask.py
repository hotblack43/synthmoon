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

PERCENT_CANDIDATES = (
    "Land_Cover_Type_1_Percent",
    "LC_Type1_Percent",
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


def _pick_sds(sds_map: dict[str, str], user_name: str | None, mode: str) -> tuple[str, str]:
    if user_name:
        if user_name not in sds_map:
            raise SystemExit(f"Requested SDS '{user_name}' not found. Available: {sorted(sds_map)}")
        return user_name, sds_map[user_name]
    candidates = PERCENT_CANDIDATES if mode == "percent" else LC_CANDIDATES
    for c in candidates:
        if c in sds_map:
            return c, sds_map[c]
    raise SystemExit(f"Could not find a land-cover SDS. Tried: {', '.join(candidates)}")


def _source_info(sds_ref: str) -> tuple[int, int, bool]:
    txt = _run(["gdalinfo", sds_ref])
    m_size = re.search(r"Size is (\d+), (\d+)", txt)
    if not m_size:
        raise SystemExit("Could not determine source raster size from gdalinfo.")
    nx = int(m_size.group(1))
    ny = int(m_size.group(2))
    is_global_ll = ("Origin = (-180.000000000000000,90.000000000000000)" in txt and 'Pixel Size = (0.050000000000000,-0.050000000000000)' in txt) or "GEOGCRS" in txt
    return nx, ny, is_global_ll


def _translate_or_warp_sds_to_fits(sds_ref: str, out_path: Path, nlon: int | None, nlat: int | None, resample: str, band: int | None) -> None:
    with tempfile.TemporaryDirectory(prefix="modis_landice_") as td:
        td = Path(td)
        tmp_tif = td / "landcover.tif"
        src_nx, src_ny, is_global_ll = _source_info(sds_ref)
        cmd = ["gdal_translate", "-ot", "Float32"]
        if band is not None:
            cmd += ["-b", str(int(band))]
        cmd += [sds_ref, str(tmp_tif)]
        subprocess.run(cmd, check=True)
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
    ap = argparse.ArgumentParser(description="Extract a global static land-ice mask from MODIS MCD12C1/MCD12Q1.")
    ap.add_argument("--in-hdf", required=True, help="Input MCD12C1/MCD12Q1 HDF file (glob accepted if unique)")
    ap.add_argument("--out-fits", required=True, help="Output FITS land-ice mask (0..1)")
    ap.add_argument("--sds", default=None, help="Optional exact SDS name; otherwise a known candidate is used")
    ap.add_argument("--mode", choices=("majority", "percent"), default="percent", help="Use hard majority class or class-percent layer")
    ap.add_argument("--snow-ice-class", type=float, default=15.0, help="Class value for permanent snow/ice in LC_Type1/IGBP")
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
    sds_name, sds_ref = _pick_sds(sds_map, args.sds, str(args.mode))

    band = None
    if str(args.mode) == "percent":
        band = int(round(float(args.snow_ice_class))) + 1

    tmp_out = Path(args.out_fits).with_suffix(".tmp.fits")
    _translate_or_warp_sds_to_fits(sds_ref, tmp_out, args.nlon, args.nlat, args.resample, band)

    lc = np.flipud(fits.getdata(tmp_out).astype(np.float32))
    tmp_out.unlink(missing_ok=True)
    lc[~np.isfinite(lc)] = np.nan
    if str(args.mode) == "percent":
        mask = np.clip(lc / 100.0, 0.0, 1.0).astype(np.float32)
    else:
        mask = np.where(np.abs(lc - float(args.snow_ice_class)) <= 0.5, 1.0, 0.0).astype(np.float32)

    hdr = fits.Header()
    hdr["BUNIT"] = ("1", "Static land ice mask")
    hdr["LONMODE"] = ("-180_180", "Longitude convention for equirectangular map")
    hdr["LCFILE"] = (in_hdf.name, "MODIS land-cover source")
    hdr["LCSDS"] = (sds_name[:68], "Source SDS")
    hdr["LCMODE"] = (str(args.mode), "Land-ice extract mode")
    hdr["LCIVAL"] = (float(args.snow_ice_class), "Class treated as permanent land ice")
    hdr["COMMENT"] = "Rows run north-to-south; columns run -180..180 eastward."
    Path(args.out_fits).parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=mask, header=hdr).writeto(args.out_fits, overwrite=True, output_verify="silentfix")
    print(f"{sds_name} -> {args.out_fits}")


if __name__ == "__main__":
    main()
