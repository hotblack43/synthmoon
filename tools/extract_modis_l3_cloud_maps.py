#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
import glob

import numpy as np
from astropy.io import fits


CF_CANDIDATES = (
    "Cloud_Fraction_Mean",
    "Cloud_Fraction_Day_Mean",
    "Cloud_Fraction_Night_Mean",
)

TAU_CANDIDATES = (
    "Cloud_Optical_Thickness_Combined_Mean",
    "Cloud_Optical_Thickness_Mean",
    "Cloud_Optical_Thickness_Liquid_Mean",
    "Cloud_Optical_Thickness_Ice_Mean",
)


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return p.stdout


def _subdatasets(hdf_path: Path) -> dict[str, str]:
    txt = _run(["gdalinfo", str(hdf_path)])
    out: dict[str, str] = {}
    pat = re.compile(r'^SUBDATASET_\d+_NAME=(.+)$')
    for line in txt.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        sds = m.group(1)
        name = sds.rsplit(":", 1)[-1]
        out[name] = sds
    return out


def _pick_sds(sds_map: dict[str, str], candidates: tuple[str, ...]) -> tuple[str, str] | None:
    for c in candidates:
        if c in sds_map:
            return c, sds_map[c]
    return None


def _convert_sds_to_fits(sds_name: str, sds_ref: str, out_path: Path) -> None:
    tmp_tif = out_path.with_suffix(".tmp.tif")
    subprocess.run(
        ["gdal_translate", "-unscale", "-ot", "Float32", sds_ref, str(tmp_tif)],
        check=True,
    )
    subprocess.run(
        ["gdal_translate", "-of", "FITS", str(tmp_tif), str(out_path)],
        check=True,
    )
    tmp_tif.unlink(missing_ok=True)
    print(f"{sds_name} -> {out_path}")


def _fill_missing_by_row_then_nearest(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    out = np.array(x, copy=True)
    ny, nx = out.shape

    row_medians = np.full(ny, np.nan, dtype=np.float32)
    for j in range(ny):
        row = out[j]
        m = np.isfinite(row)
        if np.any(m):
            row_medians[j] = np.float32(np.nanmedian(row[m]))

    valid_rows = np.where(np.isfinite(row_medians))[0]
    global_med = np.float32(np.nanmedian(out[np.isfinite(out)])) if np.any(np.isfinite(out)) else np.float32(0.0)

    for j in range(ny):
        row = out[j]
        miss = ~np.isfinite(row)
        if not np.any(miss):
            continue
        if np.isfinite(row_medians[j]):
            row_fill = row_medians[j]
        elif valid_rows.size:
            k = int(valid_rows[np.argmin(np.abs(valid_rows - j))])
            row_fill = row_medians[k]
        else:
            row_fill = global_med
        row[miss] = row_fill
        out[j] = row

    miss_all = ~np.isfinite(out)
    if np.any(miss_all):
        out[miss_all] = global_med
    return out


def _clean_cloud_map(path: Path, kind: str) -> None:
    data = fits.getdata(path).astype(np.float32)
    hdr = fits.getheader(path)

    x = np.array(data, copy=True)
    if kind == "cloud_fraction":
        x[(~np.isfinite(x)) | (x < 0.0) | (x > 1.0)] = np.nan
        x = _fill_missing_by_row_then_nearest(x)
        x = np.clip(x, 0.0, 1.0)
    elif kind == "cloud_tau":
        x[(~np.isfinite(x)) | (x < 0.0) | (x > 1.0e6)] = np.nan
        x = _fill_missing_by_row_then_nearest(x)
        x = np.clip(x, 0.0, None)
    else:
        raise ValueError(f"unknown cloud-map kind: {kind}")

    hdr["GAPFILL"] = ("ROWMED", "NaN fill: row median + nearest/global fallback")
    hdr["DATACLN"] = (1, "Invalid sentinels replaced and gaps filled")
    fits.PrimaryHDU(data=x.astype(np.float32), header=hdr).writeto(path, overwrite=True, output_verify="silentfix")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract MODIS Level-3 daily cloud maps to FITS.")
    ap.add_argument("--in-hdf", required=True, help="Input MOD08_D3/MYD08_D3 HDF file")
    ap.add_argument("--out-dir", default="DATA/MODIS", help="Output directory for FITS maps")
    ap.add_argument("--prefix", default=None, help="Optional filename prefix; defaults to HDF stem")
    args = ap.parse_args()

    matches = sorted(glob.glob(args.in_hdf))
    if matches:
        if len(matches) > 1:
            raise SystemExit(
                f"--in-hdf matched multiple files; be specific or pass one file only: {matches}"
            )
        in_hdf = Path(matches[0])
    else:
        in_hdf = Path(args.in_hdf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or in_hdf.stem.lower()

    sds_map = _subdatasets(in_hdf)
    if not sds_map:
        raise SystemExit(f"No GDAL subdatasets found in {in_hdf}")

    cf = _pick_sds(sds_map, CF_CANDIDATES)
    if cf is None:
        raise SystemExit(
            "Could not find a cloud-fraction SDS. "
            f"Tried: {', '.join(CF_CANDIDATES)}"
        )
    cf_name, cf_ref = cf
    cf_out = out_dir / f"{prefix}_cloud_fraction.fits"
    _convert_sds_to_fits(cf_name, cf_ref, cf_out)
    _clean_cloud_map(cf_out, "cloud_fraction")

    tau = _pick_sds(sds_map, TAU_CANDIDATES)
    if tau is not None:
        tau_name, tau_ref = tau
        tau_out = out_dir / f"{prefix}_cloud_tau.fits"
        _convert_sds_to_fits(tau_name, tau_ref, tau_out)
        _clean_cloud_map(tau_out, "cloud_tau")
    else:
        print("No matching cloud optical thickness SDS found; skipping tau.")


if __name__ == "__main__":
    main()
