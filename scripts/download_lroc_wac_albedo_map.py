from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np
from astropy.io import fits

# Pillow is used only to read GeoTIFF without pulling in heavy geospatial stacks.
# Installed via: uv sync (see pyproject.toml).
try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    print("ERROR: Pillow not available. Run: uv sync", file=sys.stderr)
    raise

# LROC WAC Global Morphologic Map (643 nm) - global simple cylindrical tile at 16 px/deg (~15.8 MB GeoTIFF).
# Product page: https://data.lroc.im-ldi.com/lroc/view_rdr_product/WAC_GLOBAL_E000N0000_016P
URL_TIF = (
    "https://pds.lroc.im-ldi.com/data/"
    "LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/EXTRAS/BROWSE/WAC_GLOBAL/"
    "WAC_GLOBAL_E000N0000_016P.TIF"
)

def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def download(url: str, dest: Path) -> None:
    """
    Simple streaming download with a conservative User-Agent.
    (Some servers reject default python-urllib UA strings.)
    """
    if dest.exists() and dest.stat().st_size > 0:
        print(f"OK (exists): {dest}")
        return

    ensure_parent(dest)
    req = Request(url, headers={"User-Agent": "synthmoon/0.1.0 (python-urllib)"})
    print(f"Downloading:\n  {url}\n-> {dest}")
    with urlopen(req) as r, open(dest, "wb") as f:
        total = r.headers.get("Content-Length")
        total = int(total) if total else None
        done = 0
        chunk = 1024 * 1024
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)
            done += len(b)
            if total:
                pct = 100.0 * done / total
                print(f"\r  {done/1e6:7.1f} MB / {total/1e6:7.1f} MB  ({pct:5.1f}%)", end="")
            else:
                print(f"\r  {done/1e6:7.1f} MB", end="")
        print("")

def tif_to_albedo_fits(tif_path: Path, fits_path: Path, target_mean: float, clip_max: float) -> None:
    """
    Convert the GeoTIFF mosaic to a Float32 equirectangular FITS albedo map.

    IMPORTANT PHYSICS NOTE (v0.1):
    - This LROC product is a photometrically corrected 643 nm basemap.
    - We use it as a *proxy* for spatial variations in single-scattering albedo (SSA).
      In Lambert mode, it behaves like a Lambert albedo map.
    """
    ensure_parent(fits_path)

    img = Image.open(tif_path)
    # Force single channel (most likely already 8-bit grayscale)
    img = img.convert("F")  # 32-bit float pixels in Pillow
    a = np.array(img, dtype=np.float32)

    # Normalise to 0..1 using the max present (usually 255 for 8-bit products).
    amax = float(np.nanmax(a))
    if not np.isfinite(amax) or amax <= 0:
        raise RuntimeError(f"Unexpected max value from TIFF: {amax}")

    a01 = a / amax

    # Scale to target mean (ignore NaNs; also ignore exact zeros if they dominate).
    m = np.isfinite(a01)
    nz = m & (a01 > 0)
    mean_val = float(np.mean(a01[nz])) if np.any(nz) else float(np.mean(a01[m]))
    if not np.isfinite(mean_val) or mean_val <= 0:
        raise RuntimeError(f"Unexpected mean from TIFF after normalisation: {mean_val}")

    scale = float(target_mean / mean_val)
    out = a01 * scale

    # Clip physically to [0, clip_max] (clip_max default 1.0).
    out = np.clip(out, 0.0, float(clip_max)).astype(np.float32)

    hdu = fits.PrimaryHDU(out)
    hdr = hdu.header

    # Map metadata (tile is -180..180 lon, -90..90 lat at 16 px/deg).
    hdr["MAPTYPE"] = ("EQUIRECT", "Assumed equirectangular grid")
    hdr["LONMODE"] = ("-180_180", "Longitude domain of this map")
    hdr["LON_MIN"] = (-180.0, "deg")
    hdr["LON_MAX"] = (180.0, "deg")
    hdr["LAT_MIN"] = (-90.0, "deg")
    hdr["LAT_MAX"] = (90.0, "deg")
    hdr["PPD"] = (16.0, "pixels per degree (nominal)")
    hdr["BANDNM"] = (643.0, "WAC band, nm")
    hdr["PRODID"] = ("WAC_GLOBAL_E000N0000_016P", "LROC product id")
    hdr["SOURCE"] = ("LROC WAC Global Morphologic Map", "LRO/LROC basemap")
    hdr["URL"] = (URL_TIF[:68], "Source URL (truncated)")  # keep short to avoid FITS card warnings
    hdr["SYNVER"] = ("0.1.0", "synthmoon code version")
    hdr["TMEAN"] = (float(target_mean), "Target mean for this map")
    hdr["SCALE"] = (float(scale), "Scale factor applied after 0..1 normalisation")
    hdr["CLIPMAX"] = (float(clip_max), "Maximum allowed albedo after scaling")

    fits.writeto(fits_path, out, header=hdr, overwrite=True)
    print(f"Wrote: {fits_path}")
    print(f"  dtype={out.dtype}  min={float(np.min(out)):.6f}  max={float(np.max(out)):.6f}  mean={float(np.mean(out)):.6f}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Download + convert LROC WAC global mosaic to a lunar albedo map (FITS).")
    ap.add_argument("--dest-tif", type=Path, default=Path("DATA/LROC/WAC_GLOBAL_E000N0000_016P.TIF"))
    ap.add_argument("--out-fits", type=Path, default=Path("DATA/moon_albedo_lroc_wac643_016ppd_mean0p12.fits"))
    ap.add_argument("--target-mean", type=float, default=0.12, help="Scale map so mean≈this (dimensionless).")
    ap.add_argument("--clip-max", type=float, default=1.0, help="Clip output albedo to [0, clip-max].")
    ap.add_argument("--no-download", action="store_true", help="Skip download; assume --dest-tif already exists.")
    args = ap.parse_args()

    if not args.no_download:
        download(URL_TIF, args.dest_tif)
    if not args.dest_tif.exists():
        raise SystemExit(f"Missing TIFF: {args.dest_tif}")

    tif_to_albedo_fits(args.dest_tif, args.out_fits, args.target_mean, args.clip_max)

if __name__ == "__main__":
    main()
