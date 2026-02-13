from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from astropy.io import fits


BASE = "http://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG"


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"OK (exists): {dest}")
        return
    ensure_parent(dest)
    print(f"Downloading:\n  {url}\n-> {dest}")
    urlretrieve(url, dest)
    print(f"  {dest.stat().st_size/1e6:6.1f} MB")


def parse_pds_label(lbl_path: Path) -> dict:
    """
    Very small PDS3 label parser (KEY = VALUE). We only need a handful of keys.
    """
    out: dict = {}
    txt = lbl_path.read_text(errors="ignore").splitlines()
    for line in txt:
        line = line.strip()
        if not line or line.startswith("/*"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip()
        # Drop trailing comments
        if "/*" in v:
            v = v.split("/*", 1)[0].strip()
        out[k] = v
    return out


def choose_endianness(raw: bytes, n: int) -> np.dtype:
    """
    Auto-detect int16 endianness by looking for a plausible elevation range.
    """
    a_be = np.frombuffer(raw, dtype=">i2", count=n)
    a_le = np.frombuffer(raw, dtype="<i2", count=n)

    def score(a: np.ndarray) -> float:
        # prefer data with a reasonable spread and magnitude in kilometres
        # (elevations are typically within +-20 km, i.e. +-20000 m)
        p1, p99 = np.percentile(a.astype(float), [1, 99])
        return abs(p99 - p1) + 0.001 * (100000.0 - min(100000.0, max(abs(p1), abs(p99))))

    return (">i2" if score(a_be) >= score(a_le) else "<i2")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Download LOLA LDEM_* cylindrical DEM and convert to FITS.")
    ap.add_argument("--ppd", type=int, default=16, choices=[4, 16], help="Pixels per degree (4 small/fast, 16 detailed).")
    ap.add_argument("--out", default=None, help="Output FITS path (default: DATA/moon_dem_lola_ldem{ppd}_m_int16.fits)")
    args = ap.parse_args(argv)

    ppd = int(args.ppd)
    img = Path(f"DATA/LOLA/LDEM_{ppd}.IMG")
    lbl = Path(f"DATA/LOLA/LDEM_{ppd}.LBL")

    download(f"{BASE}/LDEM_{ppd}.IMG", img)
    download(f"{BASE}/LDEM_{ppd}.LBL", lbl)

    meta = parse_pds_label(lbl)
    # Known dimensions for LDEM_4 and LDEM_16:
    #  - LDEM_4 : 720 x 1440
    #  - LDEM_16: 2880 x 5760
    # Still, try to read from label first.
    nlon = int(float(meta.get("LINE_SAMPLES", meta.get("SAMPLES", "0"))) or 0)
    nlat = int(float(meta.get("LINES", "0")) or 0)
    if nlon <= 0 or nlat <= 0:
        if ppd == 4:
            nlat, nlon = 720, 1440
        else:
            nlat, nlon = 2880, 5760

    raw = img.read_bytes()
    n = nlat * nlon
    dt = choose_endianness(raw, n)
    arr = np.frombuffer(raw, dtype=dt, count=n).reshape((nlat, nlon))

    # Scaling if present
    scale = float(meta.get("SCALING_FACTOR", "1.0"))
    offset = float(meta.get("OFFSET", "0.0"))
    if not (abs(scale - 1.0) < 1e-12 and abs(offset) < 1e-12):
        arr_f = arr.astype(np.float32) * scale + offset
        # Store as float32 if scaling is needed
        out_data = arr_f.astype(np.float32)
        out_dtype = "float32"
    else:
        # Store as int16 metres (most common)
        out_data = arr.astype(np.int16)
        out_dtype = "int16"

    out = Path(args.out) if args.out else Path(f"DATA/moon_dem_lola_ldem{ppd}_m_{out_dtype}.fits")
    out.parent.mkdir(parents=True, exist_ok=True)

    hdr = fits.Header()
    hdr["SRC"] = ("LOLA", "LRO LOLA (PDS)")
    hdr["PPD"] = (ppd, "Pixels per degree")
    hdr["LONMODE"] = ("0_360", "Lon convention for cylindrical IMG products")
    hdr["BUNIT"] = ("m", "DEM units (metres)")
    hdr["DTYPE"] = (out_dtype, "Stored dtype")

    # Write
    fits.PrimaryHDU(data=out_data, header=hdr).writeto(out, overwrite=True, output_verify="silentfix")

    # Report
    x = out_data.astype(np.float64)
    print(f"Wrote: {out}")
    print(f"  dtype={x.dtype}  min={float(np.min(x)):.3f}  max={float(np.max(x)):.3f}  mean={float(np.mean(x)):.3f}")


if __name__ == "__main__":
    main()
