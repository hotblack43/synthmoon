from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
from astropy.io import fits

BASE = "https://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG"
URL_IMG = f"{BASE}/LDEM_16.IMG"
URL_LBL = f"{BASE}/LDEM_16.LBL"

OUT_FITS = Path("DATA/moon_dem_lola_ldem16_m.fits")
CACHE_DIR = Path("DATA/LOLA_GDR/LDEM_16")
IMG_PATH = CACHE_DIR / "LDEM_16.IMG"
LBL_PATH = CACHE_DIR / "LDEM_16.LBL"

def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def download(url: str, dest: Path) -> None:
    ensure_parent(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"OK (exists): {dest}")
        return
    print(f"Downloading:\n  {url}\n-> {dest}")
    urlretrieve(url, dest)

def _parse_value(v: str):
    v = v.strip().strip('"').strip("'")
    # try int, float, else string
    try:
        if "." in v or "E" in v.upper():
            return float(v)
        return int(v)
    except Exception:
        return v

def parse_pds3_image_object(lbl_text: str) -> dict:
    """
    Parse a minimal subset of a PDS3 label sufficient to read the IMAGE object.
    We intentionally avoid heavy deps; this is robust enough for LOLA GDR labels.
    """
    lines = lbl_text.splitlines()
    in_image = False
    d = {}

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("/*"):
            continue
        if line.upper().startswith("OBJECT") and "IMAGE" in line.upper():
            in_image = True
            continue
        if line.upper().startswith("END_OBJECT") and "IMAGE" in line.upper():
            in_image = False
            continue
        if not in_image:
            continue
        if "=" not in line:
            continue

        k, v = line.split("=", 1)
        k = k.strip().upper()
        v = v.strip()

        # strip trailing units like "<meters>" if present
        if "<" in v and ">" in v:
            v = v.split("<", 1)[0].strip()

        d[k] = _parse_value(v)

    # also try to find scaling/offset even if they sit outside IMAGE object
    for raw in lines:
        line = raw.strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().upper()
        v = v.strip()
        if k in ("SCALING_FACTOR", "SCALE", "MULTIPLIER"):
            d.setdefault("SCALING_FACTOR", _parse_value(v))
        if k in ("OFFSET", "ADD_OFFSET", "INTERCEPT"):
            d.setdefault("OFFSET", _parse_value(v))

    return d

def dtype_from_sample(sample_type: str, sample_bits: int) -> np.dtype:
    st = sample_type.upper()
    bits = int(sample_bits)

    # signed/unsigned integer
    if "INTEGER" in st:
        if "UNSIGNED" in st:
            base = {16: "u2", 32: "u4"}.get(bits)
        else:
            base = {16: "i2", 32: "i4"}.get(bits)
        if base is None:
            raise ValueError(f"Unsupported integer bits: {bits}")
        endian = ">" if "MSB" in st else "<"
        return np.dtype(endian + base)

    # floats
    if "REAL" in st or "FLOAT" in st:
        base = {32: "f4", 64: "f8"}.get(bits)
        if base is None:
            raise ValueError(f"Unsupported float bits: {bits}")
        endian = ">" if "MSB" in st else "<"
        return np.dtype(endian + base)

    raise ValueError(f"Unsupported SAMPLE_TYPE: {sample_type}")

def main() -> None:
    download(URL_LBL, LBL_PATH)
    download(URL_IMG, IMG_PATH)

    lbl = LBL_PATH.read_text(errors="replace")
    meta = parse_pds3_image_object(lbl)

    lines = int(meta.get("LINES"))
    samps = int(meta.get("LINE_SAMPLES"))
    sample_bits = int(meta.get("SAMPLE_BITS", 16))
    sample_type = str(meta.get("SAMPLE_TYPE", "MSB_INTEGER"))
    scale = float(meta.get("SCALING_FACTOR", 1.0))
    offset = float(meta.get("OFFSET", 0.0))

    dt = dtype_from_sample(sample_type, sample_bits)

    print("Parsed label:")
    print("  LINES       =", lines)
    print("  LINE_SAMPLES=", samps)
    print("  SAMPLE_TYPE =", sample_type)
    print("  SAMPLE_BITS =", sample_bits)
    print("  dtype       =", dt)
    print("  scale       =", scale)
    print("  offset      =", offset)

    data = np.fromfile(IMG_PATH, dtype=dt)
    expected = lines * samps
    if data.size != expected:
        raise RuntimeError(f"Size mismatch: got {data.size}, expected {expected} = {lines}*{samps}")

    data = data.reshape((lines, samps)).astype(np.float32)
    data = data * scale + offset  # metres

    ensure_parent(OUT_FITS)
    h = fits.Header()
    h["BUNIT"] = "m"
    h["LONMODE"] = "0_360"
    h["LATMIN"] = -90.0
    h["LATMAX"] = 90.0
    h["LONMIN"] = 0.0
    h["LONMAX"] = 360.0
    h["PPD"] = 16
    h["SRC"] = "LOLA_GDR_LDEM_16"
    h["URLIMG"] = URL_IMG
    h["URLLBL"] = URL_LBL

    fits.PrimaryHDU(data=data, header=h).writeto(OUT_FITS, overwrite=True)
    print(f"Wrote: {OUT_FITS}")
    print("  dtype=", data.dtype, "min=", float(np.nanmin(data)), "max=", float(np.nanmax(data)))

if __name__ == "__main__":
    main()
