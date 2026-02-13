from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class ScaleResult:
    scaled: np.ndarray
    raw_min: float
    raw_max_clipped: float
    scale_factor: float


def scale_to_0_65535_float(img: np.ndarray) -> ScaleResult:
    """
    Scale a float image into the range [0, 65535] while staying floating-point.

    - Negative values are clipped to 0.
    - scale_factor = 65535 / max(img_clipped)
    - scaled = clip(img,0,inf) * scale_factor, clipped to [0,65535].
    """
    x = np.asarray(img)
    finite = np.isfinite(x)
    if not np.any(finite):
        z = np.zeros_like(x, dtype=np.float32)
        return ScaleResult(z, float("nan"), float("nan"), 0.0)

    raw_min = float(np.min(x[finite]))
    x_clip = np.where(finite, np.maximum(x, 0.0), 0.0)

    raw_max = float(np.max(x_clip[finite]))
    if not np.isfinite(raw_max) or raw_max <= 0.0:
        z = np.zeros_like(x_clip, dtype=np.float32)
        return ScaleResult(z, raw_min, raw_max, 0.0)

    scale = 65535.0 / raw_max
    y = (x_clip * scale).astype(np.float32)
    y = np.clip(y, 0.0, 65535.0).astype(np.float32)
    return ScaleResult(y, raw_min, raw_max, float(scale))


def write_fits(
    out_path: str | Path,
    img_raw_float: np.ndarray,
    header_cards: dict,
    primary_dtype: str = "float32",
    primary_scaled_0_65535: bool = True,
    store_raw_extension: bool = True,
    extra_images: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """
    FITS output.

    Primary HDU:
      - Floating point (BITPIX = -32 or -64)
      - Optionally scaled to 0..65535 while still float.

    Extension:
      - RAW32/RAW64: unscaled floats for analysis.

    Extra images:
      - Dict {EXTNAME: array}, written as float32 ImageHDUs.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = np.float32 if primary_dtype.lower() == "float32" else np.float64
    raw = np.asarray(img_raw_float, dtype=dtype)

    hdr = fits.Header()
    for k, (v, c) in header_cards.items():
        hdr[k] = (v, c)

    hdus: list[fits.hdu.base.ExtensionHDU] = []

    if primary_scaled_0_65535:
        sc = scale_to_0_65535_float(raw)
        prim_data = sc.scaled.astype(dtype, copy=False)
        hdr["PRIMSCAL"] = ("0_65535", "Primary 0..65535 float")
        hdr["RAWMIN"] = (sc.raw_min, "Raw min pre-clip")
        hdr["RAWMAX"] = (sc.raw_max_clipped, "Raw max post-clip")
        hdr["SCLFAC"] = (sc.scale_factor, "Scale factor")
    else:
        prim_data = raw
        hdr["PRIMSCAL"] = ("NONE", "Primary is raw float")

    hdus.append(fits.PrimaryHDU(data=prim_data, header=hdr))

    if store_raw_extension:
        extname = "RAW32" if dtype == np.float32 else "RAW64"
        hdus.append(fits.ImageHDU(data=raw, name=extname))

    if extra_images:
        for name, arr in extra_images.items():
            if arr is None:
                continue
            a = np.asarray(arr, dtype=np.float32)
            hdus.append(fits.ImageHDU(data=a, name=str(name)))

    fits.HDUList(hdus).writeto(out_path, overwrite=True, output_verify="silentfix")
