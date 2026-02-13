from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class ScaleResult:
    data_int16: np.ndarray
    bscale: float
    bzero: float
    datamin: float
    datamax: float


def scale_to_int16_full_range(img: np.ndarray) -> ScaleResult:
    """
    Map img float -> int16 with BSCALE/BZERO such that min(img) maps to -32768 and max(img) to +32767.
    """
    img = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(img)
    if not np.any(finite):
        data = np.zeros(img.shape, dtype=np.int16)
        return ScaleResult(data, 1.0, 0.0, float("nan"), float("nan"))

    vmin = float(np.min(img[finite]))
    vmax = float(np.max(img[finite]))
    if vmax == vmin:
        bscale = 1.0
        bzero = vmin
        data = np.zeros(img.shape, dtype=np.int16)
        return ScaleResult(data, bscale, bzero, vmin, vmax)

    bscale = (vmax - vmin) / 65535.0
    bzero = vmin + 32768.0 * bscale
    data = np.round((img - bzero) / bscale).astype(np.int64)
    data = np.clip(data, -32768, 32767).astype(np.int16)
    return ScaleResult(data, float(bscale), float(bzero), vmin, vmax)


def write_fits(
    out_path: str | Path,
    img_float: np.ndarray,
    header_cards: dict,
    store_int16: bool = True,
    store_float_extension: bool = True,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hdr = fits.Header()
    for k, (v, c) in header_cards.items():
        hdr[k] = (v, c)

    hdul = []

    if store_int16:
        sc = scale_to_int16_full_range(img_float)
        hdr["BSCALE"] = (sc.bscale, "FITS scaling: physical = BSCALE*val + BZERO")
        hdr["BZERO"] = (sc.bzero, "FITS scaling: physical = BSCALE*val + BZERO")
        hdr["DATAMIN"] = (sc.datamin, "Min of float image used for scaling")
        hdr["DATAMAX"] = (sc.datamax, "Max of float image used for scaling")
        prim = fits.PrimaryHDU(data=sc.data_int16, header=hdr)
    else:
        prim = fits.PrimaryHDU(data=np.asarray(img_float, dtype=np.float32), header=hdr)

    hdul.append(prim)

    if store_float_extension:
        h = fits.ImageHDU(data=np.asarray(img_float, dtype=np.float32), name="FLOAT32")
        hdul.append(h)

    fits.HDUList(hdul).writeto(out_path, overwrite=True)
