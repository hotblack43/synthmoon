from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class ScaleResult:
    scaled: np.ndarray
    raw_min: float
    raw_max_clipped: float
    scale_factor: float


def scale_to_0_65535_float(img: np.ndarray) -> ScaleResult:
    """Scale a float image into [0, 65535] while remaining floating-point.

    Notes
    -----
    * Negative values are clipped to 0.
    * scale_factor = 65535 / max(img_clipped)
    * scaled = clip(img,0,inf) * scale_factor, then clipped to [0,65535].
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


def _add_kernel_manifest(
    hdr: fits.Header,
    *,
    kernel_manifest: list[tuple[str, int, str]] | None,
    meta_kernel_path: str | None,
    meta_kernel_sha256_prefix: str | None,
) -> None:
    """Add a compact kernel manifest to the FITS header.

    Keywords are kept <= 8 characters to avoid FITS keyword warnings.
    Full per-kernel details are stored in HISTORY lines.
    """
    if meta_kernel_path is not None:
        mk = Path(meta_kernel_path)
        hdr["MKNAME"] = (mk.name[:68], "Meta-kernel filename")
        if mk.exists():
            hdr["MKBYTES"] = (int(mk.stat().st_size), "Meta-kernel size [bytes]")
        if meta_kernel_sha256_prefix:
            hdr["MKSH256"] = (str(meta_kernel_sha256_prefix)[:16], "Meta-kernel SHA256 prefix")

    if kernel_manifest is None:
        return

    hdr["KMCNT"] = (int(len(kernel_manifest)), "Loaded kernel count")
    hdr.add_history("Kernel manifest: basename bytes sha256prefix")
    for i, (path, nbytes, sha) in enumerate(kernel_manifest, start=1):
        base = Path(path).name
        sh = (sha or "")
        line = f"KRN{i:03d} {base} {int(nbytes)} {sh}".strip()
        hdr.add_history(line[:72])


def write_fits_mef(
    out_path: str | Path,
    img_raw_if: np.ndarray,
    header_cards: dict,
    primary_dtype: str = "float32",
    primary_scaled_0_65535: bool = True,
    store_raw_extension: bool = True,
    radiance_plane: Optional[Tuple[np.ndarray, dict]] = None,  # (radiance array, header dict)
    kernel_manifest: list[tuple[str, int, str]] | None = None,
    meta_kernel_path: str | None = None,
    meta_kernel_sha256_prefix: str | None = None,
) -> None:
    """Write a classic multi-extension FITS (MEF).

    * Primary: scaled 0..65535 float (optional)
    * RAW32/RAW64: unscaled I/F (optional)
    * RADTSI: broadband radiance W m-2 sr-1 (optional)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = np.float32 if primary_dtype.lower() == "float32" else np.float64
    raw = np.asarray(img_raw_if, dtype=dtype)

    hdr = fits.Header()
    for k, (v, c) in header_cards.items():
        hdr[k] = (v, c)

    _add_kernel_manifest(
        hdr,
        kernel_manifest=kernel_manifest,
        meta_kernel_path=meta_kernel_path,
        meta_kernel_sha256_prefix=meta_kernel_sha256_prefix,
    )

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
        h = fits.Header()
        h["BUNIT"] = ("I/F", "Radiance factor")
        hdus.append(fits.ImageHDU(data=raw, header=h, name=extname))

    if radiance_plane is not None:
        rad, rad_hdr_dict = radiance_plane
        rad = np.asarray(rad, dtype=dtype)
        h = fits.Header()
        for k, (v, c) in rad_hdr_dict.items():
            h[k] = (v, c)
        hdus.append(fits.ImageHDU(data=rad, header=h, name="RADTSI"))

    fits.HDUList(hdus).writeto(out_path, overwrite=True, output_verify="silentfix")


def write_fits_cube(
    out_path: str | Path,
    layers: Dict[str, Tuple[np.ndarray, str]],
    header_cards: dict,
    cube_dtype: str = "float32",
    kernel_manifest: list[tuple[str, int, str]] | None = None,
    meta_kernel_path: str | None = None,
    meta_kernel_sha256_prefix: str | None = None,
) -> None:
    """Write a single-HDU FITS cube (NAXIS=3) with named layers.

    DS9 will show a cube slider.

    Parameters
    ----------
    layers:
        Mapping {layer_name: (array2d, unit_str)}. Arrays must share (ny,nx).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = np.float32 if cube_dtype.lower() == "float32" else np.float64

    names = list(layers.keys())
    arrs = []
    units = []
    for k in names:
        a, u = layers[k]
        arrs.append(np.asarray(a, dtype=dtype))
        units.append(str(u))

    # numpy shape (nz, ny, nx) -> FITS axes (NAXIS3, NAXIS2, NAXIS1)
    cube = np.stack(arrs, axis=0)

    hdr = fits.Header()
    for k, (v, c) in header_cards.items():
        hdr[k] = (v, c)

    _add_kernel_manifest(
        hdr,
        kernel_manifest=kernel_manifest,
        meta_kernel_path=meta_kernel_path,
        meta_kernel_sha256_prefix=meta_kernel_sha256_prefix,
    )

    hdr["BUNIT"] = ("MIXED", "Units vary by layer; see LAYn/LUn")
    hdr["NLAYERS"] = (len(names), "Number of layers in cube")
    for i, (nm, un) in enumerate(zip(names, units), start=1):
        hdr[f"LAY{i}"] = (nm, "Layer name")
        hdr[f"LU{i}"] = (un, "Layer unit")

    fits.PrimaryHDU(data=cube, header=hdr).writeto(out_path, overwrite=True, output_verify="silentfix")
