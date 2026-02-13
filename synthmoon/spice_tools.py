from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
import hashlib
import numpy as np
import spiceypy as sp

AU_KM = 149_597_870.700  # IAU 2012 definition, stable

def _is_known_frame(name: str) -> bool:
    try:
        return int(sp.namfrm(str(name))) != 0
    except Exception:
        return False

def resolve_moon_frame(requested: str) -> str:
    """Resolve a user-requested lunar body-fixed frame name to one SPICE recognises.

    Common cases:
      - If you load moon_assoc_me.tf / moon_assoc_pa.tf, the generic aliases MOON_ME/MOON_PA exist.
      - If not, DE440 kernels define MOON_ME_DE440_ME421 and MOON_PA_DE440.

    If nothing matches, fall back to IAU_MOON.
    """
    req = str(requested).strip()
    if not req:
        return "IAU_MOON"
    if _is_known_frame(req):
        return req

    up = req.upper()
    if up in ("MOON_ME", "MOON_PA"):
        if up == "MOON_ME":
            candidates = [
                "MOON_ME",
                "MOON_ME_DE440_ME421",
                "MOON_ME_DE421",
                "MOON_ME_DE418",
                "MOON_ME_DE403",
            ]
        else:
            candidates = [
                "MOON_PA",
                "MOON_PA_DE440",
                "MOON_PA_DE421",
                "MOON_PA_DE418",
                "MOON_PA_DE403",
            ]
        for c in candidates:
            if _is_known_frame(c):
                return c

    # last resort
    return "IAU_MOON"


def _kernel_path(kernels_dir: str | Path, rel: str) -> Path:
    return Path(kernels_dir) / Path(rel)


def load_optional_moon_frame_kernels(
    kernels_dir: str | Path,
    auto_download_small_fk: bool = False,
) -> list[Path]:
    """Load optional Moon frame/orientation kernels if present.

    This is a convenience layer on top of the meta-kernel. It looks for the
    standard DE440 Moon frame kernels and binary PCK used to define MOON_ME/
    MOON_PA aliases and higher-accuracy body orientation.

    If `auto_download_small_fk` is True, missing *small* FK files
    (moon_assoc_me.tf and moon_assoc_pa.tf) are downloaded from the NAIF
    generic_kernels repository.

    Returns
    -------
    loaded : list[Path]
        Paths that were successfully loaded.
    """
    kd = Path(kernels_dir)
    loaded: list[Path] = []

    # Optional FK/BPC needed for MOON_ME / MOON_PA aliases (small TF files)
    assoc_dir = kd / "fk" / "satellites"
    assoc_dir.mkdir(parents=True, exist_ok=True)

    assoc_me = assoc_dir / "moon_assoc_me.tf"
    assoc_pa = assoc_dir / "moon_assoc_pa.tf"
    de440_tf  = assoc_dir / "moon_de440_250416.tf"  # naming can vary; we only try this known one
    pa_bpc    = kd / "pck" / "moon_pa_de440_200625.bpc"

    if auto_download_small_fk:
        base = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites"
        if not assoc_me.exists():
            urlretrieve(f"{base}/moon_assoc_me.tf", assoc_me)
        if not assoc_pa.exists():
            urlretrieve(f"{base}/moon_assoc_pa.tf", assoc_pa)

    for p in (assoc_me, assoc_pa, de440_tf, pa_bpc):
        try:
            if p.exists() and p.stat().st_size > 0:
                sp.furnsh(str(p))
                loaded.append(p)
        except Exception:
            # keep going; health check will report any remaining issues
            pass

    return loaded


def available_moon_frames() -> list[str]:
    """Return a short list of Moon-fixed frames that are *currently* recognised."""
    candidates = [
        "IAU_MOON",
        "MOON_ME",
        "MOON_PA",
        "MOON_ME_DE440_ME421",
        "MOON_PA_DE440",
    ]
    return [c for c in candidates if _is_known_frame(c)]


def require_moon_frame(requested: str, strict: bool = True) -> str:
    """Resolve a requested Moon frame and (optionally) fail if unavailable."""
    resolved = resolve_moon_frame(requested)
    if strict and str(requested).strip() and (resolved == "IAU_MOON") and (str(requested).strip().upper() != "IAU_MOON"):
        avail = ", ".join(available_moon_frames()) or "(none)"
        raise RuntimeError(
            "Requested moon.spice_frame=%r is not available with currently loaded kernels. "
            "Available Moon frames: %s.\n"
            "Fix: ensure these files are present and loaded:\n"
            "  KERNELS/pck/moon_pa_de440_200625.bpc\n"
            "  KERNELS/fk/satellites/moon_de440_*.tf\n"
            "  KERNELS/fk/satellites/moon_assoc_me.tf (for MOON_ME alias)\n"
            "  KERNELS/fk/satellites/moon_assoc_pa.tf (for MOON_PA alias)"
            % (requested, avail)
        )
    return resolved


def sha256_prefix(path: str | Path, n_hex: int = 16) -> str:
    """SHA256 hash prefix for reproducibility manifests (small n_hex)."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:n_hex]

@dataclass(frozen=True)
class SpiceContext:
    et: float
    utc: str

def load_kernels(meta_kernel_path: str | Path) -> None:
    mk = Path(meta_kernel_path)
    if not mk.exists():
        raise FileNotFoundError(f"Missing meta-kernel: {mk}. Run: uv run python scripts/download_kernels.py")
    sp.furnsh(str(mk))

def utc_to_et(utc: str) -> float:
    return float(sp.utc2et(utc))

def get_body_radii_km(body: str) -> np.ndarray:
    radii = sp.bodvrd(body, "RADII", 3)[1]
    return np.array(radii, dtype=float)

def earth_site_state_j2000_earthcenter(et: float, lon_deg: float, lat_deg: float, height_m: float) -> np.ndarray:
    """
    Earth site state in Earth-centered J2000 (km, km/s). Add Earth's SSB state to make it barycentric.
    """
    radii = get_body_radii_km("EARTH")
    re = float(radii[0])
    rp = float(radii[2])
    f = (re - rp) / re

    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    alt_km = float(height_m) / 1000.0

    r_ecef = np.array(sp.georec(lon, lat, alt_km, re, f), dtype=float)

    state_ecef = np.zeros(6, dtype=float)
    state_ecef[:3] = r_ecef

    xform = np.array(sp.sxform("IAU_EARTH", "J2000", et), dtype=float)  # 6x6
    return xform @ state_ecef

def spacecraft_state_j2000(cfg_obs: dict) -> np.ndarray:
    return np.array(
        [
            cfg_obs["x_km"], cfg_obs["y_km"], cfg_obs["z_km"],
            cfg_obs["vx_km_s"], cfg_obs["vy_km_s"], cfg_obs["vz_km_s"],
        ],
        dtype=float,
    )

def get_sun_earth_moon_states_ssb(et: float) -> dict:
    """States relative to Solar System Barycenter (SSB) in J2000."""
    sun = np.array(sp.spkezr("SUN", et, "J2000", "NONE", "0")[0], dtype=float)
    earth = np.array(sp.spkezr("EARTH", et, "J2000", "NONE", "0")[0], dtype=float)
    moon = np.array(sp.spkezr("MOON", et, "J2000", "NONE", "0")[0], dtype=float)
    return {"SUN": sun, "EARTH": earth, "MOON": moon}

def lunar_north_in_j2000(et: float, moon_frame: str = "IAU_MOON") -> np.ndarray:
    """
    Unit vector of the lunar +Z axis (north pole) in J2000 at time et.

    By default this uses the IAU_MOON body-fixed frame, but you can pass a
    higher-accuracy lunar frame such as MOON_ME or MOON_PA if the relevant
    FK/BPC kernels are loaded.
    """
    mf = resolve_moon_frame(str(moon_frame))
    rot = np.array(sp.pxform(str(mf), "J2000", et), dtype=float)
    v = rot @ np.array([0.0, 0.0, 1.0], dtype=float)
    return v / np.linalg.norm(v)

def list_loaded_kernels() -> list[str]:
    """List loaded kernel file paths; handle SpiceyPy 4- or 5-tuple kdata()."""
    count = sp.ktotal("ALL")
    out: list[str] = []
    for i in range(count):
        res = sp.kdata(i, "ALL")
        if isinstance(res, (tuple, list)):
            if len(res) >= 1:
                out.append(str(res[0]))
        else:
            out.append(str(res))
    return out

def inv_solar_irradiance_scale(point_km: np.ndarray, sun_pos_km: np.ndarray) -> float:
    """Return F_sun(point) with F_sun(1 AU)=1."""
    d = float(np.linalg.norm(sun_pos_km - point_km))
    return (AU_KM / d) ** 2