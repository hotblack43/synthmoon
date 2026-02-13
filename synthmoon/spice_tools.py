from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import spiceypy as sp

AU_KM = 149_597_870.700  # IAU 2012 definition, stable

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
    Return Earth-fixed surface site state in Earth-centered J2000 (km, km/s).

    NOTE: This is Earth-centered, not barycentric. If you want SSB coords, add Earth's SSB state.
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
    state_j2000 = xform @ state_ecef
    return state_j2000

def spacecraft_state_j2000(cfg_obs: dict) -> np.ndarray:
    """
    Interpret the user-provided spacecraft state as J2000 barycentric (SSB) for v0.
    """
    st = np.array(
        [
            cfg_obs["x_km"], cfg_obs["y_km"], cfg_obs["z_km"],
            cfg_obs["vx_km_s"], cfg_obs["vy_km_s"], cfg_obs["vz_km_s"],
        ],
        dtype=float,
    )
    return st

def get_sun_earth_moon_states_ssb(et: float) -> dict:
    """
    Return J2000 states (km, km/s) of Sun, Earth, Moon relative to Solar System Barycenter (SSB).
    """
    sun = np.array(sp.spkezr("SUN", et, "J2000", "NONE", "0")[0], dtype=float)
    earth = np.array(sp.spkezr("EARTH", et, "J2000", "NONE", "0")[0], dtype=float)
    moon = np.array(sp.spkezr("MOON", et, "J2000", "NONE", "0")[0], dtype=float)
    return {"SUN": sun, "EARTH": earth, "MOON": moon}

def lunar_north_in_j2000(et: float) -> np.ndarray:
    """
    Unit vector of lunar north pole in J2000 at time et.
    +Z axis of IAU_MOON transformed into J2000.
    """
    rot = np.array(sp.pxform("IAU_MOON", "J2000", et), dtype=float)
    v = rot @ np.array([0.0, 0.0, 1.0], dtype=float)
    return v / np.linalg.norm(v)

def list_loaded_kernels() -> list[str]:
    """
    Returns a list of loaded kernel file paths (strings).

    SpiceyPy versions differ: kdata() may return 4 or 5 values.
    Handle both safely.
    """
    count = sp.ktotal("ALL")
    out: list[str] = []
    for i in range(count):
        res = sp.kdata(i, "ALL")
        if isinstance(res, tuple) or isinstance(res, list):
            if len(res) == 5:
                file, ktype, source, handle, found = res
                if not found:
                    continue
                out.append(str(file))
            elif len(res) == 4:
                file, ktype, source, handle = res
                out.append(str(file))
            else:
                out.append(str(res[0]))
        else:
            out.append(str(res))
    return out

def inv_solar_irradiance_scale(point_km: np.ndarray, sun_pos_km: np.ndarray) -> float:
    """
    Return F_sun(point) in arbitrary units where F_sun(1 AU) = 1.
    """
    d = float(np.linalg.norm(sun_pos_km - point_km))
    return (AU_KM / d) ** 2
