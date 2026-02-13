from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import spiceypy as sp
from astropy.io import fits


def _normalize(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


@dataclass(frozen=True)
class LunarDEM:
    """
    Equirectangular (simple cylindrical) lunar DEM sampler + gradient.

    Assumptions:
    - Data is a 2D array [lat_index, lon_index] covering lat -90..+90 and lon either 0..360 or -180..+180.
    - First row is +90 (north), last row is -90 (south) (same convention as the albedo map sampler).

    Units:
    - DEM is interpreted as "height relative to mean radius" in metres (default),
      but you can declare any unit and apply a scale.
    """
    data_m: np.ndarray  # float64 metres
    lon_mode: str       # "0_360" or "-180_180"
    fill_m: float = 0.0

    @staticmethod
    def load_fits(path: str | Path, lon_mode: str = "0_360", fill_m: float = 0.0, scale: float = 1.0, units: str = "m") -> "LunarDEM":
        path = Path(path)
        arr = fits.getdata(path)
        if arr.ndim != 2:
            raise ValueError(f"DEM must be 2D; got shape {arr.shape} from {path}")
        # Convert to float64 in metres
        x = np.asarray(arr, dtype=np.float64)
        if units.lower() in ("m", "meter", "metre", "meters", "metres"):
            xm = x
        elif units.lower() in ("km", "kilometer", "kilometre", "kilometers", "kilometres"):
            xm = x * 1000.0
        else:
            raise ValueError(f"Unsupported DEM units: {units!r} (use 'm' or 'km')")
        xm = xm * float(scale)
        return LunarDEM(data_m=xm, lon_mode=str(lon_mode), fill_m=float(fill_m))

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data_m.shape

    def _xy(self, lon_deg: np.ndarray, lat_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat = np.asarray(lat_deg, dtype=np.float64)
        nlat, nlon = self.data_m.shape

        # longitude -> x in [0, nlon)
        if self.lon_mode == "0_360":
            lonw = np.mod(lon, 360.0)
            x = lonw / 360.0 * nlon
        elif self.lon_mode == "-180_180":
            lonw = ((lon + 180.0) % 360.0) - 180.0
            x = (lonw + 180.0) / 360.0 * nlon
        else:
            raise ValueError("lon_mode must be '0_360' or '-180_180'")

        # latitude -> y in [0, nlat), with y=0 at +90
        latc = np.clip(lat, -90.0, 90.0)
        y = (90.0 - latc) / 180.0 * nlat
        return x, y

    def sample_bilinear_m(self, lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
        """Bilinear sampling in metres (fast enough for v0.x)."""
        x, y = self._xy(lon_deg, lat_deg)
        nlat, nlon = self.data_m.shape

        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = (x0 + 1) % nlon
        y1 = np.clip(y0 + 1, 0, nlat - 1)

        x0 = np.mod(x0, nlon)
        y0 = np.clip(y0, 0, nlat - 1)

        fx = x - x0
        fy = y - y0

        a00 = self.data_m[y0, x0]
        a10 = self.data_m[y0, x1]
        a01 = self.data_m[y1, x0]
        a11 = self.data_m[y1, x1]

        out = (1 - fx) * (1 - fy) * a00 + fx * (1 - fy) * a10 + (1 - fx) * fy * a01 + fx * fy * a11
        out = np.where(np.isfinite(out), out, self.fill_m)
        return out

    def gradient_m_per_rad(self, lon_deg: np.ndarray, lat_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Central-difference gradient of DEM height with respect to lon/lat (radians).
        Returns: (dh/dlon_rad, dh/dlat_rad) in m/rad.
        """
        nlat, nlon = self.data_m.shape
        dlon_deg = 360.0 / nlon
        dlat_deg = 180.0 / nlat

        lon = np.asarray(lon_deg, dtype=np.float64)
        lat = np.asarray(lat_deg, dtype=np.float64)

        h_p = self.sample_bilinear_m(lon + dlon_deg, lat)
        h_m = self.sample_bilinear_m(lon - dlon_deg, lat)
        dh_dlon_deg = (h_p - h_m) / (2.0 * dlon_deg)

        h_p = self.sample_bilinear_m(lon, lat + dlat_deg)
        h_m = self.sample_bilinear_m(lon, lat - dlat_deg)
        dh_dlat_deg = (h_p - h_m) / (2.0 * dlat_deg)

        deg2rad = np.pi / 180.0
        return dh_dlon_deg / deg2rad, dh_dlat_deg / deg2rad


def moon_lonlat_deg_from_pts_j2000(et: float, moon_center_j2000: np.ndarray, pts_j2000: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert J2000 points to lon/lat in IAU_MOON (degrees) + return the Moon-fixed vectors.

    Returns: lon_deg, lat_deg, v_moonfixed (N,3) where v_moonfixed = R(J2000->IAU_MOON) @ (pts - moon_center)
    """
    M = np.array(sp.pxform("J2000", "IAU_MOON", et), dtype=float)
    v = pts_j2000 - moon_center_j2000[None, :]
    vf = (M @ v.T).T
    r = np.linalg.norm(vf, axis=1)
    lon = np.rad2deg(np.arctan2(vf[:, 1], vf[:, 0]))
    lat = np.rad2deg(np.arcsin(np.clip(vf[:, 2] / np.maximum(r, 1e-15), -1.0, 1.0)))
    return lon, lat, vf


def refine_hits_with_dem(
    *,
    et: float,
    origins_j2000: np.ndarray,
    dirs_j2000: np.ndarray,
    t_init_km: np.ndarray,
    moon_center_j2000: np.ndarray,
    moon_radius_km: float,
    dem: LunarDEM,
    n_iter: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Refine sphere-intersection hits using DEM heightfield.

    Returns:
      pts_j2000 (N,3),
      normals_j2000 (N,3),
      height_m (N,),
      slope_deg (N,)
    """
    t = np.array(t_init_km, dtype=np.float64, copy=True)
    t = np.where(np.isfinite(t), t, np.nan)

    # Newton-like refinement along the ray: enforce |p - C| = Rm + h(lon,lat)
    for _ in range(max(1, int(n_iter))):
        pts = origins_j2000 + t[:, None] * dirs_j2000
        lon, lat, vf = moon_lonlat_deg_from_pts_j2000(et, moon_center_j2000, pts)

        h_m = dem.sample_bilinear_m(lon, lat)
        r_target = moon_radius_km + (h_m / 1000.0)

        r = np.linalg.norm(vf, axis=1)
        f = r - r_target  # km
        if np.nanmax(np.abs(f)) < 1e-6:
            break

        # dr/dt ≈ dot(dir, radial_unit). Ignore (small) variation of r_target with t.
        radial_j = (pts - moon_center_j2000[None, :])
        radial_j = _normalize(radial_j)
        drdt = np.sum(dirs_j2000 * radial_j, axis=1)
        drdt = np.where(np.abs(drdt) < 1e-8, np.sign(drdt) * 1e-8, drdt)

        step = f / drdt
        # guard against crazy steps near grazing intersections
        step = np.clip(step, -10.0, 10.0)
        t = t - step

    # final points + DEM at those lon/lat
    pts = origins_j2000 + t[:, None] * dirs_j2000
    lon, lat, vf = moon_lonlat_deg_from_pts_j2000(et, moon_center_j2000, pts)
    h_m = dem.sample_bilinear_m(lon, lat)

    # normal from DEM gradients in the Moon-fixed frame
    dh_dlon, dh_dlat = dem.gradient_m_per_rad(lon, lat)  # m/rad
    dr_dlon = dh_dlon / 1000.0  # km/rad
    dr_dlat = dh_dlat / 1000.0  # km/rad

    lam = np.deg2rad(lon)
    phi = np.deg2rad(lat)

    r0 = moon_radius_km + h_m / 1000.0  # km
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    coslam = np.cos(lam)
    sinlam = np.sin(lam)

    # Partial derivatives of position p(lon,lat) in Moon-fixed coordinates
    # Using r=r0(lon,lat), with dr_dlon, dr_dlat.
    dpx_dlon = cosphi * (dr_dlon * coslam - r0 * sinlam)
    dpy_dlon = cosphi * (dr_dlon * sinlam + r0 * coslam)
    dpz_dlon = dr_dlon * sinphi

    common = dr_dlat * cosphi - r0 * sinphi
    dpx_dlat = coslam * common
    dpy_dlat = sinlam * common
    dpz_dlat = dr_dlat * sinphi + r0 * cosphi

    t_lon = np.stack([dpx_dlon, dpy_dlon, dpz_dlon], axis=1)
    t_lat = np.stack([dpx_dlat, dpy_dlat, dpz_dlat], axis=1)

    n_fixed = np.cross(t_lon, t_lat)
    n_fixed = _normalize(n_fixed)

    # Ensure outward normal (dot with position vector should be positive)
    v_fixed = np.stack([r0 * cosphi * coslam, r0 * cosphi * sinlam, r0 * sinphi], axis=1)
    flip = np.einsum("ij,ij->i", n_fixed, v_fixed) < 0.0
    n_fixed[flip] *= -1.0

    # Convert normals to J2000
    M = np.array(sp.pxform("IAU_MOON", "J2000", et), dtype=float)
    n_j = (M @ n_fixed.T).T
    n_j = _normalize(n_j)

    # Slope in degrees: angle between DEM normal and radial normal
    radial_fixed = _normalize(v_fixed)
    cosang = np.einsum("ij,ij->i", n_fixed, radial_fixed)
    cosang = np.clip(cosang, -1.0, 1.0)
    slope_deg = np.rad2deg(np.arccos(cosang))

    return pts, n_j, h_m, slope_deg
