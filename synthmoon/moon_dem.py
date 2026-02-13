from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import spiceypy as sp

from .albedo_maps import EquirectMap


@dataclass(frozen=True)
class LunarDEM:
    """
    Simple global lunar DEM wrapper.

    The DEM FITS is assumed to be an equirectangular (simple cylindrical) grid:
      - data shape (nlat, nlon)
      - lat spans -90..90
      - lon spans either 0..360 or -180..180 (configurable)
      - values are ABSOLUTE radius from Moon centre (not height), in metres or km

    For v0.x we use it for:
      - refining the intersection distance (iterative per-ray radius)
      - computing normals from finite differences for Lambert shading
    """
    dem_map: EquirectMap
    units: str              # "m" or "km"
    mean_radius_km: float

    @staticmethod
    def load_fits(
        path: str | Path,
        lon_mode: str = "0_360",
        units: str = "m",
        mean_radius_km: float = 1737.4,
    ) -> "LunarDEM":
        units = str(units).lower()
        if units not in ("m", "km"):
            raise ValueError("DEM units must be 'm' or 'km'")

        # Fill with mean radius in the native DEM units
        fill = mean_radius_km * (1000.0 if units == "m" else 1.0)
        emap = EquirectMap.load_fits(path, lon_mode=lon_mode, fill=float(fill))
        return LunarDEM(dem_map=emap, units=units, mean_radius_km=float(mean_radius_km))

    @property
    def shape(self) -> tuple[int, int]:
        return self.dem_map.data.shape

    def radius_km(self, lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
        r = self.dem_map.sample_bilinear(lon_deg, lat_deg)
        if self.units == "m":
            return np.asarray(r, dtype=np.float64) / 1000.0
        return np.asarray(r, dtype=np.float64)

    def elevation_m(self, lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
        r_km = self.radius_km(lon_deg, lat_deg)
        return (r_km - self.mean_radius_km) * 1000.0

    def _iau_unitvec(self, lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
        lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
        lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
        cl = np.cos(lat)
        x = cl * np.cos(lon)
        y = cl * np.sin(lon)
        z = np.sin(lat)
        return np.stack([x, y, z], axis=1)  # (N,3)

    def surface_points_iau_km(self, lon_deg: np.ndarray, lat_deg: np.ndarray, radius_km: np.ndarray | None = None) -> np.ndarray:
        u = self._iau_unitvec(lon_deg, lat_deg)
        r = self.radius_km(lon_deg, lat_deg) if radius_km is None else np.asarray(radius_km, dtype=np.float64)
        return u * r[:, None]

    def normals_iau(self, lon_deg: np.ndarray, lat_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (normals_iau, slope_deg) for given lon/lat arrays.

        Normals are computed via central differences in lon and lat on the DEM surface.
        """
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat = np.asarray(lat_deg, dtype=np.float64)

        nlat, nlon = self.dem_map.data.shape
        dlon = 360.0 / float(nlon)
        dlat = 180.0 / float(nlat)

        lon_p = lon + dlon
        lon_m = lon - dlon
        lat_p = np.clip(lat + dlat, -90.0, 90.0)
        lat_m = np.clip(lat - dlat, -90.0, 90.0)

        # Points on DEM surface in IAU frame (km)
        p0 = self.surface_points_iau_km(lon, lat)
        p_lon_p = self.surface_points_iau_km(lon_p, lat)
        p_lon_m = self.surface_points_iau_km(lon_m, lat)
        p_lat_p = self.surface_points_iau_km(lon, lat_p)
        p_lat_m = self.surface_points_iau_km(lon, lat_m)

        t_lon = p_lon_p - p_lon_m
        t_lat = p_lat_p - p_lat_m

        n = np.cross(t_lon, t_lat)
        nn = np.linalg.norm(n, axis=1)
        n = n / np.maximum(nn[:, None], 1e-15)

        # Ensure outward orientation (dot with radial > 0)
        radial = p0 / np.maximum(np.linalg.norm(p0, axis=1)[:, None], 1e-15)
        flip = np.sum(n * radial, axis=1) < 0.0
        n[flip] *= -1.0

        # Slope angle between DEM normal and radial direction
        cosang = np.clip(np.sum(n * radial, axis=1), -1.0, 1.0)
        slope_deg = np.rad2deg(np.arccos(cosang))
        return n, slope_deg

    def normals_j2000(self, et: float, lon_deg: np.ndarray, lat_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_iau, slope_deg = self.normals_iau(lon_deg, lat_deg)
        M = sp.pxform("IAU_MOON", "J2000", float(et))
        n_j2k = (M @ n_iau.T).T
        n_j2k = n_j2k / np.maximum(np.linalg.norm(n_j2k, axis=1)[:, None], 1e-15)
        return n_j2k, slope_deg
