from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class EquirectMap:
    """
    Simple equirectangular map sampler.

    Assumptions (configurable):
      - longitude spans either 0..360 or -180..180
      - latitude spans -90..90
      - data array shape is (nlat, nlon)

    The map values are interpreted as *albedo* (0..1) unless you choose to scale
    them in caller code.
    """
    data: np.ndarray            # (nlat, nlon), float64 recommended
    lon_mode: str               # "0_360" or "-180_180"
    fill: float = 0.0

    @staticmethod
    def load_fits(path: str | Path, lon_mode: str = "0_360", fill: float = 0.0) -> "EquirectMap":
        path = Path(path)
        arr = fits.getdata(path).astype(np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Map must be 2D; got shape {arr.shape} from {path}")
        if lon_mode not in ("0_360", "-180_180"):
            raise ValueError("lon_mode must be '0_360' or '-180_180'")
        return EquirectMap(data=arr, lon_mode=lon_mode, fill=float(fill))

    def sample(self, lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
        """
        Nearest-neighbour sampling (fast, adequate for v0.x).
        lon_deg, lat_deg arrays must be broadcastable to same shape.
        """
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat = np.asarray(lat_deg, dtype=np.float64)

        nlat, nlon = self.data.shape

        # Wrap longitudes into the expected domain
        if self.lon_mode == "0_360":
            lonw = np.mod(lon, 360.0)
            x = lonw / 360.0 * nlon
        else:
            lonw = ((lon + 180.0) % 360.0) - 180.0
            x = (lonw + 180.0) / 360.0 * nlon

        # Clamp latitude
        latc = np.clip(lat, -90.0, 90.0)

        # Assume first row is north (+90). If your map is flipped, flip the FITS once.
        y = (90.0 - latc) / 180.0 * nlat

        xi = np.clip(np.floor(x).astype(int), 0, nlon - 1)
        yi = np.clip(np.floor(y).astype(int), 0, nlat - 1)

        out = self.data[yi, xi]
        out = np.where(np.isfinite(out), out, self.fill)
        return out

    def sample_interp(self, lon_deg: np.ndarray, lat_deg: np.ndarray, interp: str = "nearest") -> np.ndarray:
        """Sample the map using a selectable interpolation method.

        Parameters
        ----------
        interp : str
            "nearest" (default) or "bilinear".

        Notes
        -----
        v0.x used nearest-neighbour everywhere for simplicity; "bilinear" is
        smoother and helps reduce aliasing/salt-and-pepper artifacts.
        """
        m = str(interp).strip().lower()
        if m in ("nearest", "nn"):
            return self.sample(lon_deg, lat_deg)
        if m in ("bilinear", "linear"):
            return self.sample_bilinear(lon_deg, lat_deg)
        raise ValueError("interp must be 'nearest' or 'bilinear'")

    def sample_bilinear(self, lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
        """
        Bilinear sampling (smoother than nearest-neighbour).

        - longitude wraps (periodic)
        - latitude clamps at poles
        """
        lon = np.asarray(lon_deg, dtype=np.float64)
        lat = np.asarray(lat_deg, dtype=np.float64)

        nlat, nlon = self.data.shape

        # Wrap longitudes into the expected domain
        if self.lon_mode == "0_360":
            lonw = np.mod(lon, 360.0)
            x = lonw / 360.0 * nlon
        else:
            lonw = ((lon + 180.0) % 360.0) - 180.0
            x = (lonw + 180.0) / 360.0 * nlon

        latc = np.clip(lat, -90.0, 90.0)
        y = (90.0 - latc) / 180.0 * nlat

        # wrap x into [0,nlon)
        x = np.mod(x, nlon)
        # clamp y into [0,nlat-1-eps]
        y = np.clip(y, 0.0, np.nextafter(float(nlat - 1), 0.0))

        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = (x0 + 1) % nlon
        y1 = np.minimum(y0 + 1, nlat - 1)

        dx = x - x0
        dy = y - y0

        v00 = self.data[y0, x0]
        v10 = self.data[y0, x1]
        v01 = self.data[y1, x0]
        v11 = self.data[y1, x1]

        out = (1.0 - dx) * (1.0 - dy) * v00 + dx * (1.0 - dy) * v10 + (1.0 - dx) * dy * v01 + dx * dy * v11
        out = np.where(np.isfinite(out), out, self.fill)
        return out


def toy_land_ocean_albedo(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    ocean: float = 0.06,
    land: float = 0.25,
) -> np.ndarray:
    """
    Deterministic "toy Earth" land/ocean pattern (no external data).

    This is NOT geographically accurate; it is only for getting the pipeline working:
    - land is brighter than ocean
    - broad continent-like blobs
    """
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))

    f = (
        0.55*np.sin(1.0*lon)*np.cos(1.3*lat) +
        0.35*np.cos(2.0*lon + 0.7)*np.cos(0.8*lat) +
        0.20*np.sin(3.0*lon - 1.1)*np.sin(1.1*lat)
    )
    land_mask = f > 0.15
    return np.where(land_mask, float(land), float(ocean))


def apply_simple_clouds(
    base_albedo: np.ndarray,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    cloud_amount: float = 0.3,
    cloud_albedo: float = 0.6,
) -> np.ndarray:
    """
    Overlay a deterministic cloud field; cloud_amount in [0,1].
    - controls both coverage and mixing strength (simple single knob).
    """
    c = float(np.clip(cloud_amount, 0.0, 1.0))
    if c <= 0.0:
        return np.asarray(base_albedo, dtype=np.float64)

    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))

    field = 0.5*(1.0 + np.sin(2.7*lon + 0.4)*np.cos(2.1*lat - 0.2))
    field = 0.65*field + 0.35*(0.5*(1.0 + np.sin(5.3*lon - 1.3)*np.sin(3.7*lat + 0.6)))

    thr = 1.0 - 0.85*c
    mask = field > thr

    out = np.array(base_albedo, copy=True, dtype=np.float64)
    out[mask] = (1.0 - c)*out[mask] + c*float(cloud_albedo)
    return out
