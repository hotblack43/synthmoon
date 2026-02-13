from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .spice_tools import inv_solar_irradiance_scale
from .intersect import ray_sphere_intersect


def _normalize(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


@dataclass(frozen=True)
class EarthDiskSampler:
    """
    Deterministic quasi-random samples over a spherical cap (Earth disk) using
    low-discrepancy sequences in (u,v).

    For a given angular radius alpha, we sample:
      cos(theta) = 1 - u*(1 - cos(alpha))
      phi = 2*pi*v
    Weight per sample = Omega / N, where Omega = 2*pi*(1 - cos(alpha))
    """
    n_samples: int
    u: np.ndarray  # (N,)
    v: np.ndarray  # (N,)

    @staticmethod
    def create(n_samples: int) -> "EarthDiskSampler":
        i = np.arange(n_samples, dtype=float)
        # two irrational multipliers -> low-discrepancy on unit square
        u = (i * 0.6180339887498949) % 1.0
        v = (i * 0.7548776662466927) % 1.0
        return EarthDiskSampler(n_samples=n_samples, u=u, v=v)

    def directions(self, e_dir: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
        """
        Build direction vectors (N,3) for the cap around e_dir with half-angle alpha (radians).
        Returns (omega, w) where w is per-sample solid-angle weight.
        """
        e = e_dir / np.linalg.norm(e_dir)
        # build basis u_hat, v_hat perpendicular to e
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(tmp, e)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        u_hat = np.cross(tmp, e)
        u_hat /= np.linalg.norm(u_hat)
        v_hat = np.cross(e, u_hat)

        cos_alpha = np.cos(alpha)
        cos_t = 1.0 - self.u * (1.0 - cos_alpha)
        sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
        phi = 2.0 * np.pi * self.v

        # omega = cos_t*e + sin_t*(cos(phi)*u_hat + sin(phi)*v_hat)
        omega = (cos_t[:, None] * e[None, :]) + (sin_t[:, None] * (np.cos(phi)[:, None] * u_hat[None, :] + np.sin(phi)[:, None] * v_hat[None, :]))
        omega = _normalize(omega)

        Omega = 2.0 * np.pi * (1.0 - cos_alpha)
        w = Omega / float(self.n_samples)
        return omega, float(w)


def lambert_sun_if(hit_points: np.ndarray, normals: np.ndarray, sun_pos: np.ndarray, moon_albedo: float) -> np.ndarray:
    """
    Direct solar contribution in I/F_moon units (normalised by solar irradiance at the lunar point).
    For Lambert, I/F = A_moon * mu0.
    """
    s_dir = _normalize(sun_pos[None, :] - hit_points)
    mu0 = np.maximum(0.0, np.sum(normals * s_dir, axis=1))
    return moon_albedo * mu0


def earthlight_if_tilecached(
    hit_points: np.ndarray,
    normals: np.ndarray,
    moon_center: np.ndarray,
    sun_pos: np.ndarray,
    earth_pos: np.ndarray,
    moon_albedo: float,
    earth_albedo: float,
    earth_radius_km: float,
    n_samples: int,
    tile_px: int,
    ij: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    """
    Compute earthlight contribution to I/F_moon in a tile-cached way.

    Output term is:
      I/F_moon += A_moon * E_earth / F_sun_at_moon
    where E_earth is the irradiance at lunar point due to Earth radiance.

    v0 assumptions:
      - Earth is Lambert sphere with constant albedo.
      - No atmosphere, no clouds, no terrain occlusion of Earth.
      - Within a tile, Earth disk directions and Earth radiance samples are computed at the tile representative point.
    """
    out = np.zeros(hit_points.shape[0], dtype=np.float32)

    sampler = EarthDiskSampler.create(n_samples)

    # Precompute F_sun at each lunar point (for normalisation) in units where F(1 AU)=1
    F_moon = np.array([inv_solar_irradiance_scale(hit_points[k], sun_pos) for k in range(hit_points.shape[0])], dtype=float)

    # Build a mapping from pixel -> index in hit_points array (only for points we have)
    # We'll tile in image coordinates over the ROI pixels provided in ij.
    # ij and hit_points/normals are aligned (same N).
    # Determine tile index for each pixel
    ti = ij[:, 0] // tile_px
    tj = ij[:, 1] // tile_px
    tile_id = ti + (nx // tile_px + 1) * tj

    unique_tiles = np.unique(tile_id)

    for tid in unique_tiles:
        idx = np.where(tile_id == tid)[0]
        if idx.size == 0:
            continue

        # choose representative point = first in tile
        k0 = int(idx[0])
        x0 = hit_points[k0]
        # geometric horizon test (very cheap): use radial normal from moon center
        radial0 = x0 - moon_center
        radial0 /= np.linalg.norm(radial0)
        e_dir0 = earth_pos - x0
        e_dir0 /= np.linalg.norm(e_dir0)
        if np.dot(radial0, e_dir0) <= 0.0:
            continue  # Earth below horizon for this tile (approx)

        # Earth angular radius as seen from x0
        d_em = float(np.linalg.norm(earth_pos - x0))
        alpha = np.arcsin(np.clip(earth_radius_km / d_em, 0.0, 1.0))

        omega, w = sampler.directions(e_dir0, alpha)  # (S,3), weight

        # Intersect these directions with Earth sphere to get surface points
        origins = np.repeat(x0[None, :], omega.shape[0], axis=0)
        hitE, tE = ray_sphere_intersect(origins, omega, earth_pos, earth_radius_km)
        if not np.any(hitE):
            continue
        pE = origins[hitE] + tE[hitE, None] * omega[hitE]

        nE = _normalize(pE - earth_pos[None, :])

        # Sun direction at Earth patch
        sE = _normalize(sun_pos[None, :] - pE)
        mu0E = np.maximum(0.0, np.sum(nE * sE, axis=1))

        # Visibility toward Moon (patch must face Moon): direction from patch to Moon is -omega
        omega_hit = omega[hitE]
        muE = np.maximum(0.0, np.sum(nE * (-omega_hit), axis=1))

        # Solar irradiance at Earth patch (F(1 AU)=1)
        F_E = np.array([inv_solar_irradiance_scale(pE[i], sun_pos) for i in range(pE.shape[0])], dtype=float)

        # Earth radiance toward Moon along omega (Lambert): L = (A/π) * F * mu0
        L = (earth_albedo / np.pi) * F_E * mu0E
        # apply facing factor
        L *= (muE > 0).astype(float)

        # Now for each lunar point in tile, compute E_earth = Σ L_k * max(0,n·omega_k) * w
        # We'll use the same omega directions and L samples for all pixels in tile.
        # But note L is only defined for hitE subset; keep aligned arrays.
        omega_use = omega_hit  # (M,3)
        L_use = L             # (M,)
        # dot = normals[idx] · omega_use
        dot = normals[idx] @ omega_use.T  # (P,M)
        cos_m = np.maximum(0.0, dot)

        # E_earth for each pixel in tile
        E = (cos_m * (L_use[None, :])).sum(axis=1) * w

        # Add to I/F_moon: A_moon * E / F_moon
        out[idx] = (moon_albedo * (E / F_moon[idx])).astype(np.float32)

    return out
