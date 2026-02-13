from __future__ import annotations

import numpy as np
from dataclasses import dataclass
import spiceypy as sp

from .spice_tools import inv_solar_irradiance_scale
from .intersect import ray_sphere_intersect
from .albedo_maps import EquirectMap, toy_land_ocean_albedo, apply_simple_clouds


def _normalize(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


@dataclass(frozen=True)
class EarthDiskSampler:
    n_samples: int
    u: np.ndarray
    v: np.ndarray

    @staticmethod
    def create(n_samples: int) -> "EarthDiskSampler":
        i = np.arange(n_samples, dtype=float)
        u = (i * 0.6180339887498949) % 1.0
        v = (i * 0.7548776662466927) % 1.0
        return EarthDiskSampler(n_samples=n_samples, u=u, v=v)

    def directions(self, e_dir: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
        e = e_dir / np.linalg.norm(e_dir)
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

        omega = (cos_t[:, None] * e[None, :]) + (
            sin_t[:, None] * (np.cos(phi)[:, None] * u_hat[None, :] + np.sin(phi)[:, None] * v_hat[None, :])
        )
        omega = _normalize(omega)

        Omega = 2.0 * np.pi * (1.0 - cos_alpha)
        w = Omega / float(self.n_samples)
        return omega, float(w)


def lambert_sun_if(hit_points: np.ndarray, normals: np.ndarray, sun_pos: np.ndarray, moon_albedo: np.ndarray | float) -> np.ndarray:
    s_dir = _normalize(sun_pos[None, :] - hit_points)
    mu0 = np.maximum(0.0, np.sum(normals * s_dir, axis=1))
    A = moon_albedo if np.isscalar(moon_albedo) else np.asarray(moon_albedo, dtype=float)
    return A * mu0


def earthlight_if_tilecached(
    hit_points: np.ndarray,
    normals: np.ndarray,
    moon_center: np.ndarray,
    sun_pos: np.ndarray,
    earth_pos: np.ndarray,
    et: float,
    moon_albedo: np.ndarray | float,
    earth_albedo: float,
    earth_radius_km: float,
    n_samples: int,
    tile_px: int,
    ij: np.ndarray,
    nx: int,
    ny: int,
    earth_map: EquirectMap | None = None,
    earth_ocean_albedo: float = 0.06,
    earth_land_albedo: float = 0.25,
    earth_cloud_amount: float = 0.0,
    earth_cloud_albedo: float = 0.6,
) -> np.ndarray:
    """
    Earthlight with partial-disk visibility near the lunar horizon + optional map-based Earth albedo.

    If earth_map is provided, it is sampled in IAU_EARTH lon/lat.
    Otherwise a toy land/ocean pattern is used (still useful for debugging).
    """
    out = np.zeros(hit_points.shape[0], dtype=np.float64)

    sampler = EarthDiskSampler.create(n_samples)

    # Solar irradiance scale at each lunar point for I/F normalisation (F(1 AU)=1)
    F_moon = np.array([inv_solar_irradiance_scale(hit_points[k], sun_pos) for k in range(hit_points.shape[0])], dtype=float)

    # Body-fixed transform for Earth (used for map + clouds)
    Mj2e = sp.pxform("J2000", "IAU_EARTH", et)

    ti = ij[:, 0] // tile_px
    tj = ij[:, 1] // tile_px
    tile_id = ti + (nx // tile_px + 1) * tj
    unique_tiles = np.unique(tile_id)

    for tid in unique_tiles:
        idx_all = np.where(tile_id == tid)[0]
        if idx_all.size == 0:
            continue

        k0 = int(idx_all[0])
        x0 = hit_points[k0]

        d_em0 = float(np.linalg.norm(earth_pos - x0))
        alpha0 = np.arcsin(np.clip(earth_radius_km / d_em0, 0.0, 1.0))
        sin_alpha0 = float(np.sin(alpha0))

        radial = _normalize(hit_points[idx_all] - moon_center[None, :])
        e_dir = _normalize(earth_pos[None, :] - hit_points[idx_all])
        dot0 = np.einsum("ij,ij->i", radial, e_dir)

        vis_any = dot0 > (-sin_alpha0)
        if not np.any(vis_any):
            continue

        idx = idx_all[vis_any]

        e_dir0 = earth_pos - x0
        e_dir0 /= np.linalg.norm(e_dir0)
        omega, w = sampler.directions(e_dir0, alpha0)

        origins = np.repeat(x0[None, :], omega.shape[0], axis=0)
        hitE, tE = ray_sphere_intersect(origins, omega, earth_pos, earth_radius_km)
        if not np.any(hitE):
            continue

        pE = origins[hitE] + tE[hitE, None] * omega[hitE]
        nE = _normalize(pE - earth_pos[None, :])

        # Earth patch lon/lat in IAU_EARTH
        vE = pE - earth_pos[None, :]
        vEf = (Mj2e @ vE.T).T
        r = np.linalg.norm(vEf, axis=1)
        lon = np.rad2deg(np.arctan2(vEf[:, 1], vEf[:, 0]))
        lat = np.rad2deg(np.arcsin(np.clip(vEf[:, 2] / np.maximum(r, 1e-15), -1.0, 1.0)))

        # Base albedo for Earth patches
        if earth_map is not None:
            A_E = earth_map.sample(lon, lat)
        else:
            A_E = toy_land_ocean_albedo(lon, lat, ocean=earth_ocean_albedo, land=earth_land_albedo)

        if earth_cloud_amount > 0.0:
            A_E = apply_simple_clouds(A_E, lon, lat, cloud_amount=earth_cloud_amount, cloud_albedo=earth_cloud_albedo)

        # Clamp and allow a global multiplier via earth_albedo (acts as a scale)
        A_E = np.clip(A_E, 0.0, 1.0) * float(earth_albedo)

        # Sun illumination at Earth patch (Earth phases)
        sE = _normalize(sun_pos[None, :] - pE)
        mu0E = np.maximum(0.0, np.sum(nE * sE, axis=1))

        # Patch must face the Moon: direction from patch to Moon is -omega
        omega_hit = omega[hitE]
        muE = np.maximum(0.0, np.sum(nE * (-omega_hit), axis=1))

        # Solar irradiance at Earth patch (F(1 AU)=1)
        F_E = np.array([inv_solar_irradiance_scale(pE[i], sun_pos) for i in range(pE.shape[0])], dtype=float)

        # Earth radiance toward Moon along omega (Lambert): L = (A/π) * F * mu0
        L = (A_E / np.pi) * F_E * mu0E
        L *= (muE > 0).astype(float)

        omega_use = omega_hit
        L_use = L

        dot_inc = normals[idx] @ omega_use.T
        cos_m = np.maximum(0.0, dot_inc)

        # Radial horizon mask for partial Earth visibility
        radial_idx = _normalize(hit_points[idx] - moon_center[None, :])
        dot_h = radial_idx @ omega_use.T
        cos_m = np.where(dot_h > 0.0, cos_m, 0.0)

        E = (cos_m * (L_use[None, :])).sum(axis=1) * w

        if np.isscalar(moon_albedo):
            A_m = float(moon_albedo)
            out[idx] = A_m * (E / F_moon[idx])
        else:
            A_m = np.asarray(moon_albedo, dtype=float)
            out[idx] = A_m[idx] * (E / F_moon[idx])

    return out
