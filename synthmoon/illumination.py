from __future__ import annotations

import numpy as np
from dataclasses import dataclass
import spiceypy as sp

from .spice_tools import inv_solar_irradiance_scale
from .intersect import ray_sphere_intersect
from .albedo_maps import EquirectMap, toy_land_ocean_albedo, simple_cloud_fraction_field


def _normalize(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def cox_munk_glint_albedo_increment(
    normals: np.ndarray,
    sun_dir: np.ndarray,
    view_dir: np.ndarray,
    wind_m_s: float = 6.0,
    refractive_index: float = 1.334,
    strength: float = 1.0,
    max_albedo_increment: float = 2.0,
) -> np.ndarray:
    """
    Return an albedo-equivalent ocean glint increment using an isotropic Cox-Munk proxy.

    Notes
    -----
    - This returns an additive term to an albedo-like surface quantity.
    - The underlying microfacet BRDF proxy is converted to albedo-equivalent via A_eq = pi * f_r.
    - wind_m_s controls slope variance in a simple isotropic relation.
    """
    n = _normalize(np.asarray(normals, dtype=np.float64))
    s = _normalize(np.asarray(sun_dir, dtype=np.float64))
    v = _normalize(np.asarray(view_dir, dtype=np.float64))

    mu0 = np.clip(np.einsum("ij,ij->i", n, s), 0.0, 1.0)
    muv = np.clip(np.einsum("ij,ij->i", n, v), 0.0, 1.0)
    valid = (mu0 > 1.0e-6) & (muv > 1.0e-6)

    out = np.zeros(mu0.shape[0], dtype=np.float64)
    if not np.any(valid):
        return out

    h = _normalize(s + v)
    cos_beta = np.clip(np.einsum("ij,ij->i", n, h), 1.0e-8, 1.0)
    tan2_beta = np.clip((1.0 - cos_beta * cos_beta) / np.maximum(cos_beta * cos_beta, 1.0e-12), 0.0, 1.0e6)

    # Isotropic Cox-Munk-like slope variance (dimensionless).
    wind = float(max(wind_m_s, 0.0))
    sigma2 = float(max(1.0e-5, 0.003 + 0.00512 * wind))
    D = np.exp(-tan2_beta / sigma2) / (np.pi * sigma2 * np.maximum(cos_beta**4, 1.0e-12))

    # Fresnel for air-water interface (unpolarised).
    eta = float(max(refractive_index, 1.0e-3))
    cos_i = np.clip(np.einsum("ij,ij->i", s, h), 1.0e-8, 1.0)
    sin_t2 = (1.0 / (eta * eta)) * np.maximum(0.0, 1.0 - cos_i * cos_i)
    cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin_t2))
    rs = ((cos_i - eta * cos_t) / np.maximum(cos_i + eta * cos_t, 1.0e-12)) ** 2
    rp = ((eta * cos_i - cos_t) / np.maximum(eta * cos_i + cos_t, 1.0e-12)) ** 2
    F = 0.5 * (rs + rp)

    fr = (D * F) / np.maximum(4.0 * mu0 * muv, 1.0e-12)
    a_eq = np.pi * fr
    a_eq = np.clip(a_eq, 0.0, float(max_albedo_increment))
    out[valid] = float(strength) * a_eq[valid]
    return out


def _class_mask(values: np.ndarray, targets: list[float] | tuple[float, ...], tol: float = 0.5) -> np.ndarray:
    """Return boolean mask where values match any target class (within tolerance)."""
    x = np.asarray(values, dtype=np.float64)
    if not targets:
        return np.zeros(x.shape, dtype=bool)
    m = np.zeros(x.shape, dtype=bool)
    for t in targets:
        m |= np.abs(x - float(t)) <= float(tol)
    return m


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


@dataclass(frozen=True)
class HapkeParams:
    w: float = 0.55
    b: float = 0.30
    c: float = 0.40
    b0: float = 1.00
    h: float = 0.06
    theta_deg: float = 20.0


def _hapke_h(mu: np.ndarray, w: float) -> np.ndarray:
    g = np.sqrt(np.clip(1.0 - w, 1e-9, 1.0))
    return (1.0 + 2.0 * mu) / np.maximum(1.0 + 2.0 * mu * g, 1e-12)


def _hapke_p_hg2(cosg: np.ndarray, b: float, c: float) -> np.ndarray:
    b = float(np.clip(b, 0.0, 0.99))
    c = float(np.clip(c, 0.0, 1.0))
    p_f = (1.0 - b * b) / np.maximum((1.0 + 2.0 * b * cosg + b * b) ** 1.5, 1e-12)
    p_b = (1.0 - b * b) / np.maximum((1.0 - 2.0 * b * cosg + b * b) ** 1.5, 1e-12)
    return (1.0 - c) * p_f + c * p_b


def hapke_if(
    mu0: np.ndarray,
    mu: np.ndarray,
    phase_rad: np.ndarray,
    p: HapkeParams,
) -> np.ndarray:
    mu0 = np.clip(np.asarray(mu0, dtype=float), 0.0, 1.0)
    mu = np.clip(np.asarray(mu, dtype=float), 0.0, 1.0)
    g = np.clip(np.asarray(phase_rad, dtype=float), 0.0, np.pi)

    w = float(np.clip(p.w, 1e-6, 0.999999))
    h = float(max(p.h, 1e-6))
    b0 = float(max(p.b0, 0.0))

    cosg = np.cos(g)
    P = _hapke_p_hg2(cosg, p.b, p.c)
    B = b0 / (1.0 + np.tan(0.5 * g) / h)
    H0 = _hapke_h(mu0, w)
    H = _hapke_h(mu, w)

    # Lightweight macroscopic roughness attenuation.
    th = np.deg2rad(max(float(p.theta_deg), 0.0))
    rough = np.exp(-np.tan(th) ** 2 * (1.0 - mu0) * (1.0 - mu))

    iof = (w / 4.0) * (mu0 / np.maximum(mu0 + mu, 1e-12)) * (((1.0 + B) * P) + (H0 * H - 1.0)) * rough
    iof = np.where((mu0 > 0.0) & (mu > 0.0), iof, 0.0)
    return np.clip(iof, 0.0, None)


def hapke_sun_if(
    hit_points: np.ndarray,
    normals: np.ndarray,
    sun_pos: np.ndarray,
    obs_pos: np.ndarray,
    hapke: HapkeParams,
    moon_albedo_scale: np.ndarray | float = 1.0,
) -> np.ndarray:
    s_dir = _normalize(sun_pos[None, :] - hit_points)
    v_dir = _normalize(obs_pos[None, :] - hit_points)
    mu0 = np.maximum(0.0, np.sum(normals * s_dir, axis=1))
    mu = np.maximum(0.0, np.sum(normals * v_dir, axis=1))
    phase = np.arccos(np.clip(np.sum(s_dir * v_dir, axis=1), -1.0, 1.0))
    out = hapke_if(mu0, mu, phase, hapke)
    A = moon_albedo_scale if np.isscalar(moon_albedo_scale) else np.asarray(moon_albedo_scale, dtype=float)
    return out * A


def hapke_sun_if_extended_disk(
    hit_points: np.ndarray,
    normals: np.ndarray,
    moon_center: np.ndarray,
    sun_pos: np.ndarray,
    obs_pos: np.ndarray,
    hapke: HapkeParams,
    moon_albedo_scale: np.ndarray | float = 1.0,
    n_samples: int = 64,
    sun_radius_km: float = 695700.0,
) -> np.ndarray:
    N = hit_points.shape[0]
    out = np.zeros(N, dtype=np.float64)
    A = moon_albedo_scale if np.isscalar(moon_albedo_scale) else np.asarray(moon_albedo_scale, dtype=float)

    v = sun_pos[None, :] - hit_points
    d = np.linalg.norm(v, axis=1)
    s_dir = v / np.maximum(d[:, None], 1e-15)
    alpha = np.arcsin(np.clip(float(sun_radius_km) / np.maximum(d, 1e-9), 0.0, 1.0))
    sin_alpha = np.sin(alpha)

    radial = _normalize(hit_points - moon_center[None, :])
    v_dir = _normalize(obs_pos[None, :] - hit_points)
    mu = np.maximum(0.0, np.einsum("ij,ij->i", normals, v_dir))
    dotc = np.einsum("ij,ij->i", radial, s_dir)

    full = dotc >= sin_alpha
    none = dotc <= -sin_alpha
    partial = ~(full | none)

    if np.any(full):
        mu0 = np.maximum(0.0, np.einsum("ij,ij->i", normals[full], s_dir[full]))
        phase = np.arccos(np.clip(np.einsum("ij,ij->i", s_dir[full], v_dir[full]), -1.0, 1.0))
        out[full] = hapke_if(mu0, mu[full], phase, hapke) * (float(A) if np.isscalar(A) else A[full])

    if np.any(partial):
        sampler = EarthDiskSampler.create(int(n_samples))
        idxs = np.where(partial)[0]
        for k in idxs:
            omega, _w = sampler.directions(s_dir[k], float(alpha[k]))
            vis = (omega @ radial[k]) > 0.0
            if not np.any(vis):
                continue
            om = omega[vis]
            mu0k = np.maximum(0.0, om @ normals[k])
            muk = np.full(mu0k.shape, mu[k], dtype=float)
            phasek = np.arccos(np.clip(om @ v_dir[k], -1.0, 1.0))
            ifk = hapke_if(mu0k, muk, phasek, hapke)
            mean_if = float(np.mean(ifk))
            out[k] = mean_if * (float(A) if np.isscalar(A) else float(A[k]))

    return out

def lambert_sun_if_extended_disk(
    hit_points: np.ndarray,
    normals: np.ndarray,
    moon_center: np.ndarray,
    sun_pos: np.ndarray,
    moon_albedo: np.ndarray | float,
    n_samples: int = 64,
    sun_radius_km: float = 695700.0,
) -> np.ndarray:
    """
    Lambertian sunlight I/F treating the Sun as an extended disk (not a point).

    This matters only in a narrow band near the local horizon/terminator, where
    the Sun is partially visible. We handle that by sampling directions across the
    solar disk and applying a simple horizon mask (mean-sphere horizon):
        visible if dot(radial, omega) > 0

    For speed, pixels for which the full solar disk is certainly visible use the
    point-source approximation (lambert_sun_if). Pixels with no possible visibility
    return 0. Only the partial-visibility band is sampled.

    Parameters
    ----------
    hit_points : (N,3) J2000
    normals    : (N,3) J2000 unit normals (can include DEM slopes)
    moon_center: (3,)  J2000
    sun_pos    : (3,)  J2000 (SSB or same origin as hit_points)
    moon_albedo: scalar or (N,)
    n_samples  : number of disk samples (only used near horizon)
    sun_radius_km : physical solar radius in km

    Returns
    -------
    if_sun : (N,) I/F (dimensionless)
    """
    N = hit_points.shape[0]
    out = np.zeros(N, dtype=np.float64)

    A = moon_albedo if np.isscalar(moon_albedo) else np.asarray(moon_albedo, dtype=float)

    # Direction to Sun and angular radius of Sun as seen from each point
    v = sun_pos[None, :] - hit_points
    d = np.linalg.norm(v, axis=1)
    s_dir = v / np.maximum(d[:, None], 1e-15)

    # Sun angular radius alpha (rad). Clamp to avoid asin domain errors.
    alpha = np.arcsin(np.clip(float(sun_radius_km) / np.maximum(d, 1e-9), 0.0, 1.0))
    sin_alpha = np.sin(alpha)

    radial = _normalize(hit_points - moon_center[None, :])
    dotc = np.einsum("ij,ij->i", radial, s_dir)

    # Full disk visible vs none vs partial (geometric criterion)
    full = dotc >= sin_alpha
    none = dotc <= -sin_alpha
    partial = ~(full | none)

    if np.any(full):
        mu0 = np.maximum(0.0, np.einsum("ij,ij->i", normals[full], s_dir[full]))
        if np.isscalar(moon_albedo):
            out[full] = float(moon_albedo) * mu0
        else:
            out[full] = A[full] * mu0

    if np.any(partial):
        sampler = EarthDiskSampler.create(int(n_samples))
        idxs = np.where(partial)[0]
        for k in idxs:
            # sample directions across solar disk centred on s_dir[k]
            omega, _w = sampler.directions(s_dir[k], float(alpha[k]))

            # horizon mask (mean-sphere): above local tangent plane
            vis = (omega @ radial[k]) > 0.0

            # Lambert cosine, clamped, times visibility
            mu0k = np.maximum(0.0, omega @ normals[k]) * vis.astype(float)

            # average over full disk solid angle (uniform radiance disk)
            mu0_eff = float(mu0k.mean())

            if np.isscalar(moon_albedo):
                out[k] = float(moon_albedo) * mu0_eff
            else:
                out[k] = float(A[k]) * mu0_eff

    # 'none' remain 0
    return out


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
    earth_class_map: EquirectMap | None = None,
    earth_cloud_fraction_map: EquirectMap | None = None,
    earth_cloud_tau_map: EquirectMap | None = None,
    earth_ice_fraction_map: EquirectMap | None = None,
    earth_land_ice_mask_map: EquirectMap | None = None,
    earth_class_interp: str = "nearest",
    earth_class_ocean_values: list[float] | tuple[float, ...] = (0.0,),
    earth_class_land_values: list[float] | tuple[float, ...] = (1.0,),
    earth_class_ice_values: list[float] | tuple[float, ...] = (2.0,),
    earth_ocean_albedo: float = 0.06,
    earth_land_albedo: float = 0.25,
    earth_cloud_amount: float = 0.0,
    earth_cloud_albedo: float = 0.6,
    earth_cloud_tau: float = 1.0,
    earth_cloud_tau_k: float = 1.0,
    earth_map_interp: str = "nearest",
    earth_cloud_map_interp: str = "nearest",
    earth_ice_map_interp: str = "nearest",
    earth_ice_fraction_threshold: float = 0.15,
    earth_ice_fraction_blend: bool = True,
    earth_land_ice_interp: str = "nearest",
    earth_land_ice_threshold: float = 0.5,
    earth_land_ice_blend: bool = False,
    ocean_glint_model: str = "simple",
    ocean_glint_strength: float = 0.0,
    ocean_glint_sigma_deg: float = 6.0,
    ocean_glint_threshold: float = 0.12,
    ocean_wind_m_s: float = 6.0,
    ocean_refractive_index: float = 1.334,
    ocean_glint_max_albedo_increment: float = 2.0,
    earth_seasonal_ice: bool = True,
    earth_ice_albedo: float = 0.65,
    earth_ice_lat_north_base_deg: float = 72.0,
    earth_ice_lat_north_amp_deg: float = 8.0,
    earth_ice_lat_south_base_deg: float = 68.0,
    earth_ice_lat_south_amp_deg: float = 5.0,
    earth_ice_phase_north_doy: float = 20.0,
    earth_ice_phase_south_doy: float = 200.0,
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

    # UTC day-of-year for simple seasonal ice model.
    dt_utc = sp.et2datetime(float(et))
    doy = float(dt_utc.timetuple().tm_yday) + (
        dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0 + dt_utc.microsecond / 3.6e9
    ) / 24.0

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

        # Surface classes from optional class map (EO-style): ocean/land/ice IDs.
        cls = None
        ocean_cls = None
        land_cls = None
        ice_cls = None
        if earth_class_map is not None:
            cls = earth_class_map.sample_interp(lon, lat, interp=str(earth_class_interp))
            ocean_cls = _class_mask(cls, earth_class_ocean_values, tol=0.5)
            land_cls = _class_mask(cls, earth_class_land_values, tol=0.5)
            ice_cls = _class_mask(cls, earth_class_ice_values, tol=0.5)
            A_surface = np.full(lon.shape, float(earth_land_albedo), dtype=np.float64)
            if np.any(ocean_cls):
                A_surface[ocean_cls] = float(earth_ocean_albedo)
            if np.any(ice_cls):
                A_surface[ice_cls] = float(earth_ice_albedo)
            # Where class map may be undefined, fall back to albedo map or toy field.
            unknown = ~(ocean_cls | land_cls | ice_cls)
            if np.any(unknown):
                if earth_map is not None:
                    amap = earth_map.sample_interp(lon, lat, interp=str(earth_map_interp))
                else:
                    amap = toy_land_ocean_albedo(lon, lat, ocean=earth_ocean_albedo, land=earth_land_albedo)
                A_surface[unknown] = amap[unknown]
        else:
            # Base albedo for Earth surface patches
            if earth_map is not None:
                A_surface = earth_map.sample_interp(lon, lat, interp=str(earth_map_interp))
            else:
                A_surface = toy_land_ocean_albedo(lon, lat, ocean=earth_ocean_albedo, land=earth_land_albedo)

        # Optional EO ice-fraction map overrides ocean pixels with ice albedo.
        if earth_ice_fraction_map is not None:
            ice_frac = np.asarray(
                earth_ice_fraction_map.sample_interp(lon, lat, interp=str(earth_ice_map_interp)),
                dtype=np.float64,
            )
            ice_frac = np.clip(ice_frac, 0.0, 1.0)
            if ocean_cls is not None:
                ocean_for_ice = ocean_cls.copy()
            else:
                ocean_for_ice = A_surface <= max(float(earth_ocean_albedo), float(ocean_glint_threshold))
            if bool(earth_ice_fraction_blend):
                w_ice = np.where(ocean_for_ice, ice_frac, 0.0)
                A_surface = (1.0 - w_ice) * A_surface + w_ice * float(earth_ice_albedo)
            else:
                ice_mask = ocean_for_ice & (ice_frac >= float(earth_ice_fraction_threshold))
                if np.any(ice_mask):
                    A_surface = np.array(A_surface, copy=True, dtype=np.float64)
                    A_surface[ice_mask] = float(earth_ice_albedo)

        # Optional static land-ice mask overrides land pixels with ice albedo.
        if earth_land_ice_mask_map is not None:
            land_ice = np.asarray(
                earth_land_ice_mask_map.sample_interp(lon, lat, interp=str(earth_land_ice_interp)),
                dtype=np.float64,
            )
            land_ice = np.clip(land_ice, 0.0, 1.0)
            if cls is not None:
                land_for_ice = land_cls | ice_cls
            elif ocean_cls is not None:
                land_for_ice = ~ocean_cls
            else:
                land_for_ice = A_surface > max(float(earth_ocean_albedo), float(ocean_glint_threshold))
            if bool(earth_land_ice_blend):
                w_land_ice = np.where(land_for_ice, land_ice, 0.0)
                A_surface = (1.0 - w_land_ice) * A_surface + w_land_ice * float(earth_ice_albedo)
            else:
                land_ice_mask = land_for_ice & (land_ice >= float(earth_land_ice_threshold))
                if np.any(land_ice_mask):
                    A_surface = np.array(A_surface, copy=True, dtype=np.float64)
                    A_surface[land_ice_mask] = float(earth_ice_albedo)

        omega_hit = omega[hitE]

        # Sun illumination at Earth patch (Earth phases)
        sE = _normalize(sun_pos[None, :] - pE)
        mu0E = np.maximum(0.0, np.sum(nE * sE, axis=1))

        # Patch must face the Moon: direction from patch to Moon is -omega
        muE = np.maximum(0.0, np.sum(nE * (-omega_hit), axis=1))

        # Simple seasonal polar-cap override (latitude + day-of-year).
        if bool(earth_seasonal_ice):
            ph_n = 2.0 * np.pi * (doy - float(earth_ice_phase_north_doy)) / 365.25
            ph_s = 2.0 * np.pi * (doy - float(earth_ice_phase_south_doy)) / 365.25
            lat_n_edge = float(earth_ice_lat_north_base_deg) + float(earth_ice_lat_north_amp_deg) * np.cos(ph_n)
            lat_s_edge = float(earth_ice_lat_south_base_deg) + float(earth_ice_lat_south_amp_deg) * np.cos(ph_s)
            lat_n_edge = float(np.clip(lat_n_edge, 45.0, 89.0))
            lat_s_edge = float(np.clip(lat_s_edge, 45.0, 89.0))
            ice_mask = (lat >= lat_n_edge) | (lat <= -lat_s_edge)
            if ice_cls is not None:
                ice_mask = ice_mask | ice_cls
            if np.any(ice_mask):
                A_surface = np.array(A_surface, copy=True, dtype=np.float64)
                A_surface[ice_mask] = float(earth_ice_albedo)

        # Optional ocean glint (simple gaussian or Cox-Munk proxy).
        glint_strength = float(ocean_glint_strength)
        if glint_strength > 0.0:
            ocean_thresh = float(ocean_glint_threshold)
            if ocean_cls is not None:
                ocean = ocean_cls & (mu0E > 0.0) & (muE > 0.0)
            else:
                ocean = (A_surface < ocean_thresh) & (mu0E > 0.0) & (muE > 0.0)
            if np.any(ocean):
                model = str(ocean_glint_model).strip().lower()
                if model in ("cox_munk", "coxmunk", "cox-munk"):
                    glint = cox_munk_glint_albedo_increment(
                        normals=nE,
                        sun_dir=sE,
                        view_dir=(-omega_hit),
                        wind_m_s=float(ocean_wind_m_s),
                        refractive_index=float(ocean_refractive_index),
                        strength=glint_strength,
                        max_albedo_increment=float(ocean_glint_max_albedo_increment),
                    )
                else:
                    sigma_deg = float(ocean_glint_sigma_deg)
                    sigma = np.deg2rad(max(sigma_deg, 0.1))
                    h = _normalize(sE + (-omega_hit))
                    cos_th = np.clip(np.sum(nE * h, axis=1), -1.0, 1.0)
                    theta = np.arccos(cos_th)
                    glint = glint_strength * np.exp(- (theta / sigma) ** 2)
                A_surface = A_surface + glint * ocean.astype(float)

        # Cloud/surface blend:
        #   A_eff = (1-fc_local) * A_surface + fc_local * A_cloud
        # with fc_local = cloud_field * cloud_amount * (1-exp(-k*tau)).
        cloud_amt = float(np.clip(earth_cloud_amount, 0.0, 1.0))
        if cloud_amt > 0.0:
            tau_k = float(max(earth_cloud_tau_k, 0.0))
            if earth_cloud_fraction_map is not None:
                cloud_field = np.asarray(
                    earth_cloud_fraction_map.sample_interp(lon, lat, interp=str(earth_cloud_map_interp)),
                    dtype=np.float64,
                )
                cloud_field = np.clip(cloud_field, 0.0, 1.0)
            else:
                cloud_field = simple_cloud_fraction_field(lon, lat)

            if earth_cloud_tau_map is not None:
                tau_local = np.asarray(
                    earth_cloud_tau_map.sample_interp(lon, lat, interp=str(earth_cloud_map_interp)),
                    dtype=np.float64,
                )
                tau_local = np.maximum(tau_local, 0.0)
            else:
                tau_local = np.full(lon.shape, float(max(earth_cloud_tau, 0.0)), dtype=np.float64)

            trans = 1.0 - np.exp(-tau_k * tau_local)
            fc_local = np.clip(cloud_amt * trans * cloud_field, 0.0, 1.0)
            A_E = (1.0 - fc_local) * A_surface + fc_local * float(earth_cloud_albedo)
        else:
            A_E = A_surface

        # Clamp and allow a global multiplier via earth_albedo (acts as a scale)
        A_E = np.clip(A_E, 0.0, 1.0) * float(earth_albedo)

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


def earthlight_if_point_source(
    hit_points: np.ndarray,
    normals: np.ndarray,
    moon_center: np.ndarray,
    sun_pos: np.ndarray,
    earth_pos: np.ndarray,
    moon_albedo: np.ndarray | float,
    earth_albedo: float,
    earth_radius_km: float,
) -> np.ndarray:
    """Point-source Earthshine approximation.

    This is a true point-Earth model (no Earth-disk quadrature): Earthshine arrives
    from a single direction (Earth center), with brightness scaled by Lambert-sphere
    phase and inverse-square range.
    """
    N = hit_points.shape[0]
    out = np.zeros(N, dtype=np.float64)

    # Direction to Earth and visibility from each lunar point.
    e_vec = earth_pos[None, :] - hit_points
    d_em = np.linalg.norm(e_vec, axis=1)
    e_dir = e_vec / np.maximum(d_em[:, None], 1e-15)
    radial = _normalize(hit_points - moon_center[None, :])
    vis = np.einsum("ij,ij->i", radial, e_dir) > 0.0
    if not np.any(vis):
        return out

    # Lunar incidence from the Earth direction.
    mu_m = np.maximum(0.0, np.einsum("ij,ij->i", normals, e_dir))

    # Earth phase angle as seen from the Moon point.
    s_e = _normalize(sun_pos[None, :] - earth_pos[None, :])  # (1,3)
    m_e = _normalize(hit_points - earth_pos[None, :])        # (N,3)
    cos_g = np.clip(np.einsum("ij,ij->i", np.repeat(s_e, N, axis=0), m_e), -1.0, 1.0)
    g = np.arccos(cos_g)

    # Lambert-sphere phase function (normalised so Phi(0)=1).
    phi = (np.sin(g) + (np.pi - g) * np.cos(g)) / np.pi
    phi = np.clip(phi, 0.0, 1.0)

    # Solar irradiance scaling at Earth and at Moon points (F(1 AU)=1).
    f_e = float(inv_solar_irradiance_scale(earth_pos, sun_pos))
    f_m = np.array([inv_solar_irradiance_scale(hit_points[k], sun_pos) for k in range(N)], dtype=float)

    # Irradiance on Moon facet from point-Earth source:
    # E = (2/3) * A_E * F_E * Phi(g) * (R_E/d)^2 * mu_m
    geom = (float(earth_radius_km) / np.maximum(d_em, 1e-12)) ** 2
    E = (2.0 / 3.0) * float(earth_albedo) * f_e * phi * geom * mu_m
    E = np.where(vis, E, 0.0)

    if np.isscalar(moon_albedo):
        A_m = float(moon_albedo)
        out = A_m * (E / np.maximum(f_m, 1e-12))
    else:
        A_m = np.asarray(moon_albedo, dtype=float)
        out = A_m * (E / np.maximum(f_m, 1e-12))

    return out
