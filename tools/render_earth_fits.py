#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import spiceypy as sp
from astropy.time import Time
from astropy.io import fits

try:
    from synthmoon.config import load_config
    from synthmoon.spice_tools import (
        load_kernels,
        load_optional_moon_frame_kernels,
        utc_to_et,
        earth_site_state_j2000_earthcenter,
        get_sun_earth_moon_states_ssb,
        inv_solar_irradiance_scale,
    )
    from synthmoon.fits_io import write_fits_cube
    from synthmoon.albedo_maps import EquirectMap, toy_land_ocean_albedo, simple_cloud_fraction_field
    from synthmoon.illumination import cox_munk_glint_albedo_increment
    from synthmoon.earth_rgb_physical import (
        earth_color_model,
        resolve_class_rgb_table,
        resolve_cloud_rgb,
        resolve_default_surface_rgb,
        resolve_ice_rgb,
        simple_brdf_factor,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from synthmoon.config import load_config
    from synthmoon.spice_tools import (
        load_kernels,
        load_optional_moon_frame_kernels,
        utc_to_et,
        earth_site_state_j2000_earthcenter,
        get_sun_earth_moon_states_ssb,
        inv_solar_irradiance_scale,
    )
    from synthmoon.fits_io import write_fits_cube
    from synthmoon.albedo_maps import EquirectMap, toy_land_ocean_albedo, simple_cloud_fraction_field
    from synthmoon.illumination import cox_munk_glint_albedo_increment
    from synthmoon.earth_rgb_physical import (
        earth_color_model,
        resolve_class_rgb_table,
        resolve_cloud_rgb,
        resolve_default_surface_rgb,
        resolve_ice_rgb,
        simple_brdf_factor,
    )


def _normalize(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def _basis_from_forward_up(forward: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = _normalize(np.asarray(forward, dtype=float))
    u = np.asarray(up_hint, dtype=float)
    u = u - np.dot(u, f) * f
    nu = float(np.linalg.norm(u))
    if nu < 1e-12:
        x = np.array([1.0, 0.0, 0.0], dtype=float)
        u = x - np.dot(x, f) * f
        nu = float(np.linalg.norm(u))
        if nu < 1e-12:
            y = np.array([0.0, 1.0, 0.0], dtype=float)
            u = y - np.dot(y, f) * f
            nu = float(np.linalg.norm(u))
    u /= max(nu, 1e-12)
    r = np.cross(u, f)
    r /= max(float(np.linalg.norm(r)), 1e-12)
    u = np.cross(f, r)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    return r, u, f


def _class_mask(values: np.ndarray, targets: list[float] | tuple[float, ...], tol: float = 0.5) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if not targets:
        return np.zeros(x.shape, dtype=bool)
    m = np.zeros(x.shape, dtype=bool)
    for t in targets:
        m |= np.abs(x - float(t)) <= float(tol)
    return m


def _parse_rgb_triplet(value: object, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(float(np.clip(v, 0.0, 1.0)) for v in value)
    return default


def _physical_layer_sum_cards(layers: dict[str, tuple[np.ndarray, str]]) -> dict[str, tuple[float, str]]:
    key_map = {
        "RAD_R": "SUMRADR",
        "RAD_G": "SUMRADG",
        "RAD_B": "SUMRADB",
    }
    out: dict[str, tuple[float, str]] = {}
    for layer_name, (arr, unit) in layers.items():
        hdr_key = key_map.get(layer_name)
        if hdr_key is None or "W m-2" not in str(unit):
            continue
        out[hdr_key] = (float(np.nansum(np.asarray(arr, dtype=np.float64))), f"Sum of {layer_name}")
    return out


def _class_rgb_table(earth_cfg: dict) -> dict[int, tuple[float, float, float]]:
    return resolve_class_rgb_table(earth_cfg)


def _rgb_from_class_ids(
    class_id: np.ndarray,
    *,
    table: dict[int, tuple[float, float, float]],
    default_rgb: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.empty((class_id.shape[0], 3), dtype=np.float64)
    rgb[:, 0] = default_rgb[0]
    rgb[:, 1] = default_rgb[1]
    rgb[:, 2] = default_rgb[2]
    known = np.zeros(class_id.shape, dtype=bool)
    for cls_value, triplet in table.items():
        m = np.abs(class_id - float(cls_value)) <= 0.5
        if not np.any(m):
            continue
        rgb[m, 0] = triplet[0]
        rgb[m, 1] = triplet[1]
        rgb[m, 2] = triplet[2]
        known[m] = True
    return rgb, known


def _rayleigh_phase(cos_theta: np.ndarray) -> np.ndarray:
    c2 = np.clip(cos_theta, -1.0, 1.0) ** 2
    return (3.0 / (16.0 * np.pi)) * (1.0 + c2)


def _hg_phase(cos_theta: np.ndarray, g: float) -> np.ndarray:
    gg = float(np.clip(g, -0.95, 0.95))
    denom = np.maximum(1.0 + gg * gg - 2.0 * gg * np.clip(cos_theta, -1.0, 1.0), 1.0e-8)
    return ((1.0 - gg * gg) / (4.0 * np.pi)) / np.power(denom, 1.5)


def _apply_earth_atmosphere(
    *,
    earth_cfg: dict,
    fsun: np.ndarray,
    mu0: np.ndarray,
    muv: np.ndarray,
    s: np.ndarray,
    vobs: np.ndarray,
    rad_surface_rgb: np.ndarray,
    if_surface_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not bool(earth_cfg.get("atmosphere_enable", True)):
        z = np.zeros_like(rad_surface_rgb, dtype=np.float64)
        return rad_surface_rgb, if_surface_rgb, z, np.ones_like(rad_surface_rgb, dtype=np.float64), z

    mu0e = np.maximum(mu0, float(earth_cfg.get("atmosphere_mu_floor", 0.03)))
    muve = np.maximum(muv, float(earth_cfg.get("atmosphere_mu_floor", 0.03)))
    sec_sun = 1.0 / mu0e
    sec_view = 1.0 / muve

    tau_ray = np.array(
        _parse_rgb_triplet(earth_cfg.get("rayleigh_tau_rgb", [0.18, 0.10, 0.05]), (0.18, 0.10, 0.05)),
        dtype=np.float64,
    )
    tau_aer = np.array(
        _parse_rgb_triplet(earth_cfg.get("aerosol_tau_rgb", [0.04, 0.04, 0.035]), (0.04, 0.04, 0.035)),
        dtype=np.float64,
    )
    tau_tot = tau_ray + tau_aer

    trans_sun = np.exp(-sec_sun[:, None] * tau_tot[None, :])
    trans_view = np.exp(-sec_view[:, None] * tau_tot[None, :])
    trans_total = trans_sun * trans_view

    cos_scatter = np.einsum("ij,ij->i", s, vobs)
    ph_ray = _rayleigh_phase(cos_scatter)
    ph_aer = _hg_phase(cos_scatter, float(earth_cfg.get("aerosol_g", 0.70)))

    path_scale = 1.0 - np.exp(-(sec_sun[:, None] + sec_view[:, None]) * tau_tot[None, :])
    limb_boost_pow = float(max(earth_cfg.get("atmosphere_limb_boost_power", 0.15), 0.0))
    limb_boost = np.power(np.clip(1.0 / muve, 1.0, None), limb_boost_pow)[:, None]
    twilight_mu = float(max(earth_cfg.get("atmosphere_twilight_mu", 0.12), 1.0e-4))
    sunlit_weight = np.clip(mu0 / twilight_mu, 0.0, 1.0)[:, None]

    ray_frac = tau_ray[None, :] / np.maximum(tau_tot[None, :], 1.0e-12)
    aer_frac = tau_aer[None, :] / np.maximum(tau_tot[None, :], 1.0e-12)

    phase_mix = ray_frac * ph_ray[:, None] + aer_frac * ph_aer[:, None]
    sky_rgb = np.array(
        _parse_rgb_triplet(earth_cfg.get("atmosphere_sky_rgb", [0.60, 0.72, 1.00]), (0.60, 0.72, 1.00)),
        dtype=np.float64,
    )
    atm_strength = float(max(earth_cfg.get("atmosphere_strength", 1.0), 0.0))
    path_rad_rgb = atm_strength * (fsun[:, None] / np.pi) * path_scale * phase_mix * sky_rgb[None, :] * limb_boost * sunlit_weight

    rad_total_rgb = rad_surface_rgb * trans_total + path_rad_rgb
    if_total_rgb = if_surface_rgb * trans_total + (path_rad_rgb * np.pi) / np.maximum(fsun[:, None], 1.0e-12)
    return rad_total_rgb, if_total_rgb, path_rad_rgb, trans_total, trans_view


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render synthetic Earth disk to FITS using synthmoon Earth model")
    ap.add_argument("--config", default="scene.toml", help="Path to scene.toml")
    ap.add_argument("--utc", default=None, help="UTC override, e.g. 2006-02-17T06:18:45Z")
    ap.add_argument("--jd", type=float, default=None, help="Julian Day UTC (alternative to --utc)")
    ap.add_argument("--out", default=None, help="Output FITS path")
    ap.add_argument("--nx", type=int, default=1024, help="Output width")
    ap.add_argument("--ny", type=int, default=1024, help="Output height")
    ap.add_argument("--only-layer-index", type=int, default=None, help="If set to N>0, write only the Nth layer (1-based)")
    ap.add_argument(
        "--view",
        default="moon_center",
        choices=["moon_center", "earth_site"],
        help="Observer viewpoint for Earth image",
    )
    ap.add_argument("--lon", type=float, default=None, help="Observer lon deg (used with --view earth_site)")
    ap.add_argument("--lat", type=float, default=None, help="Observer lat deg (used with --view earth_site)")
    ap.add_argument("--alt-m", type=float, default=None, help="Observer height m (used with --view earth_site)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.utc and args.jd is not None:
        raise SystemExit("Use either --utc or --jd, not both.")

    utc = str(args.utc or cfg.utc)
    if args.jd is not None:
        utc = Time(float(args.jd), format="jd", scale="utc").isot + "Z"
    jd = Time(utc, format="isot", scale="utc").jd

    kernels_dir = str(cfg.paths.get("spice_kernels_dir", "KERNELS"))
    mk = cfg.paths.get("meta_kernel", str(Path(kernels_dir) / "generic.tm"))
    load_kernels(mk)
    auto_fk = bool(cfg.paths.get("auto_download_small_kernels", False) or cfg.moon.get("auto_download_small_fk", False))
    load_optional_moon_frame_kernels(kernels_dir, auto_download_small_fk=auto_fk)

    et = utc_to_et(utc)
    states = get_sun_earth_moon_states_ssb(et)
    sun_pos = states["SUN"][:3]
    earth_state_ssb = states["EARTH"]
    earth_pos = earth_state_ssb[:3]
    moon_pos = states["MOON"][:3]

    # Viewpoint
    if args.view == "moon_center":
        obs_pos = moon_pos
    else:
        obs_cfg = cfg.observer
        lon = float(args.lon if args.lon is not None else obs_cfg.get("lon_deg", 0.0))
        lat = float(args.lat if args.lat is not None else obs_cfg.get("lat_deg", 0.0))
        alt_m = float(args.alt_m if args.alt_m is not None else obs_cfg.get("height_m", 0.0))
        obs_state_ec = earth_site_state_j2000_earthcenter(et, lon, lat, alt_m)
        obs_pos = (obs_state_ec + earth_state_ssb)[:3]

    # Camera basis on the sky plane of the observer.
    # Visible hemisphere is centred on the Earth-center -> observer direction.
    # Using the opposite sign would render the far side of Earth and invert the
    # expected Earth/Moon phase complementarity.
    forward = obs_pos - earth_pos
    # Earth north axis in J2000 to orient image up.
    r_e2j = np.array(sp.pxform("IAU_EARTH", "J2000", float(et)), dtype=float)
    earth_north_j2k = r_e2j @ np.array([0.0, 0.0, 1.0], dtype=float)
    right, up, fwd = _basis_from_forward_up(forward, earth_north_j2k)

    nx = int(args.nx)
    ny = int(args.ny)
    if nx <= 0 or ny <= 0:
        raise SystemExit("nx/ny must be positive")

    # Orthographic disk grid.
    x = (np.arange(nx, dtype=np.float64) + 0.5 - 0.5 * nx) / (0.5 * nx)
    y = (np.arange(ny, dtype=np.float64) + 0.5 - 0.5 * ny) / (0.5 * ny)
    xx, yy = np.meshgrid(x, y)
    rr2 = xx * xx + yy * yy
    mask = rr2 <= 1.0
    zz = np.sqrt(np.clip(1.0 - rr2, 0.0, 1.0))

    # Visible-sphere normals in J2000.
    n = np.zeros((ny, nx, 3), dtype=np.float64)
    n[..., :] = (xx[..., None] * right[None, None, :]) + (yy[..., None] * up[None, None, :]) + (zz[..., None] * fwd[None, None, :])
    n = _normalize(n)

    re_km = float(cfg.earth.get("radius_km", 6378.1366))
    p = earth_pos[None, None, :] + re_km * n

    pv = p[mask]
    nv = n[mask]

    # Earth lon/lat in IAU_EARTH.
    Mj2e = np.array(sp.pxform("J2000", "IAU_EARTH", float(et)), dtype=float)
    vf = (Mj2e @ (pv - earth_pos[None, :]).T).T
    r = np.linalg.norm(vf, axis=1)
    lon = np.rad2deg(np.arctan2(vf[:, 1], vf[:, 0]))
    lat = np.rad2deg(np.arcsin(np.clip(vf[:, 2] / np.maximum(r, 1e-15), -1.0, 1.0)))

    earth_map = None
    earth_map_path = cfg.earth.get("albedo_map_fits", None)
    if earth_map_path:
        earth_lon_mode = str(cfg.earth.get("albedo_map_lon_mode", "0_360"))
        earth_map = EquirectMap.load_fits(
            earth_map_path,
            lon_mode=earth_lon_mode,
            fill=float(cfg.earth.get("albedo_map_fill", 0.30)),
        )
    earth_class_map = None
    earth_class_map_path = cfg.earth.get("class_map_fits", None)
    if earth_class_map_path:
        earth_class_map_path = str(earth_class_map_path)
        if not Path(earth_class_map_path).exists():
            earth_class_map_path = None
        else:
            earth_class_lon_mode = str(cfg.earth.get("class_map_lon_mode", str(cfg.earth.get("albedo_map_lon_mode", "0_360"))))
            earth_class_map = EquirectMap.load_fits(
                earth_class_map_path,
                lon_mode=earth_class_lon_mode,
                fill=float(cfg.earth.get("class_map_fill", np.nan)),
            )
    class_interp = str(cfg.earth.get("class_map_interp", "nearest"))
    class_ocean_values = tuple(float(v) for v in cfg.earth.get("class_ocean_values", [0]))
    class_land_values = tuple(float(v) for v in cfg.earth.get("class_land_values", [1]))
    class_ice_values = tuple(float(v) for v in cfg.earth.get("class_ice_values", [2]))

    earth_cloud_fraction_map = None
    earth_cloud_fraction_map_path = cfg.earth.get("cloud_fraction_map_fits", None)
    if earth_cloud_fraction_map_path:
        earth_cloud_lon_mode = str(cfg.earth.get("cloud_map_lon_mode", str(cfg.earth.get("albedo_map_lon_mode", "0_360"))))
        earth_cloud_fraction_map = EquirectMap.load_fits(
            earth_cloud_fraction_map_path,
            lon_mode=earth_cloud_lon_mode,
            fill=float(cfg.earth.get("cloud_fraction_map_fill", np.nan)),
        )

    earth_cloud_tau_map = None
    earth_cloud_tau_map_path = cfg.earth.get("cloud_tau_map_fits", None)
    if earth_cloud_tau_map_path:
        earth_cloud_lon_mode = str(cfg.earth.get("cloud_map_lon_mode", str(cfg.earth.get("albedo_map_lon_mode", "0_360"))))
        earth_cloud_tau_map = EquirectMap.load_fits(
            earth_cloud_tau_map_path,
            lon_mode=earth_cloud_lon_mode,
            fill=float(cfg.earth.get("cloud_tau_map_fill", np.nan)),
        )
    cloud_interp = str(cfg.earth.get("cloud_map_interp", "nearest"))

    earth_ice_fraction_map = None
    earth_ice_fraction_map_path = cfg.earth.get("ice_fraction_map_fits", None)
    if earth_ice_fraction_map_path:
        earth_ice_lon_mode = str(cfg.earth.get("ice_map_lon_mode", str(cfg.earth.get("albedo_map_lon_mode", "0_360"))))
        earth_ice_fraction_map = EquirectMap.load_fits(
            earth_ice_fraction_map_path,
            lon_mode=earth_ice_lon_mode,
            fill=float(cfg.earth.get("ice_fraction_map_fill", np.nan)),
        )
    ice_interp = str(cfg.earth.get("ice_map_interp", "nearest"))

    earth_land_ice_mask_map = None
    earth_land_ice_mask_map_alt = None
    earth_land_ice_mask_path = cfg.earth.get("land_ice_mask_fits", None)
    if earth_land_ice_mask_path:
        earth_land_ice_lon_mode = str(cfg.earth.get("land_ice_mask_lon_mode", str(cfg.earth.get("albedo_map_lon_mode", "0_360"))))
        earth_land_ice_mask_map = EquirectMap.load_fits(
            earth_land_ice_mask_path,
            lon_mode=earth_land_ice_lon_mode,
            fill=float(cfg.earth.get("land_ice_mask_fill", np.nan)),
        )
        alt_lon_mode = "0_360" if earth_land_ice_lon_mode == "-180_180" else "-180_180"
        earth_land_ice_mask_map_alt = EquirectMap.load_fits(
            earth_land_ice_mask_path,
            lon_mode=alt_lon_mode,
            fill=float(cfg.earth.get("land_ice_mask_fill", np.nan)),
        )
    land_ice_interp = str(cfg.earth.get("land_ice_mask_interp", "nearest"))

    ocean_cls = None
    land_cls = None
    ice_cls = None
    class_id = None
    a_surface_rgb = None
    earth_scalar_interp = str(cfg.earth.get("albedo_map_interp", "nearest"))
    default_surface_rgb = resolve_default_surface_rgb(cfg.earth)
    if earth_class_map is not None:
        cls = earth_class_map.sample_interp(lon, lat, interp=class_interp)
        class_id = cls
        ocean_cls = _class_mask(cls, class_ocean_values, tol=0.5)
        ice_cls = _class_mask(cls, class_ice_values, tol=0.5)
        if class_land_values:
            land_cls = _class_mask(cls, class_land_values, tol=0.5)
        else:
            land_cls = np.isfinite(cls) & ~(ocean_cls | ice_cls)
        a_surface = np.full(lon.shape, float(cfg.earth.get("land_albedo", 0.25)), dtype=np.float64)
        if np.any(ocean_cls):
            a_surface[ocean_cls] = float(cfg.earth.get("ocean_albedo", 0.06))
        if np.any(ice_cls):
            a_surface[ice_cls] = float(cfg.earth.get("ice_albedo", 0.65))
        class_rgb_table = _class_rgb_table(cfg.earth)
        if class_rgb_table:
            a_surface_rgb, known_rgb = _rgb_from_class_ids(
                cls,
                table=class_rgb_table,
                default_rgb=default_surface_rgb,
            )
        else:
            a_surface_rgb = np.repeat(a_surface[:, None], 3, axis=1)
            known_rgb = np.zeros(lon.shape, dtype=bool)
        unknown = ~(ocean_cls | land_cls | ice_cls)
        if np.any(unknown):
            if earth_map is not None:
                amap = earth_map.sample_interp(lon, lat, interp=earth_scalar_interp)
            else:
                amap = toy_land_ocean_albedo(
                    lon,
                    lat,
                    ocean=float(cfg.earth.get("ocean_albedo", 0.06)),
                    land=float(cfg.earth.get("land_albedo", 0.25)),
                )
            a_surface[unknown] = amap[unknown]
        if a_surface_rgb is not None:
            if earth_map is not None:
                amap = earth_map.sample_interp(lon, lat, interp=earth_scalar_interp)
            else:
                amap = toy_land_ocean_albedo(
                    lon,
                    lat,
                    ocean=float(cfg.earth.get("ocean_albedo", 0.06)),
                    land=float(cfg.earth.get("land_albedo", 0.25)),
                )
            unknown_rgb = ~known_rgb
            if np.any(unknown_rgb):
                a_surface_rgb[unknown_rgb, :] = amap[unknown_rgb, None]
    else:
        if earth_map is not None:
            a_surface = earth_map.sample_interp(lon, lat, interp=earth_scalar_interp)
        else:
            a_surface = toy_land_ocean_albedo(
                lon,
                lat,
                ocean=float(cfg.earth.get("ocean_albedo", 0.06)),
                land=float(cfg.earth.get("land_albedo", 0.25)),
            )
        a_surface_rgb = np.repeat(a_surface[:, None], 3, axis=1)

    if earth_ice_fraction_map is not None:
        ice_frac = np.asarray(earth_ice_fraction_map.sample_interp(lon, lat, interp=ice_interp), dtype=np.float64)
        valid_ice_frac = np.isfinite(ice_frac)
        ice_frac = np.where(valid_ice_frac, np.clip(ice_frac, 0.0, 1.0), 0.0)
        if ocean_cls is not None:
            ocean_for_ice = ocean_cls.copy()
        else:
            ocean_for_ice = a_surface <= max(float(cfg.earth.get("ocean_albedo", 0.06)), float(cfg.earth.get("ocean_glint_threshold", 0.12)))
        if bool(cfg.earth.get("ice_fraction_blend", True)):
            w_ice = np.where(ocean_for_ice & valid_ice_frac, ice_frac, 0.0)
            a_surface = (1.0 - w_ice) * a_surface + w_ice * float(cfg.earth.get("ice_albedo", 0.65))
            ice_rgb = np.array(resolve_ice_rgb(cfg.earth), dtype=np.float64)
            a_surface_rgb = (1.0 - w_ice[:, None]) * a_surface_rgb + w_ice[:, None] * ice_rgb[None, :]
        else:
            ice_mask_map = ocean_for_ice & valid_ice_frac & (ice_frac >= float(cfg.earth.get("ice_fraction_threshold", 0.15)))
            if np.any(ice_mask_map):
                a_surface = np.array(a_surface, copy=True)
                a_surface[ice_mask_map] = float(cfg.earth.get("ice_albedo", 0.65))
                a_surface_rgb = np.array(a_surface_rgb, copy=True)
                a_surface_rgb[ice_mask_map, :] = np.array(resolve_ice_rgb(cfg.earth), dtype=np.float64)[None, :]

    if earth_land_ice_mask_map is not None:
        land_ice = np.asarray(earth_land_ice_mask_map.sample_interp(lon, lat, interp=land_ice_interp), dtype=np.float64)
        if earth_land_ice_mask_map_alt is not None:
            land_ice_alt = np.asarray(earth_land_ice_mask_map_alt.sample_interp(lon, lat, interp=land_ice_interp), dtype=np.float64)
            both = np.isfinite(land_ice) & np.isfinite(land_ice_alt)
            land_ice = np.where(both, np.maximum(land_ice, land_ice_alt), np.where(np.isfinite(land_ice), land_ice, land_ice_alt))
        valid_land_ice = np.isfinite(land_ice)
        land_ice = np.where(valid_land_ice, np.clip(land_ice, 0.0, 1.0), 0.0)
        if bool(cfg.earth.get("land_ice_mask_blend", False)):
            # The explicit land-ice mask is authoritative, even if the coarse
            # land-cover class map mislabeled Antarctic pixels as ocean.
            w_land_ice = np.where(valid_land_ice, land_ice, 0.0)
            a_surface = (1.0 - w_land_ice) * a_surface + w_land_ice * float(cfg.earth.get("ice_albedo", 0.65))
            ice_rgb = np.array(resolve_ice_rgb(cfg.earth), dtype=np.float64)
            a_surface_rgb = (1.0 - w_land_ice[:, None]) * a_surface_rgb + w_land_ice[:, None] * ice_rgb[None, :]
        else:
            land_ice_mask = valid_land_ice & (land_ice >= float(cfg.earth.get("land_ice_mask_threshold", 0.5)))
            if np.any(land_ice_mask):
                a_surface = np.array(a_surface, copy=True)
                a_surface[land_ice_mask] = float(cfg.earth.get("ice_albedo", 0.65))
                a_surface_rgb = np.array(a_surface_rgb, copy=True)
                a_surface_rgb[land_ice_mask, :] = np.array(resolve_ice_rgb(cfg.earth), dtype=np.float64)[None, :]
                if class_id is not None:
                    class_id = np.array(class_id, copy=True)
                    class_id[land_ice_mask] = float(cfg.earth.get("land_ice_class_value", 15.0))

    # Seasonal polar ice override.
    if bool(cfg.earth.get("seasonal_ice_enable", True)):
        dt_utc = sp.et2datetime(float(et))
        doy = float(dt_utc.timetuple().tm_yday) + (
            dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0 + dt_utc.microsecond / 3.6e9
        ) / 24.0
        ph_n = 2.0 * np.pi * (doy - float(cfg.earth.get("ice_phase_north_doy", 20.0))) / 365.25
        ph_s = 2.0 * np.pi * (doy - float(cfg.earth.get("ice_phase_south_doy", 200.0))) / 365.25
        lat_n_edge = float(cfg.earth.get("ice_lat_north_base_deg", 72.0)) + float(cfg.earth.get("ice_lat_north_amp_deg", 8.0)) * np.cos(ph_n)
        lat_s_edge = float(cfg.earth.get("ice_lat_south_base_deg", 68.0)) + float(cfg.earth.get("ice_lat_south_amp_deg", 5.0)) * np.cos(ph_s)
        lat_n_edge = float(np.clip(lat_n_edge, 45.0, 89.0))
        lat_s_edge = float(np.clip(lat_s_edge, 45.0, 89.0))
        ice_mask = (lat >= lat_n_edge) | (lat <= -lat_s_edge)
        if ice_cls is not None:
            ice_mask = ice_mask | ice_cls
        a_surface = np.array(a_surface, copy=True)
        a_surface[ice_mask] = float(cfg.earth.get("ice_albedo", 0.65))
        a_surface_rgb = np.array(a_surface_rgb, copy=True)
        a_surface_rgb[ice_mask, :] = np.array(resolve_ice_rgb(cfg.earth), dtype=np.float64)[None, :]

    # Illumination / viewing.
    s = _normalize(sun_pos[None, :] - pv)
    vobs = _normalize(obs_pos[None, :] - pv)
    mu0 = np.clip(np.einsum("ij,ij->i", nv, s), 0.0, 1.0)
    muv = np.clip(np.einsum("ij,ij->i", nv, vobs), 0.0, 1.0)

    # Ocean glint (simple gaussian or Cox-Munk proxy).
    glint_strength = float(cfg.earth.get("ocean_glint_strength", 0.0))
    if glint_strength > 0.0:
        model = str(cfg.earth.get("ocean_glint_model", "simple")).strip().lower()
        ocean_thresh = float(cfg.earth.get("ocean_glint_threshold", 0.12))
        if ocean_cls is not None:
            ocean = ocean_cls & (mu0 > 0.0) & (muv > 0.0)
        else:
            ocean = (a_surface < ocean_thresh) & (mu0 > 0.0) & (muv > 0.0)
        if np.any(ocean):
            if model in ("cox_munk", "coxmunk", "cox-munk"):
                glint = cox_munk_glint_albedo_increment(
                    normals=nv,
                    sun_dir=s,
                    view_dir=vobs,
                    wind_m_s=float(cfg.earth.get("ocean_wind_m_s", 6.0)),
                    refractive_index=float(cfg.earth.get("ocean_refractive_index", 1.334)),
                    strength=glint_strength,
                    max_albedo_increment=float(cfg.earth.get("ocean_glint_max_albedo_increment", 2.0)),
                )
            else:
                sigma = np.deg2rad(max(float(cfg.earth.get("ocean_glint_sigma_deg", 6.0)), 0.1))
                h = _normalize(s + vobs)
                cos_th = np.clip(np.einsum("ij,ij->i", nv, h), -1.0, 1.0)
                theta = np.arccos(cos_th)
                glint = glint_strength * np.exp(-((theta / sigma) ** 2))
            a_surface = a_surface + glint * ocean.astype(float)
            glint_rgb = np.array(
                _parse_rgb_triplet(cfg.earth.get("ocean_glint_rgb", [1.0, 1.0, 1.0]), (1.0, 1.0, 1.0)),
                dtype=np.float64,
            )
            a_surface_rgb = a_surface_rgb + (glint * ocean.astype(float))[:, None] * glint_rgb[None, :]

    # Cloud-over-surface blend.
    cloud_amt = float(np.clip(cfg.earth.get("cloud_amount", 0.0), 0.0, 1.0))
    cloud_alb = float(cfg.earth.get("cloud_albedo", 0.6))
    tau = float(max(cfg.earth.get("cloud_tau", 1.0), 0.0))
    tau_k = float(max(cfg.earth.get("cloud_tau_k", 1.0), 0.0))
    if earth_cloud_fraction_map is not None:
        cloud_field = simple_cloud_fraction_field(lon, lat)
        cloud_field_map = np.asarray(earth_cloud_fraction_map.sample_interp(lon, lat, interp=cloud_interp), dtype=np.float64)
        valid_cloud = np.isfinite(cloud_field_map)
        cloud_field[valid_cloud] = np.clip(cloud_field_map[valid_cloud], 0.0, 1.0)
    else:
        cloud_field = simple_cloud_fraction_field(lon, lat)
    if earth_cloud_tau_map is not None:
        tau_local = np.full(lon.shape, tau, dtype=np.float64)
        tau_local_map = np.asarray(earth_cloud_tau_map.sample_interp(lon, lat, interp=cloud_interp), dtype=np.float64)
        valid_tau = np.isfinite(tau_local_map)
        tau_local[valid_tau] = np.maximum(tau_local_map[valid_tau], 0.0)
    else:
        tau_local = np.full(lon.shape, tau, dtype=np.float64)
    trans = 1.0 - np.exp(-tau_k * tau_local)
    cloud_frac = np.clip(cloud_amt * trans * cloud_field, 0.0, 1.0)
    cloud_rgb = np.array(resolve_cloud_rgb(cfg.earth), dtype=np.float64)
    a_eff = (1.0 - cloud_frac) * a_surface + cloud_frac * cloud_alb
    a_eff = np.clip(a_eff, 0.0, 1.0) * float(cfg.earth.get("albedo", 1.0))
    a_eff_rgb = (1.0 - cloud_frac[:, None]) * a_surface_rgb + cloud_frac[:, None] * cloud_rgb[None, :]
    a_eff_rgb = np.clip(a_eff_rgb, 0.0, 1.0) * float(cfg.earth.get("albedo", 1.0))
    brdf_factor = simple_brdf_factor(
        cfg.earth,
        class_id=class_id,
        ocean_mask=ocean_cls,
        ice_mask=ice_cls,
        mu0=mu0,
        muv=muv,
    )
    a_eff = np.clip(a_eff * brdf_factor, 0.0, 1.0)
    a_eff_rgb = np.clip(a_eff_rgb * brdf_factor[:, None], 0.0, 1.0)
    a_eff_luma = 0.2126 * a_eff_rgb[:, 0] + 0.7152 * a_eff_rgb[:, 1] + 0.0722 * a_eff_rgb[:, 2]
    if earth_class_map is not None and np.any(np.isfinite(a_eff_luma)):
        a_eff = a_eff_luma

    tsi_1au = float(cfg.output.get("tsi_w_m2", 1361.0))
    fsun = tsi_1au * np.array([inv_solar_irradiance_scale(pv[i], sun_pos) for i in range(pv.shape[0])], dtype=np.float64)

    # Lambert Earth radiance proxy toward observer before atmospheric effects.
    rad_surface = (a_eff / np.pi) * fsun * mu0
    rad_surface_rgb = (a_eff_rgb / np.pi) * fsun[:, None] * mu0[:, None]
    if_surface = a_eff * mu0
    if_surface_rgb = a_eff_rgb * mu0[:, None]

    rad_rgb, if_earth_rgb, path_rad_rgb, atm_trans_rgb, atm_view_trans_rgb = _apply_earth_atmosphere(
        earth_cfg=cfg.earth,
        fsun=fsun,
        mu0=mu0,
        muv=muv,
        s=s,
        vobs=vobs,
        rad_surface_rgb=rad_surface_rgb,
        if_surface_rgb=if_surface_rgb,
    )
    rad = 0.2126 * rad_rgb[:, 0] + 0.7152 * rad_rgb[:, 1] + 0.0722 * rad_rgb[:, 2]
    if_earth = 0.2126 * if_earth_rgb[:, 0] + 0.7152 * if_earth_rgb[:, 1] + 0.0722 * if_earth_rgb[:, 2]
    path_rad = 0.2126 * path_rad_rgb[:, 0] + 0.7152 * path_rad_rgb[:, 1] + 0.0722 * path_rad_rgb[:, 2]
    atm_trans = 0.2126 * atm_trans_rgb[:, 0] + 0.7152 * atm_trans_rgb[:, 1] + 0.0722 * atm_trans_rgb[:, 2]

    # Assemble output layers.
    z2 = np.zeros((ny, nx), dtype=np.float64)
    n2 = np.full((ny, nx), np.nan, dtype=np.float64)
    layer_mask = z2.copy()
    layer_mask[mask] = 1.0
    layer_lon = n2.copy(); layer_lon[mask] = lon
    layer_lat = n2.copy(); layer_lat[mask] = lat
    layer_mu0 = z2.copy(); layer_mu0[mask] = mu0
    layer_muv = z2.copy(); layer_muv[mask] = muv
    layer_fsun = z2.copy(); layer_fsun[mask] = fsun
    layer_as = z2.copy(); layer_as[mask] = a_surface
    layer_cf = z2.copy(); layer_cf[mask] = cloud_frac
    layer_ae = z2.copy(); layer_ae[mask] = a_eff
    layer_if = z2.copy(); layer_if[mask] = if_earth
    layer_rad = z2.copy(); layer_rad[mask] = rad
    layer_rad_surf = z2.copy(); layer_rad_surf[mask] = rad_surface
    layer_if_surf = z2.copy(); layer_if_surf[mask] = if_surface
    layer_atm = z2.copy(); layer_atm[mask] = path_rad
    layer_atm_r = z2.copy(); layer_atm_r[mask] = path_rad_rgb[:, 0]
    layer_atm_g = z2.copy(); layer_atm_g[mask] = path_rad_rgb[:, 1]
    layer_atm_b = z2.copy(); layer_atm_b[mask] = path_rad_rgb[:, 2]
    layer_atmt = z2.copy(); layer_atmt[mask] = atm_trans
    layer_atmt_r = z2.copy(); layer_atmt_r[mask] = atm_trans_rgb[:, 0]
    layer_atmt_g = z2.copy(); layer_atmt_g[mask] = atm_trans_rgb[:, 1]
    layer_atmt_b = z2.copy(); layer_atmt_b[mask] = atm_trans_rgb[:, 2]
    layer_as_r = z2.copy(); layer_as_r[mask] = a_surface_rgb[:, 0]
    layer_as_g = z2.copy(); layer_as_g[mask] = a_surface_rgb[:, 1]
    layer_as_b = z2.copy(); layer_as_b[mask] = a_surface_rgb[:, 2]
    layer_ae_r = z2.copy(); layer_ae_r[mask] = a_eff_rgb[:, 0]
    layer_ae_g = z2.copy(); layer_ae_g[mask] = a_eff_rgb[:, 1]
    layer_ae_b = z2.copy(); layer_ae_b[mask] = a_eff_rgb[:, 2]
    layer_if_r = z2.copy(); layer_if_r[mask] = if_earth_rgb[:, 0]
    layer_if_g = z2.copy(); layer_if_g[mask] = if_earth_rgb[:, 1]
    layer_if_b = z2.copy(); layer_if_b[mask] = if_earth_rgb[:, 2]
    layer_rad_r = z2.copy(); layer_rad_r[mask] = rad_rgb[:, 0]
    layer_rad_g = z2.copy(); layer_rad_g[mask] = rad_rgb[:, 1]
    layer_rad_b = z2.copy(); layer_rad_b[mask] = rad_rgb[:, 2]
    # Put physical-units radiance first for convenience in FITS viewers.
    layers = {
        "RAD_EAR": (layer_rad, "W m-2 sr-1"),
        "IF_EARTH": (layer_if, "I/F proxy"),
        "RAD_SURF": (layer_rad_surf, "W m-2 sr-1"),
        "IF_SURF": (layer_if_surf, "I/F proxy"),
        "RAD_ATM": (layer_atm, "W m-2 sr-1"),
        "RAD_R": (layer_rad_r, "W m-2 sr-1"),
        "RAD_G": (layer_rad_g, "W m-2 sr-1"),
        "RAD_B": (layer_rad_b, "W m-2 sr-1"),
        "ATM_R": (layer_atm_r, "W m-2 sr-1"),
        "ATM_G": (layer_atm_g, "W m-2 sr-1"),
        "ATM_B": (layer_atm_b, "W m-2 sr-1"),
        "IF_R": (layer_if_r, "I/F proxy"),
        "IF_G": (layer_if_g, "I/F proxy"),
        "IF_B": (layer_if_b, "I/F proxy"),
        "ATMTOT": (layer_atmt, "transmittance"),
        "ATMT_R": (layer_atmt_r, "transmittance"),
        "ATMT_G": (layer_atmt_g, "transmittance"),
        "ATMT_B": (layer_atmt_b, "transmittance"),
        "A_EFF": (layer_ae, "albedo"),
        "AEFF_R": (layer_ae_r, "albedo"),
        "AEFF_G": (layer_ae_g, "albedo"),
        "AEFF_B": (layer_ae_b, "albedo"),
        "A_SURF": (layer_as, "albedo"),
        "ASRF_R": (layer_as_r, "albedo"),
        "ASRF_G": (layer_as_g, "albedo"),
        "ASRF_B": (layer_as_b, "albedo"),
        "CLOUDF": (layer_cf, "fraction"),
        "MU0": (layer_mu0, "cos(sun)"),
        "MUV": (layer_muv, "cos(view)"),
        "FSUN": (layer_fsun, "W m-2"),
        "ELON": (layer_lon, "deg"),
        "ELAT": (layer_lat, "deg"),
        "MASK": (layer_mask, "1=disk"),
    }
    if class_id is not None:
        layer_cls = n2.copy()
        layer_cls[mask] = class_id
        layers["ECLASS"] = (layer_cls, "class id")

    out = args.out
    if not out:
        stamp = utc.replace(":", "").replace("-", "")
        out = f"OUTPUT/earth_synth_{stamp}.fits"

    hdr = {
        "DATE-OBS": (utc, "UTC"),
        "JD-OBS": (f"{jd:15.7f}", "JD f15.7"),
        "VIEWMODE": (str(args.view), "Viewpoint mode"),
        "NX": (nx, "Output width"),
        "NY": (ny, "Output height"),
        "EARTRAD": (re_km, "Earth radius km"),
        "ALBSCL": (float(cfg.earth.get("albedo", 1.0)), "Earth albedo scale"),
        "TSI1AU": (tsi_1au, "TSI 1 AU"),
        "CLOUDAMT": (cloud_amt, "Cloud amount"),
        "CLOUDTAU": (tau, "Cloud optical thickness"),
        "CLOUDK": (tau_k, "Cloud tau gain"),
        "ECLDMAP": (Path(str(earth_cloud_fraction_map_path)).name if earth_cloud_fraction_map_path else "", "Earth cloud frac map"),
        "ECLDTMAP": (Path(str(earth_cloud_tau_map_path)).name if earth_cloud_tau_map_path else "", "Earth cloud tau map"),
        "EICEMAP": (Path(str(earth_ice_fraction_map_path)).name if earth_ice_fraction_map_path else "", "Earth ice frac map"),
        "ELIMAP": (Path(str(earth_land_ice_mask_path)).name if earth_land_ice_mask_path else "", "Earth land ice map"),
        "GLINTS": (float(cfg.earth.get("ocean_glint_strength", 0.0)), "Glint strength"),
        "GLINTMD": (str(cfg.earth.get("ocean_glint_model", "simple"))[:16], "Glint model"),
        "WINDMS": (float(cfg.earth.get("ocean_wind_m_s", 6.0)), "Wind speed m/s"),
        "SEAICE": (int(bool(cfg.earth.get("seasonal_ice_enable", True))), "Seasonal ice 0/1"),
        "ECLSMAP": (Path(str(earth_class_map_path)).name if earth_class_map_path else "", "Earth class map"),
        "CLRPRES": (str(cfg.earth.get("class_rgb_preset", ""))[:16], "Class RGB preset"),
        "CLRMODE": (earth_color_model(cfg.earth)[:16], "Earth RGB color model"),
    }
    hdr.update(_physical_layer_sum_cards(layers))

    if args.only_layer_index is not None:
        idx = int(args.only_layer_index)
        if idx <= 0 or idx > len(layers):
            raise SystemExit(f"--only-layer-index must be in 1..{len(layers)}")
        name = list(layers.keys())[idx - 1]
        arr, unit = layers[name]
        h = fits.Header()
        for k, (v, c) in hdr.items():
            h[k] = (v, c)
        h["LAYER"] = (name, "Selected layer name")
        h["BUNIT"] = (unit[:68], "Selected layer unit")
        if "W m-2" in str(unit):
            h["SUMPIX"] = (float(np.nansum(np.asarray(arr, dtype=np.float64))), f"Sum of {name}")
        fits.PrimaryHDU(data=np.asarray(arr, dtype=np.float64), header=h).writeto(out, overwrite=True, output_verify="silentfix")
        print(f"Wrote Earth image: {out}  (layer: {name})")
    else:
        write_fits_cube(out_path=out, layers=layers, header_cards=hdr, cube_dtype="float64")
        print(f"Wrote Earth cube: {out}  (layers: {', '.join(layers.keys())})")


if __name__ == "__main__":
    main()
