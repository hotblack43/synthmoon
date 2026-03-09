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
        earth_class_lon_mode = str(cfg.earth.get("class_map_lon_mode", str(cfg.earth.get("albedo_map_lon_mode", "0_360"))))
        earth_class_map = EquirectMap.load_fits(
            earth_class_map_path,
            lon_mode=earth_class_lon_mode,
            fill=float(cfg.earth.get("class_map_fill", -999.0)),
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
            fill=float(cfg.earth.get("cloud_fraction_map_fill", 0.0)),
        )

    earth_cloud_tau_map = None
    earth_cloud_tau_map_path = cfg.earth.get("cloud_tau_map_fits", None)
    if earth_cloud_tau_map_path:
        earth_cloud_lon_mode = str(cfg.earth.get("cloud_map_lon_mode", str(cfg.earth.get("albedo_map_lon_mode", "0_360"))))
        earth_cloud_tau_map = EquirectMap.load_fits(
            earth_cloud_tau_map_path,
            lon_mode=earth_cloud_lon_mode,
            fill=float(cfg.earth.get("cloud_tau_map_fill", 0.0)),
        )
    cloud_interp = str(cfg.earth.get("cloud_map_interp", "nearest"))

    earth_ice_fraction_map = None
    earth_ice_fraction_map_path = cfg.earth.get("ice_fraction_map_fits", None)
    if earth_ice_fraction_map_path:
        earth_ice_lon_mode = str(cfg.earth.get("ice_map_lon_mode", str(cfg.earth.get("albedo_map_lon_mode", "0_360"))))
        earth_ice_fraction_map = EquirectMap.load_fits(
            earth_ice_fraction_map_path,
            lon_mode=earth_ice_lon_mode,
            fill=float(cfg.earth.get("ice_fraction_map_fill", 0.0)),
        )
    ice_interp = str(cfg.earth.get("ice_map_interp", "nearest"))

    earth_land_ice_mask_map = None
    earth_land_ice_mask_path = cfg.earth.get("land_ice_mask_fits", None)
    if earth_land_ice_mask_path:
        earth_land_ice_lon_mode = str(cfg.earth.get("land_ice_mask_lon_mode", str(cfg.earth.get("albedo_map_lon_mode", "0_360"))))
        earth_land_ice_mask_map = EquirectMap.load_fits(
            earth_land_ice_mask_path,
            lon_mode=earth_land_ice_lon_mode,
            fill=float(cfg.earth.get("land_ice_mask_fill", 0.0)),
        )
    land_ice_interp = str(cfg.earth.get("land_ice_mask_interp", "nearest"))

    ocean_cls = None
    land_cls = None
    ice_cls = None
    class_id = None
    if earth_class_map is not None:
        cls = earth_class_map.sample_interp(lon, lat, interp=class_interp)
        class_id = cls
        ocean_cls = _class_mask(cls, class_ocean_values, tol=0.5)
        land_cls = _class_mask(cls, class_land_values, tol=0.5)
        ice_cls = _class_mask(cls, class_ice_values, tol=0.5)
        a_surface = np.full(lon.shape, float(cfg.earth.get("land_albedo", 0.25)), dtype=np.float64)
        if np.any(ocean_cls):
            a_surface[ocean_cls] = float(cfg.earth.get("ocean_albedo", 0.06))
        if np.any(ice_cls):
            a_surface[ice_cls] = float(cfg.earth.get("ice_albedo", 0.65))
        unknown = ~(ocean_cls | land_cls | ice_cls)
        if np.any(unknown):
            if earth_map is not None:
                amap = earth_map.sample_interp(lon, lat, interp=str(cfg.earth.get("albedo_map_interp", "nearest")))
            else:
                amap = toy_land_ocean_albedo(
                    lon,
                    lat,
                    ocean=float(cfg.earth.get("ocean_albedo", 0.06)),
                    land=float(cfg.earth.get("land_albedo", 0.25)),
                )
            a_surface[unknown] = amap[unknown]
    else:
        if earth_map is not None:
            a_surface = earth_map.sample_interp(lon, lat, interp=str(cfg.earth.get("albedo_map_interp", "nearest")))
        else:
            a_surface = toy_land_ocean_albedo(
                lon,
                lat,
                ocean=float(cfg.earth.get("ocean_albedo", 0.06)),
                land=float(cfg.earth.get("land_albedo", 0.25)),
            )

    if earth_ice_fraction_map is not None:
        ice_frac = np.asarray(earth_ice_fraction_map.sample_interp(lon, lat, interp=ice_interp), dtype=np.float64)
        ice_frac = np.clip(ice_frac, 0.0, 1.0)
        if ocean_cls is not None:
            ocean_for_ice = ocean_cls.copy()
        else:
            ocean_for_ice = a_surface <= max(float(cfg.earth.get("ocean_albedo", 0.06)), float(cfg.earth.get("ocean_glint_threshold", 0.12)))
        if bool(cfg.earth.get("ice_fraction_blend", True)):
            w_ice = np.where(ocean_for_ice, ice_frac, 0.0)
            a_surface = (1.0 - w_ice) * a_surface + w_ice * float(cfg.earth.get("ice_albedo", 0.65))
        else:
            ice_mask_map = ocean_for_ice & (ice_frac >= float(cfg.earth.get("ice_fraction_threshold", 0.15)))
            if np.any(ice_mask_map):
                a_surface = np.array(a_surface, copy=True)
                a_surface[ice_mask_map] = float(cfg.earth.get("ice_albedo", 0.65))

    if earth_land_ice_mask_map is not None:
        land_ice = np.asarray(earth_land_ice_mask_map.sample_interp(lon, lat, interp=land_ice_interp), dtype=np.float64)
        land_ice = np.clip(land_ice, 0.0, 1.0)
        if earth_class_map is not None:
            land_for_ice = land_cls | ice_cls
        elif ocean_cls is not None:
            land_for_ice = ~ocean_cls
        else:
            land_for_ice = a_surface > max(float(cfg.earth.get("ocean_albedo", 0.06)), float(cfg.earth.get("ocean_glint_threshold", 0.12)))
        if bool(cfg.earth.get("land_ice_mask_blend", False)):
            w_land_ice = np.where(land_for_ice, land_ice, 0.0)
            a_surface = (1.0 - w_land_ice) * a_surface + w_land_ice * float(cfg.earth.get("ice_albedo", 0.65))
        else:
            land_ice_mask = land_for_ice & (land_ice >= float(cfg.earth.get("land_ice_mask_threshold", 0.5)))
            if np.any(land_ice_mask):
                a_surface = np.array(a_surface, copy=True)
                a_surface[land_ice_mask] = float(cfg.earth.get("ice_albedo", 0.65))
                if class_id is not None:
                    class_id = np.array(class_id, copy=True)
                    class_id[land_ice_mask] = float(cfg.earth.get("land_ice_class_value", 2.0))

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

    # Cloud-over-surface blend.
    cloud_amt = float(np.clip(cfg.earth.get("cloud_amount", 0.0), 0.0, 1.0))
    cloud_alb = float(cfg.earth.get("cloud_albedo", 0.6))
    tau = float(max(cfg.earth.get("cloud_tau", 1.0), 0.0))
    tau_k = float(max(cfg.earth.get("cloud_tau_k", 1.0), 0.0))
    if earth_cloud_fraction_map is not None:
        cloud_field = np.asarray(earth_cloud_fraction_map.sample_interp(lon, lat, interp=cloud_interp), dtype=np.float64)
        cloud_field = np.clip(cloud_field, 0.0, 1.0)
    else:
        cloud_field = simple_cloud_fraction_field(lon, lat)
    if earth_cloud_tau_map is not None:
        tau_local = np.asarray(earth_cloud_tau_map.sample_interp(lon, lat, interp=cloud_interp), dtype=np.float64)
        tau_local = np.maximum(tau_local, 0.0)
    else:
        tau_local = np.full(lon.shape, tau, dtype=np.float64)
    trans = 1.0 - np.exp(-tau_k * tau_local)
    cloud_frac = np.clip(cloud_amt * trans * cloud_field, 0.0, 1.0)
    a_eff = (1.0 - cloud_frac) * a_surface + cloud_frac * cloud_alb
    a_eff = np.clip(a_eff, 0.0, 1.0) * float(cfg.earth.get("albedo", 1.0))

    fsun = np.array([inv_solar_irradiance_scale(pv[i], sun_pos) for i in range(pv.shape[0])], dtype=np.float64)

    # Lambert Earth radiance proxy toward observer:
    # L = (A_eff/pi) * F_sun * mu0
    rad = (a_eff / np.pi) * fsun * mu0
    if_earth = a_eff * mu0

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
    layer_cls = n2.copy()
    if class_id is not None:
        layer_cls[mask] = class_id

    # Put physical-units radiance first for convenience in FITS viewers.
    layers = {
        "RAD_EAR": (layer_rad, "W m-2 sr-1 (scaled)"),
        "IF_EARTH": (layer_if, "I/F proxy"),
        "ECLASS": (layer_cls, "class id"),
        "A_EFF": (layer_ae, "albedo"),
        "A_SURF": (layer_as, "albedo"),
        "CLOUDF": (layer_cf, "fraction"),
        "MU0": (layer_mu0, "cos(sun)"),
        "MUV": (layer_muv, "cos(view)"),
        "FSUN": (layer_fsun, "F/F_1AU"),
        "ELON": (layer_lon, "deg"),
        "ELAT": (layer_lat, "deg"),
        "MASK": (layer_mask, "1=disk"),
    }

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
    }

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
        fits.PrimaryHDU(data=np.asarray(arr, dtype=np.float64), header=h).writeto(out, overwrite=True, output_verify="silentfix")
        print(f"Wrote Earth image: {out}  (layer: {name})")
    else:
        write_fits_cube(out_path=out, layers=layers, header_cards=hdr, cube_dtype="float64")
        print(f"Wrote Earth cube: {out}  (layers: {', '.join(layers.keys())})")


if __name__ == "__main__":
    main()
