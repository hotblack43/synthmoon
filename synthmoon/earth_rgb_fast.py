from __future__ import annotations

from pathlib import Path

import numpy as np
import spiceypy as sp
from astropy.time import Time

from synthmoon.albedo_maps import EquirectMap, simple_cloud_fraction_field, toy_land_ocean_albedo
from synthmoon.config import load_config
from synthmoon.illumination import cox_munk_glint_albedo_increment
from synthmoon.spice_tools import (
    AU_KM,
    earth_site_state_j2000_earthcenter,
    get_sun_earth_moon_states_ssb,
    load_kernels,
    load_optional_moon_frame_kernels,
    utc_to_et,
)
from tools.render_earth_fits import (
    _apply_earth_atmosphere,
    _basis_from_forward_up,
    _class_mask,
    _class_rgb_table,
    _normalize,
    _parse_rgb_triplet,
)


def _vectorized_inv_solar_irradiance_scale(points_km: np.ndarray, sun_pos_km: np.ndarray) -> np.ndarray:
    delta = np.asarray(sun_pos_km, dtype=np.float64)[None, :] - np.asarray(points_km, dtype=np.float64)
    d = np.linalg.norm(delta, axis=1)
    return (AU_KM / np.maximum(d, 1.0e-12)) ** 2


class EarthRGBFastRenderer:
    _loaded_kernel_keys: set[tuple[str, str, bool]] = set()

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.cfg = load_config(self.config_path)
        self._ensure_kernels_loaded()
        self._load_maps()

    def _ensure_kernels_loaded(self) -> None:
        kernels_dir = str(self.cfg.paths.get("spice_kernels_dir", "KERNELS"))
        mk = str(self.cfg.paths.get("meta_kernel", str(Path(kernels_dir) / "generic.tm")))
        auto_fk = bool(self.cfg.paths.get("auto_download_small_kernels", False) or self.cfg.moon.get("auto_download_small_fk", False))
        key = (mk, kernels_dir, auto_fk)
        if key in self._loaded_kernel_keys:
            return
        load_kernels(mk)
        load_optional_moon_frame_kernels(kernels_dir, auto_download_small_fk=auto_fk)
        self._loaded_kernel_keys.add(key)

    def _load_map_if_present(self, path_value: object, *, lon_mode: str, fill: float) -> EquirectMap | None:
        if not path_value:
            return None
        p = Path(str(path_value))
        if not p.exists():
            return None
        return EquirectMap.load_fits(p, lon_mode=lon_mode, fill=fill)

    def _load_maps(self) -> None:
        earth = self.cfg.earth
        albedo_lon_mode = str(earth.get("albedo_map_lon_mode", "0_360"))
        self.earth_map = self._load_map_if_present(
            earth.get("albedo_map_fits", None),
            lon_mode=albedo_lon_mode,
            fill=float(earth.get("albedo_map_fill", 0.30)),
        )

        class_lon_mode = str(earth.get("class_map_lon_mode", albedo_lon_mode))
        self.earth_class_map_path = earth.get("class_map_fits", None)
        self.earth_class_map = self._load_map_if_present(
            self.earth_class_map_path,
            lon_mode=class_lon_mode,
            fill=float(earth.get("class_map_fill", np.nan)),
        )

        cloud_lon_mode = str(earth.get("cloud_map_lon_mode", albedo_lon_mode))
        self.earth_cloud_fraction_map_path = earth.get("cloud_fraction_map_fits", None)
        self.earth_cloud_fraction_map = self._load_map_if_present(
            self.earth_cloud_fraction_map_path,
            lon_mode=cloud_lon_mode,
            fill=float(earth.get("cloud_fraction_map_fill", np.nan)),
        )
        self.earth_cloud_tau_map_path = earth.get("cloud_tau_map_fits", None)
        self.earth_cloud_tau_map = self._load_map_if_present(
            self.earth_cloud_tau_map_path,
            lon_mode=cloud_lon_mode,
            fill=float(earth.get("cloud_tau_map_fill", np.nan)),
        )

        ice_lon_mode = str(earth.get("ice_map_lon_mode", albedo_lon_mode))
        self.earth_ice_fraction_map_path = earth.get("ice_fraction_map_fits", None)
        self.earth_ice_fraction_map = self._load_map_if_present(
            self.earth_ice_fraction_map_path,
            lon_mode=ice_lon_mode,
            fill=float(earth.get("ice_fraction_map_fill", np.nan)),
        )

        land_ice_lon_mode = str(earth.get("land_ice_mask_lon_mode", albedo_lon_mode))
        self.earth_land_ice_mask_path = earth.get("land_ice_mask_fits", None)
        self.earth_land_ice_mask_map = self._load_map_if_present(
            self.earth_land_ice_mask_path,
            lon_mode=land_ice_lon_mode,
            fill=float(earth.get("land_ice_mask_fill", np.nan)),
        )
        self.earth_land_ice_mask_map_alt = None
        if self.earth_land_ice_mask_map is not None:
            alt_lon_mode = "0_360" if land_ice_lon_mode == "-180_180" else "-180_180"
            self.earth_land_ice_mask_map_alt = EquirectMap.load_fits(
                self.earth_land_ice_mask_path,
                lon_mode=alt_lon_mode,
                fill=float(earth.get("land_ice_mask_fill", np.nan)),
            )

        self.class_interp = str(earth.get("class_map_interp", "nearest"))
        self.class_ocean_values = tuple(float(v) for v in earth.get("class_ocean_values", [0]))
        self.class_land_values = tuple(float(v) for v in earth.get("class_land_values", [1]))
        self.class_ice_values = tuple(float(v) for v in earth.get("class_ice_values", [2]))
        self.cloud_interp = str(earth.get("cloud_map_interp", "nearest"))
        self.ice_interp = str(earth.get("ice_map_interp", "nearest"))
        self.land_ice_interp = str(earth.get("land_ice_mask_interp", "nearest"))
        self.scalar_interp = str(earth.get("albedo_map_interp", "nearest"))
        self.default_surface_rgb = _parse_rgb_triplet(
            earth.get("class_rgb_default", [0.20, 0.20, 0.20]),
            (0.20, 0.20, 0.20),
        )
        self.class_rgb_table = _class_rgb_table(earth)

    def render_rgb(
        self,
        *,
        jd: float | None = None,
        utc: str | None = None,
        nx: int = 1024,
        ny: int = 1024,
        view: str = "moon_center",
        lon_deg: float | None = None,
        lat_deg: float | None = None,
        alt_m: float | None = None,
    ) -> dict[str, object]:
        cfg = self.cfg
        if utc is not None and jd is not None:
            raise ValueError("Use either jd or utc, not both.")
        if utc is None:
            utc = str(cfg.utc)
        if jd is not None:
            utc = Time(float(jd), format="jd", scale="utc").isot + "Z"
        jd_val = float(Time(utc, format="isot", scale="utc").jd)
        et = utc_to_et(utc)

        states = get_sun_earth_moon_states_ssb(et)
        sun_pos = states["SUN"][:3]
        earth_state_ssb = states["EARTH"]
        earth_pos = earth_state_ssb[:3]
        moon_pos = states["MOON"][:3]

        if view == "moon_center":
            obs_pos = moon_pos
        else:
            obs_cfg = cfg.observer
            lon = float(lon_deg if lon_deg is not None else obs_cfg.get("lon_deg", 0.0))
            lat = float(lat_deg if lat_deg is not None else obs_cfg.get("lat_deg", 0.0))
            alt = float(alt_m if alt_m is not None else obs_cfg.get("height_m", 0.0))
            obs_state_ec = earth_site_state_j2000_earthcenter(et, lon, lat, alt)
            obs_pos = (obs_state_ec + earth_state_ssb)[:3]

        forward = obs_pos - earth_pos
        r_e2j = np.array(sp.pxform("IAU_EARTH", "J2000", float(et)), dtype=float)
        earth_north_j2k = r_e2j @ np.array([0.0, 0.0, 1.0], dtype=float)
        right, up, fwd = _basis_from_forward_up(forward, earth_north_j2k)

        x = (np.arange(nx, dtype=np.float64) + 0.5 - 0.5 * nx) / (0.5 * nx)
        y = (np.arange(ny, dtype=np.float64) + 0.5 - 0.5 * ny) / (0.5 * ny)
        xx, yy = np.meshgrid(x, y)
        rr2 = xx * xx + yy * yy
        mask = rr2 <= 1.0
        zz = np.sqrt(np.clip(1.0 - rr2, 0.0, 1.0))

        n = np.zeros((ny, nx, 3), dtype=np.float64)
        n[..., :] = (xx[..., None] * right[None, None, :]) + (yy[..., None] * up[None, None, :]) + (zz[..., None] * fwd[None, None, :])
        n = _normalize(n)

        re_km = float(cfg.earth.get("radius_km", 6378.1366))
        p = earth_pos[None, None, :] + re_km * n
        pv = p[mask]
        nv = n[mask]

        mj2e = np.array(sp.pxform("J2000", "IAU_EARTH", float(et)), dtype=float)
        vf = (mj2e @ (pv - earth_pos[None, :]).T).T
        r = np.linalg.norm(vf, axis=1)
        lon = np.rad2deg(np.arctan2(vf[:, 1], vf[:, 0]))
        lat = np.rad2deg(np.arcsin(np.clip(vf[:, 2] / np.maximum(r, 1e-15), -1.0, 1.0)))

        earth = cfg.earth
        class_id = None
        ocean_cls = None
        land_cls = None
        ice_cls = None

        if self.earth_class_map is not None:
            cls = self.earth_class_map.sample_interp(lon, lat, interp=self.class_interp)
            class_id = cls
            ocean_cls = _class_mask(cls, self.class_ocean_values, tol=0.5)
            ice_cls = _class_mask(cls, self.class_ice_values, tol=0.5)
            if self.class_land_values:
                land_cls = _class_mask(cls, self.class_land_values, tol=0.5)
            else:
                land_cls = np.isfinite(cls) & ~(ocean_cls | ice_cls)
            a_surface = np.full(lon.shape, float(earth.get("land_albedo", 0.25)), dtype=np.float64)
            if np.any(ocean_cls):
                a_surface[ocean_cls] = float(earth.get("ocean_albedo", 0.06))
            if np.any(ice_cls):
                a_surface[ice_cls] = float(earth.get("ice_albedo", 0.65))
            if self.class_rgb_table:
                a_surface_rgb = np.empty((cls.shape[0], 3), dtype=np.float64)
                a_surface_rgb[:, 0] = self.default_surface_rgb[0]
                a_surface_rgb[:, 1] = self.default_surface_rgb[1]
                a_surface_rgb[:, 2] = self.default_surface_rgb[2]
                known_rgb = np.zeros(lon.shape, dtype=bool)
                for cls_value, triplet in self.class_rgb_table.items():
                    m = np.abs(cls - float(cls_value)) <= 0.5
                    if not np.any(m):
                        continue
                    a_surface_rgb[m, 0] = triplet[0]
                    a_surface_rgb[m, 1] = triplet[1]
                    a_surface_rgb[m, 2] = triplet[2]
                    known_rgb[m] = True
            else:
                a_surface_rgb = np.repeat(a_surface[:, None], 3, axis=1)
                known_rgb = np.zeros(lon.shape, dtype=bool)
            unknown = ~(ocean_cls | land_cls | ice_cls)
            if np.any(unknown):
                if self.earth_map is not None:
                    amap = self.earth_map.sample_interp(lon, lat, interp=self.scalar_interp)
                else:
                    amap = toy_land_ocean_albedo(
                        lon,
                        lat,
                        ocean=float(earth.get("ocean_albedo", 0.06)),
                        land=float(earth.get("land_albedo", 0.25)),
                    )
                a_surface[unknown] = amap[unknown]
            if self.earth_map is not None:
                amap = self.earth_map.sample_interp(lon, lat, interp=self.scalar_interp)
            else:
                amap = toy_land_ocean_albedo(
                    lon,
                    lat,
                    ocean=float(earth.get("ocean_albedo", 0.06)),
                    land=float(earth.get("land_albedo", 0.25)),
                )
            unknown_rgb = ~known_rgb
            if np.any(unknown_rgb):
                a_surface_rgb[unknown_rgb, :] = amap[unknown_rgb, None]
        else:
            if self.earth_map is not None:
                a_surface = self.earth_map.sample_interp(lon, lat, interp=self.scalar_interp)
            else:
                a_surface = toy_land_ocean_albedo(
                    lon,
                    lat,
                    ocean=float(earth.get("ocean_albedo", 0.06)),
                    land=float(earth.get("land_albedo", 0.25)),
                )
            a_surface_rgb = np.repeat(a_surface[:, None], 3, axis=1)

        if self.earth_ice_fraction_map is not None:
            ice_frac = np.asarray(self.earth_ice_fraction_map.sample_interp(lon, lat, interp=self.ice_interp), dtype=np.float64)
            valid_ice_frac = np.isfinite(ice_frac)
            ice_frac = np.where(valid_ice_frac, np.clip(ice_frac, 0.0, 1.0), 0.0)
            if ocean_cls is not None:
                ocean_for_ice = ocean_cls.copy()
            else:
                ocean_for_ice = a_surface <= max(float(earth.get("ocean_albedo", 0.06)), float(earth.get("ocean_glint_threshold", 0.12)))
            ice_rgb = np.array(_parse_rgb_triplet(earth.get("ice_rgb", [0.78, 0.82, 0.86]), (0.78, 0.82, 0.86)), dtype=np.float64)
            if bool(earth.get("ice_fraction_blend", True)):
                w_ice = np.where(ocean_for_ice & valid_ice_frac, ice_frac, 0.0)
                a_surface = (1.0 - w_ice) * a_surface + w_ice * float(earth.get("ice_albedo", 0.65))
                a_surface_rgb = (1.0 - w_ice[:, None]) * a_surface_rgb + w_ice[:, None] * ice_rgb[None, :]
            else:
                ice_mask_map = ocean_for_ice & valid_ice_frac & (ice_frac >= float(earth.get("ice_fraction_threshold", 0.15)))
                if np.any(ice_mask_map):
                    a_surface = np.array(a_surface, copy=True)
                    a_surface[ice_mask_map] = float(earth.get("ice_albedo", 0.65))
                    a_surface_rgb = np.array(a_surface_rgb, copy=True)
                    a_surface_rgb[ice_mask_map, :] = ice_rgb[None, :]

        if self.earth_land_ice_mask_map is not None:
            land_ice = np.asarray(self.earth_land_ice_mask_map.sample_interp(lon, lat, interp=self.land_ice_interp), dtype=np.float64)
            if self.earth_land_ice_mask_map_alt is not None:
                land_ice_alt = np.asarray(self.earth_land_ice_mask_map_alt.sample_interp(lon, lat, interp=self.land_ice_interp), dtype=np.float64)
                both = np.isfinite(land_ice) & np.isfinite(land_ice_alt)
                land_ice = np.where(both, np.maximum(land_ice, land_ice_alt), np.where(np.isfinite(land_ice), land_ice, land_ice_alt))
            valid_land_ice = np.isfinite(land_ice)
            land_ice = np.where(valid_land_ice, np.clip(land_ice, 0.0, 1.0), 0.0)
            ice_rgb = np.array(_parse_rgb_triplet(earth.get("ice_rgb", [0.78, 0.82, 0.86]), (0.78, 0.82, 0.86)), dtype=np.float64)
            if bool(earth.get("land_ice_mask_blend", False)):
                w_land_ice = np.where(valid_land_ice, land_ice, 0.0)
                a_surface = (1.0 - w_land_ice) * a_surface + w_land_ice * float(earth.get("ice_albedo", 0.65))
                a_surface_rgb = (1.0 - w_land_ice[:, None]) * a_surface_rgb + w_land_ice[:, None] * ice_rgb[None, :]
            else:
                land_ice_mask = valid_land_ice & (land_ice >= float(earth.get("land_ice_mask_threshold", 0.5)))
                if np.any(land_ice_mask):
                    a_surface = np.array(a_surface, copy=True)
                    a_surface[land_ice_mask] = float(earth.get("ice_albedo", 0.65))
                    a_surface_rgb = np.array(a_surface_rgb, copy=True)
                    a_surface_rgb[land_ice_mask, :] = ice_rgb[None, :]

        if bool(earth.get("seasonal_ice_enable", True)):
            dt_utc = sp.et2datetime(float(et))
            doy = float(dt_utc.timetuple().tm_yday) + (
                dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0 + dt_utc.microsecond / 3.6e9
            ) / 24.0
            ph_n = 2.0 * np.pi * (doy - float(earth.get("ice_phase_north_doy", 20.0))) / 365.25
            ph_s = 2.0 * np.pi * (doy - float(earth.get("ice_phase_south_doy", 200.0))) / 365.25
            lat_n_edge = float(earth.get("ice_lat_north_base_deg", 72.0)) + float(earth.get("ice_lat_north_amp_deg", 8.0)) * np.cos(ph_n)
            lat_s_edge = float(earth.get("ice_lat_south_base_deg", 68.0)) + float(earth.get("ice_lat_south_amp_deg", 5.0)) * np.cos(ph_s)
            lat_n_edge = float(np.clip(lat_n_edge, 45.0, 89.0))
            lat_s_edge = float(np.clip(lat_s_edge, 45.0, 89.0))
            ice_mask = (lat >= lat_n_edge) | (lat <= -lat_s_edge)
            if ice_cls is not None:
                ice_mask = ice_mask | ice_cls
            ice_rgb = np.array(_parse_rgb_triplet(earth.get("ice_rgb", [0.78, 0.82, 0.86]), (0.78, 0.82, 0.86)), dtype=np.float64)
            a_surface = np.array(a_surface, copy=True)
            a_surface[ice_mask] = float(earth.get("ice_albedo", 0.65))
            a_surface_rgb = np.array(a_surface_rgb, copy=True)
            a_surface_rgb[ice_mask, :] = ice_rgb[None, :]

        s = _normalize(sun_pos[None, :] - pv)
        vobs = _normalize(obs_pos[None, :] - pv)
        mu0 = np.clip(np.einsum("ij,ij->i", nv, s), 0.0, 1.0)
        muv = np.clip(np.einsum("ij,ij->i", nv, vobs), 0.0, 1.0)

        glint_strength = float(earth.get("ocean_glint_strength", 0.0))
        if glint_strength > 0.0:
            model = str(earth.get("ocean_glint_model", "simple")).strip().lower()
            ocean_thresh = float(earth.get("ocean_glint_threshold", 0.12))
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
                        wind_m_s=float(earth.get("ocean_wind_m_s", 6.0)),
                        refractive_index=float(earth.get("ocean_refractive_index", 1.334)),
                        strength=glint_strength,
                        max_albedo_increment=float(earth.get("ocean_glint_max_albedo_increment", 2.0)),
                    )
                else:
                    sigma = np.deg2rad(max(float(earth.get("ocean_glint_sigma_deg", 6.0)), 0.1))
                    h = _normalize(s + vobs)
                    cos_th = np.clip(np.einsum("ij,ij->i", nv, h), -1.0, 1.0)
                    theta = np.arccos(cos_th)
                    glint = glint_strength * np.exp(-((theta / sigma) ** 2))
                a_surface = a_surface + glint * ocean.astype(float)
                glint_rgb = np.array(_parse_rgb_triplet(earth.get("ocean_glint_rgb", [1.0, 1.0, 1.0]), (1.0, 1.0, 1.0)), dtype=np.float64)
                a_surface_rgb = a_surface_rgb + (glint * ocean.astype(float))[:, None] * glint_rgb[None, :]

        cloud_amt = float(np.clip(earth.get("cloud_amount", 0.0), 0.0, 1.0))
        cloud_alb = float(earth.get("cloud_albedo", 0.6))
        tau = float(max(earth.get("cloud_tau", 1.0), 0.0))
        tau_k = float(max(earth.get("cloud_tau_k", 1.0), 0.0))
        if self.earth_cloud_fraction_map is not None:
            cloud_field = simple_cloud_fraction_field(lon, lat)
            cloud_field_map = np.asarray(self.earth_cloud_fraction_map.sample_interp(lon, lat, interp=self.cloud_interp), dtype=np.float64)
            valid_cloud = np.isfinite(cloud_field_map)
            cloud_field[valid_cloud] = np.clip(cloud_field_map[valid_cloud], 0.0, 1.0)
        else:
            cloud_field = simple_cloud_fraction_field(lon, lat)
        if self.earth_cloud_tau_map is not None:
            tau_local = np.full(lon.shape, tau, dtype=np.float64)
            tau_local_map = np.asarray(self.earth_cloud_tau_map.sample_interp(lon, lat, interp=self.cloud_interp), dtype=np.float64)
            valid_tau = np.isfinite(tau_local_map)
            tau_local[valid_tau] = np.maximum(tau_local_map[valid_tau], 0.0)
        else:
            tau_local = np.full(lon.shape, tau, dtype=np.float64)
        trans = 1.0 - np.exp(-tau_k * tau_local)
        cloud_frac = np.clip(cloud_amt * trans * cloud_field, 0.0, 1.0)
        cloud_rgb = np.array(_parse_rgb_triplet(earth.get("cloud_rgb", [0.78, 0.80, 0.82]), (0.78, 0.80, 0.82)), dtype=np.float64)
        a_eff = (1.0 - cloud_frac) * a_surface + cloud_frac * cloud_alb
        a_eff = np.clip(a_eff, 0.0, 1.0) * float(earth.get("albedo", 1.0))
        a_eff_rgb = (1.0 - cloud_frac[:, None]) * a_surface_rgb + cloud_frac[:, None] * cloud_rgb[None, :]
        a_eff_rgb = np.clip(a_eff_rgb, 0.0, 1.0) * float(earth.get("albedo", 1.0))
        a_eff_luma = 0.2126 * a_eff_rgb[:, 0] + 0.7152 * a_eff_rgb[:, 1] + 0.0722 * a_eff_rgb[:, 2]
        if self.earth_class_map is not None and np.any(np.isfinite(a_eff_luma)):
            a_eff = a_eff_luma

        tsi_1au = float(cfg.output.get("tsi_w_m2", 1361.0))
        fsun = tsi_1au * _vectorized_inv_solar_irradiance_scale(pv, sun_pos)

        rad_surface_rgb = (a_eff_rgb / np.pi) * fsun[:, None] * mu0[:, None]
        if_surface_rgb = a_eff_rgb * mu0[:, None]

        rad_rgb, _, _, _, _ = _apply_earth_atmosphere(
            earth_cfg=earth,
            fsun=fsun,
            mu0=mu0,
            muv=muv,
            s=s,
            vobs=vobs,
            rad_surface_rgb=rad_surface_rgb,
            if_surface_rgb=if_surface_rgb,
        )

        rgb = np.zeros((ny, nx, 3), dtype=np.float64)
        rgb[mask, 0] = rad_rgb[:, 0]
        rgb[mask, 1] = rad_rgb[:, 1]
        rgb[mask, 2] = rad_rgb[:, 2]
        return {
            "utc": utc,
            "jd": jd_val,
            "rgb": rgb,
            "sum_r": float(np.nansum(rad_rgb[:, 0])),
            "sum_g": float(np.nansum(rad_rgb[:, 1])),
            "sum_b": float(np.nansum(rad_rgb[:, 2])),
        }
