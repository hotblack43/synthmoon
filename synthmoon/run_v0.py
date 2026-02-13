from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from astropy.time import Time
import spiceypy as sp

from .config import load_config
from .spice_tools import (
    load_kernels,
    utc_to_et,
    earth_site_state_j2000_earthcenter,
    spacecraft_state_j2000,
    get_sun_earth_moon_states_ssb,
    lunar_north_in_j2000,
    list_loaded_kernels,
    inv_solar_irradiance_scale,
)
from .camera import camera_basis_from_boresight_and_up, pixel_rays, moon_roi_bbox
from .intersect import ray_sphere_intersect, ray_sphere_intersect_varradius
from .illumination import lambert_sun_if, lambert_sun_if_extended_disk, earthlight_if_tilecached
from .fits_io import write_fits_cube, scale_to_0_65535_float
from .albedo_maps import EquirectMap
from .moon_dem import LunarDEM


def _short(s: str, n: int = 68) -> str:
    if len(s) <= n:
        return s
    return "…" + s[-(n - 1):]


def _moon_lonlat_deg(et: float, moon_center_j2000: np.ndarray, pts_j2000: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert J2000 vectors (Moon-centred) to IAU_MOON lon/lat in degrees."""
    M = sp.pxform("J2000", "IAU_MOON", float(et))
    v = pts_j2000 - moon_center_j2000[None, :]
    vf = (M @ v.T).T
    r = np.linalg.norm(vf, axis=1)
    lon = np.rad2deg(np.arctan2(vf[:, 1], vf[:, 0]))
    lat = np.rad2deg(np.arcsin(np.clip(vf[:, 2] / np.maximum(r, 1e-15), -1.0, 1.0)))
    return lon, lat


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="synthmoon renderer (v0.x)")
    ap.add_argument("--config", default="scene.toml", help="Path to scene.toml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    mk = cfg.paths.get("meta_kernel", str(Path(cfg.paths.get("spice_kernels_dir", "KERNELS")) / "generic.tm"))
    load_kernels(mk)

    utc = cfg.utc
    et = utc_to_et(utc)

    # Body states (SSB origin) in J2000
    states = get_sun_earth_moon_states_ssb(et)
    sun_pos = states["SUN"][:3]
    earth_state_ssb = states["EARTH"]
    earth_pos = earth_state_ssb[:3]
    moon_pos = states["MOON"][:3]

    # Observer state in SAME origin (SSB)
    obs_cfg = cfg.observer
    if cfg.observer_mode == "earth_site":
        obs_state_ec = earth_site_state_j2000_earthcenter(
            et,
            float(obs_cfg["lon_deg"]),
            float(obs_cfg["lat_deg"]),
            float(obs_cfg["height_m"]),
        )
        obs_state = obs_state_ec + earth_state_ssb
    elif cfg.observer_mode == "spacecraft_state":
        obs_state = spacecraft_state_j2000(obs_cfg)
    else:
        raise ValueError(f"Unknown observer.mode: {cfg.observer_mode}")

    obs_pos = obs_state[:3]

    # Camera basis
    boresight = moon_pos - obs_pos
    dist_om = float(np.linalg.norm(boresight))
    boresight_u = boresight / dist_om

    north = lunar_north_in_j2000(et)
    R = camera_basis_from_boresight_and_up(boresight_u, north, float(cfg.camera.get("roll_deg", 0.0)))

    nx = int(cfg.camera.get("nx", 512))
    ny = int(cfg.camera.get("ny", 512))
    fov_deg = float(cfg.camera.get("fov_deg", 1.0))

    # Base Moon radius (km)
    Rm = float(cfg.moon.get("radius_km", 1737.4))

    moon_ang_radius_deg = np.rad2deg(np.arcsin(np.clip(Rm / dist_om, 0.0, 1.0)))
    bbox = moon_roi_bbox(nx, ny, boresight_u, moon_ang_radius_deg, fov_deg, margin_px=12)

    ij, dirs = pixel_rays(nx, ny, fov_deg, R, bbox)
    origins = np.repeat(obs_pos[None, :], dirs.shape[0], axis=0)

    # Initial sphere hit
    hit, t = ray_sphere_intersect(origins, dirs, moon_pos, Rm)

    # Output images
    img_if_total = np.zeros((ny, nx), dtype=np.float64)
    img_if_sun   = np.zeros((ny, nx), dtype=np.float64)
    img_if_earth = np.zeros((ny, nx), dtype=np.float64)
    img_alb_moon = np.zeros((ny, nx), dtype=np.float64)
    img_elev_m   = np.zeros((ny, nx), dtype=np.float64)
    img_slope_deg= np.zeros((ny, nx), dtype=np.float64)

    # Optional albedo maps
    moon_map = None
    moon_map_path = cfg.moon.get("albedo_map_fits", None)
    if moon_map_path:
        moon_lon_mode = str(cfg.moon.get("albedo_map_lon_mode", "0_360"))
        moon_map = EquirectMap.load_fits(moon_map_path, lon_mode=moon_lon_mode, fill=float(cfg.moon.get("albedo_map_fill", 0.12)))

    earth_map = None
    earth_map_path = cfg.earth.get("albedo_map_fits", None)
    if earth_map_path:
        earth_lon_mode = str(cfg.earth.get("albedo_map_lon_mode", "0_360"))
        earth_map = EquirectMap.load_fits(earth_map_path, lon_mode=earth_lon_mode, fill=float(cfg.earth.get("albedo_map_fill", 0.30)))

    # Optional lunar DEM (absolute radius) for orography
    moon_dem = None
    dem_path = cfg.moon.get("dem_fits", None)
    dem_lon_mode = str(cfg.moon.get("dem_lon_mode", "0_360"))
    dem_units = str(cfg.moon.get("dem_units", "m"))
    dem_scale = float(cfg.moon.get("dem_scale", 1.0))  # interpreted as RELIEF scale (1 = physical)
    dem_refine_iter = int(cfg.moon.get("dem_refine_iter", 3))

    if dem_path:
        moon_dem = LunarDEM.load_fits(
            dem_path,
            lon_mode=dem_lon_mode,
            units=dem_units,
            mean_radius_km=Rm,
        )

    if np.any(hit):
        ij_hit = ij[hit]
        dirs_hit = dirs[hit]
        origins_hit = origins[hit]
        t_hit = t[hit]

        pts = origins_hit + t_hit[:, None] * dirs_hit

        lon = None
        lat = None
        if (moon_map is not None) or (moon_dem is not None):
            lon, lat = _moon_lonlat_deg(et, moon_pos, pts)

        # DEM refinement: iterate per-ray radius based on DEM (treat dem_scale as relief scaler)
        if moon_dem is not None and dem_refine_iter > 0:
            for _ in range(dem_refine_iter):
                r_raw = moon_dem.radius_km(lon, lat)  # absolute radius (km)
                r_km = Rm + (r_raw - Rm) * dem_scale  # relief-scaled radius

                hit_ref, t_ref = ray_sphere_intersect_varradius(origins_hit, dirs_hit, moon_pos, r_km)
                if not np.any(hit_ref):
                    break

                # Keep previous t for rays that fail this refinement step
                t_hit = np.where(hit_ref, t_ref, t_hit)
                pts = origins_hit + t_hit[:, None] * dirs_hit
                lon, lat = _moon_lonlat_deg(et, moon_pos, pts)

        # Normals + DEM diagnostics
        if moon_dem is not None:
            normals, slope_deg = moon_dem.normals_j2000(et, lon, lat)
            elev_m = moon_dem.elevation_m(lon, lat) * dem_scale  # relief-scaled elevation

            img_elev_m[ij_hit[:, 1], ij_hit[:, 0]] = elev_m
            img_slope_deg[ij_hit[:, 1], ij_hit[:, 0]] = slope_deg
        else:
            normals = pts - moon_pos[None, :]
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)

        # Moon albedo: constant or map
        A0 = float(cfg.moon.get("albedo", 0.12))
        A_m = np.full(pts.shape[0], A0, dtype=np.float64)
        if moon_map is not None:
            if lon is None or lat is None:
                lon, lat = _moon_lonlat_deg(et, moon_pos, pts)
            A_map = moon_map.sample(lon, lat)
            A_scale = float(cfg.moon.get("albedo_map_scale", 1.0))
            A_m = np.clip(A_map * A_scale, 0.0, 1.0)

        img_alb_moon[ij_hit[:, 1], ij_hit[:, 0]] = A_m

        sun_if = np.zeros(pts.shape[0], dtype=np.float64)
        earth_if = np.zeros(pts.shape[0], dtype=np.float64)

        if bool(cfg.illumination.get("include_sun", True)):
            sun_cfg = dict(cfg.raw.get("sun", {}))
            use_ext = bool(sun_cfg.get("extended_disk", False))
            sun_samples = int(sun_cfg.get("disk_samples", 64))
            sun_radius_km = float(sun_cfg.get("radius_km", 695700.0))

            if use_ext:
                sun_if = lambert_sun_if_extended_disk(
                    hit_points=pts,
                    normals=normals,
                    moon_center=moon_pos,
                    sun_pos=sun_pos,
                    moon_albedo=A_m,
                    n_samples=sun_samples,
                    sun_radius_km=sun_radius_km,
                ).astype(np.float64)
            else:
                sun_if = lambert_sun_if(pts, normals, sun_pos, A_m).astype(np.float64)

        if bool(cfg.illumination.get("include_earthlight", True)):
            earth_albedo_scale = float(cfg.earth.get("albedo", 1.0))  # acts as scale
            earth_radius = float(cfg.earth.get("radius_km", 6378.1366))
            n_samples = int(cfg.earth.get("earth_disk_samples", 192))
            tile_px = int(cfg.earth.get("earthlight_tile_px", 16))

            earth_if = earthlight_if_tilecached(
                hit_points=pts,
                normals=normals,
                moon_center=moon_pos,
                sun_pos=sun_pos,
                earth_pos=earth_pos,
                et=et,
                moon_albedo=A_m,
                earth_albedo=earth_albedo_scale,
                earth_radius_km=earth_radius,
                n_samples=n_samples,
                tile_px=tile_px,
                ij=ij_hit,
                nx=nx,
                ny=ny,
                earth_map=earth_map,
                earth_ocean_albedo=float(cfg.earth.get("ocean_albedo", 0.06)),
                earth_land_albedo=float(cfg.earth.get("land_albedo", 0.25)),
                earth_cloud_amount=float(cfg.earth.get("cloud_amount", 0.0)),
                earth_cloud_albedo=float(cfg.earth.get("cloud_albedo", 0.6)),
            ).astype(np.float64)

        total_if = sun_if + earth_if

        img_if_sun[ij_hit[:, 1], ij_hit[:, 0]] = sun_if
        img_if_earth[ij_hit[:, 1], ij_hit[:, 0]] = earth_if
        img_if_total[ij_hit[:, 1], ij_hit[:, 0]] = total_if
    else:
        print("No Moon intersections found. Check UTC/site/FOV.")

    # Geometry / bookkeeping
    jd = Time(utc, format="isot", scale="utc").jd
    jd_str = f"{jd:15.7f}"

    dist_me = float(np.linalg.norm(earth_pos - moon_pos))
    earth_ang_diam = np.rad2deg(2.0 * np.arcsin(np.clip(float(cfg.earth.get("radius_km", 6378.1366)) / dist_me, 0.0, 1.0)))

    # Sun apparent diameter at Moon centre (deg) and config for extended-disk modelling
    sun_cfg_hdr = dict(cfg.raw.get("sun", {}))
    sun_radius_km_hdr = float(sun_cfg_hdr.get("radius_km", 695700.0))
    dist_sm = float(np.linalg.norm(sun_pos - moon_pos))
    sun_ang_diam = np.rad2deg(2.0 * np.arcsin(np.clip(sun_radius_km_hdr / max(dist_sm, 1e-9), 0.0, 1.0)))
    sun_ext = int(bool(sun_cfg_hdr.get("extended_disk", False)))
    sun_samples_hdr = int(sun_cfg_hdr.get("disk_samples", 64))

    v1 = (sun_pos - moon_pos); v1 /= np.linalg.norm(v1)
    v2 = (obs_pos - moon_pos); v2 /= np.linalg.norm(v2)
    phase = np.rad2deg(np.arccos(np.clip(float(np.dot(v1, v2)), -1.0, 1.0)))

    kernels = list_loaded_kernels()

    tsi_1au = float(cfg.output.get("tsi_w_m2", 1361.0))
    fsun_moon = tsi_1au * float(inv_solar_irradiance_scale(moon_pos, sun_pos))
    rad_fac = fsun_moon / np.pi

    img_rad_total = img_if_total * rad_fac
    img_rad_sun   = img_if_sun * rad_fac
    img_rad_earth = img_if_earth * rad_fac

    header_cards = {
        "DATE-OBS": (utc, "UTC"),
        "JD-OBS": (jd_str, "JD f15.7"),
        "FOV_DEG": (fov_deg, "FOV deg"),
        "OBSMODE": (cfg.observer_mode, "Obs mode"),
        "OBSLON": (float(obs_cfg.get("lon_deg", 0.0)), "Lon deg"),
        "OBSLAT": (float(obs_cfg.get("lat_deg", 0.0)), "Lat deg"),
        "OBSHGT": (float(obs_cfg.get("height_m", 0.0)), "Hgt m"),
        "MOONRAD": (float(cfg.moon.get("radius_km", 1737.4)), "Moon km"),
        "EARTRAD": (float(cfg.earth.get("radius_km", 6378.1366)), "Earth km"),
        "ALBMOON0": (float(cfg.moon.get("albedo", 0.12)), "Moon base alb"),
        "DEMFILE": (Path(dem_path).name if dem_path else "", "Moon DEM"),
        "DEMLON": (dem_lon_mode if dem_path else "", "DEM lon"),
        "DEMUNIT": (dem_units if dem_path else "", "DEM unit"),
        "DEMSCAL": (dem_scale if dem_path else 1.0, "DEM relief scl"),
        "DEMREFI": (dem_refine_iter if dem_path else 0, "DEM refine"),
        "ALBEARTH": (float(cfg.earth.get("albedo", 1.0)), "Earth alb scl"),
        "EDSAMP": (int(cfg.earth.get("earth_disk_samples", 192)), "E samp"),
        "TILEPX": (int(cfg.earth.get("earthlight_tile_px", 16)), "Tile px"),
        "INCSUN": (int(bool(cfg.illumination.get("include_sun", True))), "Sun 0/1"),
        "SUNEXT": (sun_ext, "Sun ext 0/1"),
        "SUNSAMP": (sun_samples_hdr, "Sun disk samp"),
        "SUNRADKM": (sun_radius_km_hdr, "Sun R km"),
        "SUNDIA": (sun_ang_diam, "Sun dia deg"),
        "INCEARTH": (int(bool(cfg.illumination.get("include_earthlight", True))), "Earth 0/1"),
        "DIST_OM": (dist_om, "Obs-Moon"),
        "DIST_ME": (dist_me, "Moon-Earth"),
        "EANGDIA": (earth_ang_diam, "E dia deg"),
        "MPHASE": (phase, "Phase deg"),
        "TSI1AU": (tsi_1au, "TSI 1 AU"),
        "FSUNM": (fsun_moon, "Fsun Moon"),
        "RADFAC": (rad_fac, "Fsun/pi"),
        "MKFILE": (Path(mk).name, "MK"),
        "KCOUNT": (len(kernels), "K n"),
    }

    out_path = cfg.paths.get("out_fits", "OUTPUT/synth_moon_cube.fits")
    primary_dtype = str(cfg.output.get("primary_dtype", "float32"))

    sc = scale_to_0_65535_float(img_if_total)

    layers = {
        "SCALED":   (sc.scaled, "0..65535 (float)"),
        "IFTOTAL":  (img_if_total, "I/F"),
        "IF_SUN":   (img_if_sun, "I/F"),
        "IF_EARTH": (img_if_earth, "I/F"),
        "RADTOT":   (img_rad_total, "W m-2 sr-1"),
        "RAD_SUN":  (img_rad_sun, "W m-2 sr-1"),
        "RAD_EAR":  (img_rad_earth, "W m-2 sr-1"),
        "ALBMOON":  (img_alb_moon, "albedo"),
        "ELEV_M":   (img_elev_m, "elev m (DEM-mean)"),
        "SLOPDEG":  (img_slope_deg, "slope deg"),
    }

    write_fits_cube(
        out_path=out_path,
        layers=layers,
        header_cards=header_cards,
        cube_dtype=primary_dtype,
    )

    # Kernel list in HISTORY (keeps header compact)
    from astropy.io import fits
    with fits.open(out_path, mode="update") as hdul:
        hdr = hdul[0].header
        hdr.add_history("Loaded SPICE kernels:")
        for k in kernels:
            hdr.add_history(_short(str(k)))
        hdul.flush()

    print(f"Wrote cube: {out_path}  (layers: {', '.join(layers.keys())})")


if __name__ == "__main__":
    main()
