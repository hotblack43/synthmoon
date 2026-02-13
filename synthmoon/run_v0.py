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
from .intersect import ray_sphere_intersect
from .illumination import lambert_sun_if, earthlight_if_tilecached
from .fits_io import write_fits_cube, scale_to_0_65535_float
from .albedo_maps import EquirectMap


def _short(s: str, n: int = 68) -> str:
    if len(s) <= n:
        return s
    return "…" + s[-(n - 1):]


def _moon_lonlat_deg(et: float, moon_center_j2000: np.ndarray, pts_j2000: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert J2000 vectors (Moon-centred) to IAU_MOON lon/lat in degrees."""
    M = sp.pxform("J2000", "IAU_MOON", et)
    v = pts_j2000 - moon_center_j2000[None, :]
    vf = (M @ v.T).T
    r = np.linalg.norm(vf, axis=1)
    lon = np.rad2deg(np.arctan2(vf[:, 1], vf[:, 0]))
    lat = np.rad2deg(np.arcsin(np.clip(vf[:, 2] / np.maximum(r, 1e-15), -1.0, 1.0)))
    return lon, lat


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="synthmoon v0 renderer (Lambert + optional earthlight).")
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

    boresight = moon_pos - obs_pos
    dist_om = float(np.linalg.norm(boresight))
    boresight_u = boresight / dist_om

    north = lunar_north_in_j2000(et)
    R = camera_basis_from_boresight_and_up(boresight_u, north, float(cfg.camera.get("roll_deg", 0.0)))

    nx = int(cfg.camera.get("nx", 512))
    ny = int(cfg.camera.get("ny", 512))
    fov_deg = float(cfg.camera.get("fov_deg", 1.0))

    Rm = float(cfg.moon.get("radius_km", 1737.4))
    moon_ang_radius_deg = np.rad2deg(np.arcsin(np.clip(Rm / dist_om, 0.0, 1.0)))
    bbox = moon_roi_bbox(nx, ny, boresight_u, moon_ang_radius_deg, fov_deg, margin_px=12)

    ij, dirs = pixel_rays(nx, ny, fov_deg, R, bbox)
    origins = np.repeat(obs_pos[None, :], dirs.shape[0], axis=0)

    hit, t = ray_sphere_intersect(origins, dirs, moon_pos, Rm)

    # Component images
    img_if_total = np.zeros((ny, nx), dtype=np.float64)
    img_if_sun   = np.zeros((ny, nx), dtype=np.float64)
    img_if_earth = np.zeros((ny, nx), dtype=np.float64)
    img_alb_moon = np.zeros((ny, nx), dtype=np.float64)

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

    if np.any(hit):
        ij_hit = ij[hit]
        dirs_hit = dirs[hit]
        origins_hit = origins[hit]
        t_hit = t[hit]

        pts = origins_hit + t_hit[:, None] * dirs_hit
        normals = pts - moon_pos[None, :]
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        # Moon albedo: constant * optional map
        A0 = float(cfg.moon.get("albedo", 0.12))
        A_m = np.full(pts.shape[0], A0, dtype=np.float64)
        if moon_map is not None:
            lon, lat = _moon_lonlat_deg(et, moon_pos, pts)
            A_map = moon_map.sample(lon, lat)
            A_scale = float(cfg.moon.get("albedo_map_scale", 1.0))
            A_m = np.clip(A_map * A_scale, 0.0, 1.0)

        img_alb_moon[ij_hit[:, 1], ij_hit[:, 0]] = A_m

        sun_if = np.zeros(pts.shape[0], dtype=np.float64)
        earth_if = np.zeros(pts.shape[0], dtype=np.float64)

        if bool(cfg.illumination.get("include_sun", True)):
            sun_if = lambert_sun_if(pts, normals, sun_pos, A_m).astype(np.float64)

        if bool(cfg.illumination.get("include_earthlight", True)):
            earth_albedo_scale = float(cfg.earth.get("albedo", 1.0))  # acts as scale on map/toy
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
        "ALBMOON0": (float(cfg.moon.get("albedo", 0.12)), "Moon base albedo"),
        "ALBEARTHS": (float(cfg.earth.get("albedo", 1.0)), "Earth albedo scale"),
        "EDSAMP": (int(cfg.earth.get("earth_disk_samples", 192)), "E samp"),
        "TILEPX": (int(cfg.earth.get("earthlight_tile_px", 16)), "Tile px"),
        "INCSUN": (int(bool(cfg.illumination.get("include_sun", True))), "Sun 0/1"),
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
        "ALBMOON":  (img_alb_moon, "unitless (albedo)"),
    }

    write_fits_cube(
        out_path=out_path,
        layers=layers,
        header_cards=header_cards,
        cube_dtype=primary_dtype,
    )

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
