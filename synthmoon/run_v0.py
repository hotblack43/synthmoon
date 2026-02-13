from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from astropy.time import Time

from .config import load_config
from .spice_tools import (
    load_kernels,
    utc_to_et,
    earth_site_state_j2000_earthcenter,
    spacecraft_state_j2000,
    get_sun_earth_moon_states_ssb,
    lunar_north_in_j2000,
    list_loaded_kernels,
)
from .camera import camera_basis_from_boresight_and_up, pixel_rays, moon_roi_bbox
from .intersect import ray_sphere_intersect
from .illumination import lambert_sun_if, earthlight_if_tilecached, _normalize
from .fits_io import write_fits


def _short(s: str, n: int = 68) -> str:
    if len(s) <= n:
        return s
    return "…" + s[-(n - 1):]


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

    # Pointing: moon_center (v0)
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

    # output images
    img_total = np.zeros((ny, nx), dtype=np.float32)
    img_sun = np.zeros((ny, nx), dtype=np.float32)
    img_earth = np.zeros((ny, nx), dtype=np.float32)
    img_mu_earth = np.zeros((ny, nx), dtype=np.float32)
    img_mu_sun = np.zeros((ny, nx), dtype=np.float32)

    if np.any(hit):
        ij_hit = ij[hit]
        dirs_hit = dirs[hit]
        origins_hit = origins[hit]
        t_hit = t[hit]

        pts = origins_hit + t_hit[:, None] * dirs_hit
        normals = pts - moon_pos[None, :]
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        moon_albedo = float(cfg.moon.get("albedo", 0.12))

        # Diagnostics: cosine to Earth center and Sun
        e_dir = _normalize(earth_pos[None, :] - pts)
        s_dir = _normalize(sun_pos[None, :] - pts)
        muE = np.maximum(0.0, np.sum(normals * e_dir, axis=1)).astype(np.float32)
        muS = np.maximum(0.0, np.sum(normals * s_dir, axis=1)).astype(np.float32)
        img_mu_earth[ij_hit[:, 1], ij_hit[:, 0]] = muE
        img_mu_sun[ij_hit[:, 1], ij_hit[:, 0]] = muS

        out_total = np.zeros(pts.shape[0], dtype=np.float32)

        if bool(cfg.illumination.get("include_sun", True)):
            sun_if = lambert_sun_if(pts, normals, sun_pos, moon_albedo).astype(np.float32)
            img_sun[ij_hit[:, 1], ij_hit[:, 0]] = sun_if
            out_total += sun_if

        if bool(cfg.illumination.get("include_earthlight", True)):
            earth_albedo = float(cfg.earth.get("albedo", 0.30))
            earth_radius = float(cfg.earth.get("radius_km", 6378.1366))
            n_samples = int(cfg.earth.get("earth_disk_samples", 192))
            tile_px = int(cfg.earth.get("earthlight_tile_px", 16))

            earth_if = earthlight_if_tilecached(
                hit_points=pts,
                normals=normals,
                moon_center=moon_pos,
                sun_pos=sun_pos,
                earth_pos=earth_pos,
                moon_albedo=moon_albedo,
                earth_albedo=earth_albedo,
                earth_radius_km=earth_radius,
                n_samples=n_samples,
                tile_px=tile_px,
                ij=ij_hit,
                nx=nx,
                ny=ny,
            ).astype(np.float32)

            img_earth[ij_hit[:, 1], ij_hit[:, 0]] = earth_if
            out_total += earth_if

        img_total[ij_hit[:, 1], ij_hit[:, 0]] = out_total
    else:
        print("No Moon intersections found. Check UTC/site/FOV.")

    jd = Time(utc, format="isot", scale="utc").jd
    jd_str = f"{jd:15.7f}"

    dist_me = float(np.linalg.norm(earth_pos - moon_pos))
    earth_ang_diam = np.rad2deg(2.0 * np.arcsin(np.clip(float(cfg.earth.get("radius_km", 6378.1366)) / dist_me, 0.0, 1.0)))

    v1 = (sun_pos - moon_pos)
    v1 /= np.linalg.norm(v1)
    v2 = (obs_pos - moon_pos)
    v2 /= np.linalg.norm(v2)
    phase = np.rad2deg(np.arccos(np.clip(float(np.dot(v1, v2)), -1.0, 1.0)))

    kernels = list_loaded_kernels()

    header_cards = {
        "DATE-OBS": (utc, "UTC"),
        "JD-OBS": (jd_str, "JD f15.7"),
        "BUNIT": (str(cfg.output.get("bunit", "I/F_moon")), "Units"),
        "FOV_DEG": (fov_deg, "FOV deg"),
        "OBSMODE": (cfg.observer_mode, "Obs mode"),
        "OBSLON": (float(obs_cfg.get("lon_deg", 0.0)), "Lon deg"),
        "OBSLAT": (float(obs_cfg.get("lat_deg", 0.0)), "Lat deg"),
        "OBSHGT": (float(obs_cfg.get("height_m", 0.0)), "Hgt m"),
        "MOONRAD": (float(cfg.moon.get("radius_km", 1737.4)), "Moon km"),
        "EARTRAD": (float(cfg.earth.get("radius_km", 6378.1366)), "Earth km"),
        "ALBMOON": (float(cfg.moon.get("albedo", 0.12)), "A moon"),
        "ALBEARTH": (float(cfg.earth.get("albedo", 0.30)), "A earth"),
        "EDSAMP": (int(cfg.earth.get("earth_disk_samples", 192)), "E samp"),
        "TILEPX": (int(cfg.earth.get("earthlight_tile_px", 16)), "Tile px"),
        "INCSUN": (int(bool(cfg.illumination.get("include_sun", True))), "Sun 0/1"),
        "INCEARTH": (int(bool(cfg.illumination.get("include_earthlight", True))), "Earth 0/1"),
        "DIST_OM": (dist_om, "Obs-Moon"),
        "DIST_ME": (dist_me, "Moon-Earth"),
        "EANGDIA": (earth_ang_diam, "E dia deg"),
        "MPHASE": (phase, "Phase deg"),
        "MKFILE": (Path(mk).name, "MK"),
        "KCOUNT": (len(kernels), "K n"),
    }

    out_path = cfg.paths.get("out_fits", "OUTPUT/synth_moon_v0.fits")

    primary_dtype = str(cfg.output.get("primary_dtype", "float32"))
    prim_scaled = bool(cfg.output.get("primary_scaled_0_65535", True))
    store_raw_ext = bool(cfg.output.get("store_raw_extension", True))

    extra = {
        "SUNIF": img_sun,
        "EARTHIF": img_earth,
        "MU_EARTH": img_mu_earth,
        "MU_SUN": img_mu_sun,
    }

    write_fits(
        out_path=out_path,
        img_raw_float=img_total,
        header_cards=header_cards,
        primary_dtype=primary_dtype,
        primary_scaled_0_65535=prim_scaled,
        store_raw_extension=store_raw_ext,
        extra_images=extra,
    )

    from astropy.io import fits
    with fits.open(out_path, mode="update") as hdul:
        hdr = hdul[0].header
        hdr.add_history("Loaded SPICE kernels:")
        for k in kernels:
            hdr.add_history(_short(str(k)))
        hdul.flush()

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
