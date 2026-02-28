from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.time import Time
import spiceypy as sp

from .config import load_config
from .spice_tools import (
    load_kernels,
    load_optional_moon_frame_kernels,
    utc_to_et,
    earth_site_state_j2000_earthcenter,
    spacecraft_state_j2000,
    get_sun_earth_moon_states_ssb,
    lunar_north_in_j2000,
    list_loaded_kernels,
    inv_solar_irradiance_scale,
    resolve_moon_frame,
    available_moon_frames,
    sha256_prefix,
)
from .camera import camera_basis_from_boresight_and_up, pixel_rays, moon_roi_bbox
from .intersect import ray_sphere_intersect, ray_sphere_intersect_varradius
from .illumination import (
    lambert_sun_if,
    lambert_sun_if_extended_disk,
    hapke_sun_if,
    hapke_sun_if_extended_disk,
    earthlight_if_tilecached,
    earthlight_if_point_source,
    EarthDiskSampler,
    HapkeParams,
)
from .fits_io import write_fits_cube, scale_to_0_65535_float
from .albedo_maps import EquirectMap
from .moon_dem import LunarDEM


def _short(s: str, n: int = 68) -> str:
    if len(s) <= n:
        return s
    return "…" + s[-(n - 1):]


def _normalize(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def _moon_lonlat_deg(et: float, moon_center_j2000: np.ndarray, pts_j2000: np.ndarray, moon_frame: str = "IAU_MOON") -> tuple[np.ndarray, np.ndarray]:
    """Convert J2000 vectors (Moon-centred) to Moon-fixed lon/lat in degrees."""
    M = sp.pxform("J2000", str(moon_frame), float(et))
    v = pts_j2000 - moon_center_j2000[None, :]
    vf = (M @ v.T).T
    r = np.linalg.norm(vf, axis=1)
    lon = np.rad2deg(np.arctan2(vf[:, 1], vf[:, 0]))
    lat = np.rad2deg(np.arcsin(np.clip(vf[:, 2] / np.maximum(r, 1e-15), -1.0, 1.0)))
    return lon, lat


def _sun_shadow_mask_dem(
    et: float,
    moon_pos_j2000: np.ndarray,
    pts_j2000: np.ndarray,
    normals_j2000: np.ndarray,
    sun_pos_j2000: np.ndarray,
    moon_dem: "LunarDEM",
    moon_frame: str,
    mean_radius_km: float,
    dem_scale: float,
    refine_iter: int = 3,
) -> np.ndarray:
    """Return a boolean mask (len N) for points shadowed from the Sun
    by the sphere+DEM shape model.

    Notes
    -----
    - This implements hard shadows for a point-source Sun.
    - For extended-Sun penumbra, we currently fall back to the mean-sphere model.
    """
    if pts_j2000.size == 0:
        return np.zeros((0,), dtype=bool)

    # Move the ray origin a small distance above the surface to avoid self-intersection.
    eps_km = 1e-3  # 1 m
    origins = pts_j2000 + eps_km * normals_j2000
    dirs = _normalize(sun_pos_j2000 - pts_j2000)

    # Conservative outer envelope: mean radius scaled up to the max DEM radius.
    rmin_km, rmax_km = moon_dem.min_max_radius_km()
    r_env_km = mean_radius_km + max(0.0, (rmax_km - mean_radius_km) * dem_scale)

    hit_env, t_env = ray_sphere_intersect(origins, dirs, moon_pos_j2000, r_env_km)
    if not np.any(hit_env):
        return hit_env

    # Refine against per-ray radius from the DEM at the current hit point.
    hit = hit_env.copy()
    t = t_env.copy()
    for _ in range(int(max(1, refine_iter))):
        idx = np.where(hit)[0]
        if idx.size == 0:
            break
        pts_hit = origins[idx] + t[idx, None] * dirs[idx]
        lon, lat = _moon_lonlat_deg(et, moon_pos_j2000, pts_hit, moon_frame=moon_frame)
        r_dem = moon_dem.radius_km(lon, lat)
        r_km = mean_radius_km + (r_dem - mean_radius_km) * dem_scale
        hit_new, t_new = ray_sphere_intersect_varradius(origins[idx], dirs[idx], moon_pos_j2000, r_km)
        # Update only the rays still hitting.
        hit[idx] = hit_new
        t[idx] = t_new

    return hit


def _sun_shadow_mask_dem_dirs(
    et: float,
    moon_pos_j2000: np.ndarray,
    pts_j2000: np.ndarray,
    normals_j2000: np.ndarray,
    dirs_j2000: np.ndarray,
    moon_dem: "LunarDEM",
    moon_frame: str,
    mean_radius_km: float,
    dem_scale: float,
    refine_iter: int = 3,
) -> np.ndarray:
    """Return shadow mask for arbitrary solar ray directions (one direction per point)."""
    if pts_j2000.size == 0:
        return np.zeros((0,), dtype=bool)

    eps_km = 1e-3
    origins = pts_j2000 + eps_km * normals_j2000
    dirs = _normalize(dirs_j2000)

    _rmin_km, rmax_km = moon_dem.min_max_radius_km()
    r_env_km = mean_radius_km + max(0.0, (rmax_km - mean_radius_km) * dem_scale)

    hit_env, t_env = ray_sphere_intersect(origins, dirs, moon_pos_j2000, r_env_km)
    if not np.any(hit_env):
        return hit_env

    hit = hit_env.copy()
    t = t_env.copy()
    for _ in range(int(max(1, refine_iter))):
        idx = np.where(hit)[0]
        if idx.size == 0:
            break
        pts_hit = origins[idx] + t[idx, None] * dirs[idx]
        lon, lat = _moon_lonlat_deg(et, moon_pos_j2000, pts_hit, moon_frame=moon_frame)
        r_dem = moon_dem.radius_km(lon, lat)
        r_km = mean_radius_km + (r_dem - mean_radius_km) * dem_scale
        hit_new, t_new = ray_sphere_intersect_varradius(origins[idx], dirs[idx], moon_pos_j2000, r_km)
        hit[idx] = hit_new
        t[idx] = t_new

    return hit


def _sun_visibility_fraction_dem_extended(
    *,
    et: float,
    moon_pos_j2000: np.ndarray,
    pts_j2000: np.ndarray,
    normals_j2000: np.ndarray,
    sun_pos_j2000: np.ndarray,
    moon_dem: "LunarDEM",
    moon_frame: str,
    mean_radius_km: float,
    dem_scale: float,
    n_samples: int,
    sun_radius_km: float,
    refine_iter: int = 3,
) -> np.ndarray:
    """Estimate per-point visible fraction of the extended Sun disk using DEM occlusion."""
    n = pts_j2000.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    v = sun_pos_j2000[None, :] - pts_j2000
    d = np.linalg.norm(v, axis=1)
    s_dir = _normalize(v)
    alpha = np.arcsin(np.clip(float(sun_radius_km) / np.maximum(d, 1e-9), 0.0, 1.0))

    # Only points near geometric horizon can have partial visibility.
    radial = _normalize(pts_j2000 - moon_pos_j2000[None, :])
    dotc = np.einsum("ij,ij->i", radial, s_dir)
    sin_alpha = np.sin(alpha)
    full = dotc >= sin_alpha
    none = dotc <= -sin_alpha
    partial = ~(full | none)

    vis = np.zeros(n, dtype=np.float64)
    vis[full] = 1.0
    vis[none] = 0.0
    idx_part = np.where(partial)[0]
    if idx_part.size == 0:
        return vis

    sampler = EarthDiskSampler.create(max(1, int(n_samples)))
    s = sampler.n_samples

    pts_rep = np.repeat(pts_j2000[idx_part], s, axis=0)
    nor_rep = np.repeat(normals_j2000[idx_part], s, axis=0)
    dirs_all = np.zeros((idx_part.size * s, 3), dtype=np.float64)

    for i, k in enumerate(idx_part):
        omega, _w = sampler.directions(s_dir[k], float(alpha[k]))
        dirs_all[i * s : (i + 1) * s] = omega

    blocked = _sun_shadow_mask_dem_dirs(
        et=et,
        moon_pos_j2000=moon_pos_j2000,
        pts_j2000=pts_rep,
        normals_j2000=nor_rep,
        dirs_j2000=dirs_all,
        moon_dem=moon_dem,
        moon_frame=moon_frame,
        mean_radius_km=mean_radius_km,
        dem_scale=dem_scale,
        refine_iter=refine_iter,
    ).reshape(idx_part.size, s)

    vis[idx_part] = 1.0 - blocked.mean(axis=1)
    return np.clip(vis, 0.0, 1.0)


def _derive_variant_path(base_out: str | Path, suffix: str) -> Path:
    p = Path(base_out)
    return p.with_name(f"{p.stem}{suffix}{p.suffix}")


def _run_child_render(
    *,
    config_path: str,
    out_path: Path,
    utc: str | None,
    only_layer_index: int | None,
    legacy_parallel: bool,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "synthmoon.run_v0",
        "--config",
        str(config_path),
        "--out",
        str(out_path),
        "--comparison-child",
    ]
    if utc:
        cmd.extend(["--utc", str(utc)])
    if only_layer_index is not None:
        cmd.extend(["--only-layer-index", str(int(only_layer_index))])
    if legacy_parallel:
        cmd.append("--legacy-parallel")
    subprocess.run(cmd, check=True)


def _read_cube_layer(path: str | Path, layer_name: str) -> np.ndarray:
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data)
        hdr = hdul[0].header
    if data.ndim != 3:
        raise ValueError(f"{path}: expected FITS cube (NAXIS=3), got ndim={data.ndim}")

    nl = int(hdr.get("NLAYERS", data.shape[0]))
    if nl != data.shape[0]:
        nl = data.shape[0]

    for i in range(1, nl + 1):
        if str(hdr.get(f"LAY{i}", "")).strip().upper() == layer_name.upper():
            return np.asarray(data[i - 1], dtype=np.float64)
    raise ValueError(f"{path}: layer {layer_name!r} not found")


def _write_comparison_diff(
    *,
    advanced_path: Path,
    legacy_path: Path,
    diff_path: Path,
    primary_dtype: str,
    pct_floor_if: float = 1e-5,
    pct_clip_percent: float = 1.0,
) -> None:
    adv = _read_cube_layer(advanced_path, "IFTOTAL")
    leg = _read_cube_layer(legacy_path, "IFTOTAL")
    if adv.shape != leg.shape:
        raise ValueError(f"Cannot diff IFTOTAL: shape mismatch {adv.shape} vs {leg.shape}")

    diff = adv - leg
    absdiff = np.abs(diff)
    # Percent difference in floating-point everywhere.
    # Use a signed denominator floor to avoid divide-by-zero while preserving sign.
    eps = max(float(pct_floor_if), 1e-12)
    den = np.where(np.abs(adv) > eps, adv, np.where(adv >= 0.0, eps, -eps))
    pct = 100.0 * (leg - adv) / den
    den_rob = np.maximum(np.maximum(np.abs(adv), np.abs(leg)), eps)
    pct_robust = 100.0 * (leg - adv) / den_rob
    abspct = np.abs(pct)
    abspct_robust = np.abs(pct_robust)
    clipv = max(float(pct_clip_percent), 1e-12)
    pct_clip = np.clip(pct, -clipv, clipv)
    abspct_clip = np.clip(abspct, 0.0, clipv)
    pct_nonneg = np.where(np.isfinite(pct), np.maximum(pct, 0.0), np.nan)
    layers = {
        "ADV_IF": (adv, "I/F advanced"),
        "LEG_IF": (leg, "I/F legacy"),
        "DIFF_IF": (diff, "I/F (adv-legacy)"),
        "ABSDIFF": (absdiff, "I/F abs"),
        "PCTDIFF": (pct, "% (legacy-adv)/adv"),
        "PCTROB": (pct_robust, "% (legacy-adv)/max(|adv|,|legacy|)"),
        "PCTDIFF_CLIP": (pct_clip, f"% clipped to +/-{clipv:g}"),
        "ABSPCT": (abspct, "% abs"),
        "ABSPCTROB": (abspct_robust, "% abs robust"),
        "ABSPCT_CLIP": (abspct_clip, f"% abs clipped to {clipv:g}"),
        "PCTPOS": (pct_nonneg, "% max((legacy-adv)/adv,0)"),
    }
    header_cards = {
        "RUNMODE": ("comparison_diff", "Render mode"),
        "SRCADV": (_short(str(advanced_path)), "Advanced output"),
        "SRCLEG": (_short(str(legacy_path)), "Legacy output"),
        "PCTFLOOR": (float(eps), "Pct floor on |advanced I/F|"),
        "PCTCLIP": (float(clipv), "Pct clip |%| for display"),
    }
    write_fits_cube(
        out_path=diff_path,
        layers=layers,
        header_cards=header_cards,
        cube_dtype=primary_dtype,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="synthmoon renderer (v0.x)")
    ap.add_argument("--config", default="scene.toml", help="Path to scene.toml")
    ap.add_argument("--utc", default=None, help="Override time.utc (ISO-8601, e.g. 2026-02-13T03:12:45Z)")
    ap.add_argument("--out", default=None, help="Override paths.out_fits (output FITS path)")
    ap.add_argument(
        "--only-layer-index",
        type=int,
        default=None,
        help="If set to N>0, write only the Nth layer (1-based) instead of the full cube. (E.g. 5=RADTOT)",
    )
    ap.add_argument("--comparison-child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--legacy-parallel", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    # --- CLI overrides (convenience for sequences / reproducibility) ---
    if args.utc:
        cfg.raw.setdefault("time", {})["utc"] = str(args.utc)
    if args.out:
        cfg.raw.setdefault("paths", {})["out_fits"] = str(args.out)
    if args.only_layer_index is not None:
        cfg.raw.setdefault("output", {})["only_layer_index"] = int(args.only_layer_index)
    if args.legacy_parallel:
        # Legacy comparison mode: point-source Sun + point-source Earth.
        # Keep earthlight enabled, but force true point-Earth approximation.
        cfg.raw.setdefault("sun", {})["extended_disk"] = False
        cfg.raw.setdefault("illumination", {})["include_earthlight"] = True
        cfg.raw.setdefault("earth", {})["point_source"] = True

    comparison = dict(cfg.raw.get("comparison", {}))
    comparison_enabled = bool(comparison.get("enabled", False))
    if comparison_enabled and (not args.comparison_child):
        base_out = Path(cfg.paths.get("out_fits", "OUTPUT/synth_moon_cube.fits"))
        advanced_suffix = str(comparison.get("advanced_suffix", "_advanced"))
        legacy_suffix = str(comparison.get("legacy_suffix", "_legacy_parallel"))
        diff_suffix = str(comparison.get("diff_suffix", "_diff_if"))
        write_diff = bool(comparison.get("write_diff", True))
        pct_floor_if = float(comparison.get("pct_floor_if", 1e-5))
        pct_clip_percent = float(comparison.get("pct_clip_percent", 1.0))
        primary_dtype = str(cfg.output.get("primary_dtype", "float32"))

        out_advanced = _derive_variant_path(base_out, advanced_suffix)
        out_legacy = _derive_variant_path(base_out, legacy_suffix)
        out_diff = _derive_variant_path(base_out, diff_suffix)

        only_layer_index = cfg.output.get("only_layer_index", None)
        try:
            only_layer_index = int(only_layer_index) if only_layer_index is not None else None
        except Exception:
            only_layer_index = None

        print(f"[comparison] advanced -> {out_advanced}")
        _run_child_render(
            config_path=str(args.config),
            out_path=out_advanced,
            utc=args.utc,
            only_layer_index=only_layer_index,
            legacy_parallel=False,
        )
        print(f"[comparison] legacy_parallel -> {out_legacy}")
        _run_child_render(
            config_path=str(args.config),
            out_path=out_legacy,
            utc=args.utc,
            only_layer_index=only_layer_index,
            legacy_parallel=True,
        )

        if write_diff:
            if only_layer_index is not None:
                print("[comparison] Skipping diff: output.only_layer_index is set, full IFTOTAL cube is required.")
            else:
                _write_comparison_diff(
                    advanced_path=out_advanced,
                    legacy_path=out_legacy,
                    diff_path=out_diff,
                    primary_dtype=primary_dtype,
                    pct_floor_if=pct_floor_if,
                    pct_clip_percent=pct_clip_percent,
                )
                print(f"[comparison] diff -> {out_diff}")
        return


    kernels_dir = str(cfg.paths.get("spice_kernels_dir", "KERNELS"))
    mk = cfg.paths.get("meta_kernel", str(Path(kernels_dir) / "generic.tm"))
    load_kernels(mk)

    # Optional (but important) Moon fixed frames (ME/PA) live in extra FK/BPC kernels.
    # We load them if present. If enabled, we can also auto-download the tiny "assoc" FK files.
    auto_fk = bool(cfg.paths.get("auto_download_small_kernels", False) or cfg.moon.get("auto_download_small_fk", False))
    load_optional_moon_frame_kernels(kernels_dir, auto_download_small_fk=auto_fk)

    utc = cfg.utc
    et = utc_to_et(utc)

    moon_frame_req = str(cfg.moon.get("spice_frame", "IAU_MOON"))
    moon_frame = resolve_moon_frame(moon_frame_req)

    # Health check: if you requested a high-accuracy frame, do not silently fall back.
    strict_default = moon_frame_req.upper() not in ("IAU_MOON", "")
    strict = bool(cfg.moon.get("require_spice_frame", strict_default))
    if strict and (moon_frame == "IAU_MOON") and (moon_frame_req.upper() not in ("IAU_MOON", "")):
        avail = available_moon_frames()
        raise RuntimeError(
            "Requested moon.spice_frame=%r but that frame is not available.\n"
            "Loaded Moon frames include: %s\n\n"
            "Fix: ensure you have these kernels and they are loaded (either via generic.tm or auto-load):\n"
            "  KERNELS/pck/moon_pa_de440_200625.bpc\n"
            "  KERNELS/fk/satellites/moon_de440_250416.tf\n"
            "  KERNELS/fk/satellites/moon_assoc_me.tf\n"
            "  KERNELS/fk/satellites/moon_assoc_pa.tf\n"
            "(You already downloaded them once; re-run with paths correct.)\n" % (moon_frame_req, ", ".join(avail))
        )
    if moon_frame != moon_frame_req:
        print(f"NOTE: Requested moon.spice_frame={moon_frame_req!r} -> using {moon_frame!r} (loaded frames)")

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

    north = lunar_north_in_j2000(et, moon_frame=moon_frame)
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
    img_selon_deg = np.full((ny, nx), np.nan, dtype=np.float64)
    img_selat_deg = np.full((ny, nx), np.nan, dtype=np.float64)

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

        # Selenographic lon/lat (deg) for each hit-point in the chosen Moon-fixed frame
        lon, lat = _moon_lonlat_deg(et, moon_pos, pts, moon_frame=moon_frame)

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
                lon, lat = _moon_lonlat_deg(et, moon_pos, pts, moon_frame=moon_frame)

        # Normals + DEM diagnostics
        if moon_dem is not None:
            dem_normal = str(cfg.moon.get("dem_normal", "finite_diff"))
            normals, slope_deg = moon_dem.normals_j2000(et, lon, lat, moon_frame=moon_frame, method=dem_normal)
            elev_m = moon_dem.elevation_m(lon, lat) * dem_scale  # relief-scaled elevation

            img_elev_m[ij_hit[:, 1], ij_hit[:, 0]] = elev_m
            img_slope_deg[ij_hit[:, 1], ij_hit[:, 0]] = slope_deg

            # Store selenographic lon/lat per pixel (degrees); NaN outside the lunar disk
            # lon is in (-180, 180] by construction; users can wrap to 0..360 if desired.
            img_selon_deg[ij_hit[:, 1], ij_hit[:, 0]] = lon
            img_selat_deg[ij_hit[:, 1], ij_hit[:, 0]] = lat
        else:
            normals = pts - moon_pos[None, :]
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)


            # Store selenographic lon/lat per pixel (degrees); NaN outside the lunar disk
            img_selon_deg[ij_hit[:, 1], ij_hit[:, 0]] = lon
            img_selat_deg[ij_hit[:, 1], ij_hit[:, 0]] = lat

        # Moon albedo: constant or map
        A0 = float(cfg.moon.get("albedo", 0.12))
        A_m = np.full(pts.shape[0], A0, dtype=np.float64)
        if moon_map is not None:
            if lon is None or lat is None:
                lon, lat = _moon_lonlat_deg(et, moon_pos, pts, moon_frame=moon_frame)
            interp = str(cfg.moon.get("albedo_map_interp", "nearest"))
            A_map = moon_map.sample_interp(lon, lat, interp=interp)
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
            moon_brdf = str(cfg.moon.get("brdf", "lambert")).strip().lower()
            hapke_cfg = dict(cfg.moon.get("hapke", {}))
            hapke = HapkeParams(
                w=float(hapke_cfg.get("single_scattering_albedo", hapke_cfg.get("w", 0.55))),
                b=float(hapke_cfg.get("phase_b", hapke_cfg.get("b", 0.30))),
                c=float(hapke_cfg.get("phase_c", hapke_cfg.get("c", 0.40))),
                b0=float(hapke_cfg.get("opposition_b0", hapke_cfg.get("b0", 1.00))),
                h=float(hapke_cfg.get("opposition_h", hapke_cfg.get("h", 0.06))),
                theta_deg=float(hapke_cfg.get("roughness_deg", hapke_cfg.get("theta_deg", 20.0))),
            )

            if moon_brdf == "hapke":
                # Keep map/constant albedo as a local multiplicative scale relative to A0.
                alb_scale = A_m / max(A0, 1e-12)
                if use_ext:
                    sun_if = hapke_sun_if_extended_disk(
                        hit_points=pts,
                        normals=normals,
                        moon_center=moon_pos,
                        sun_pos=sun_pos,
                        obs_pos=obs_pos,
                        hapke=hapke,
                        moon_albedo_scale=alb_scale,
                        n_samples=sun_samples,
                        sun_radius_km=sun_radius_km,
                    ).astype(np.float64)
                else:
                    sun_if = hapke_sun_if(
                        hit_points=pts,
                        normals=normals,
                        sun_pos=sun_pos,
                        obs_pos=obs_pos,
                        hapke=hapke,
                        moon_albedo_scale=alb_scale,
                    ).astype(np.float64)
            else:
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

            # Optional terrain shadows:
            # - point-source Sun: hard shadow mask
            # - extended Sun: DEM-based visible-fraction correction (penumbra approximation)
            shadow_sun = str(cfg.shadows.get("sun", cfg.shadows.get("mode", "simple"))).lower()
            if shadow_sun in ("dem", "terrain"):
                if moon_dem is None:
                    print("NOTE: shadows.sun='dem' requested but moon.dem_fits is not set; ignoring.")
                elif not use_ext:
                    mask = _sun_shadow_mask_dem(
                        et=et,
                        moon_pos_j2000=moon_pos,
                        pts_j2000=pts,
                        normals_j2000=normals,
                        sun_pos_j2000=sun_pos,
                        moon_dem=moon_dem,
                        moon_frame=moon_frame,
                        mean_radius_km=Rm,
                        dem_scale=dem_scale,
                        refine_iter=max(1, int(cfg.moon.get("dem_refine_iter", 3))),
                    )
                    sun_if[mask] = 0.0
                else:
                    ext_shadow_samples = int(sun_cfg.get("shadow_disk_samples", max(8, sun_samples // 2)))
                    vis_frac = _sun_visibility_fraction_dem_extended(
                        et=et,
                        moon_pos_j2000=moon_pos,
                        pts_j2000=pts,
                        normals_j2000=normals,
                        sun_pos_j2000=sun_pos,
                        moon_dem=moon_dem,
                        moon_frame=moon_frame,
                        mean_radius_km=Rm,
                        dem_scale=dem_scale,
                        n_samples=ext_shadow_samples,
                        sun_radius_km=sun_radius_km,
                        refine_iter=max(1, int(cfg.moon.get("dem_refine_iter", 3))),
                    )
                    sun_if *= vis_frac

        if bool(cfg.illumination.get("include_earthlight", True)):
            earth_albedo_scale = float(cfg.earth.get("albedo", 1.0))  # acts as scale
            earth_radius = float(cfg.earth.get("radius_km", 6378.1366))
            n_samples = int(cfg.earth.get("earth_disk_samples", 192))
            tile_px = int(cfg.earth.get("earthlight_tile_px", 16))
            earth_point = bool(cfg.earth.get("point_source", False))

            if earth_point:
                earth_if = earthlight_if_point_source(
                    hit_points=pts,
                    normals=normals,
                    moon_center=moon_pos,
                    sun_pos=sun_pos,
                    earth_pos=earth_pos,
                    moon_albedo=A_m,
                    earth_albedo=earth_albedo_scale,
                    earth_radius_km=earth_radius,
                ).astype(np.float64)
            else:
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
                    earth_map_interp=str(cfg.earth.get("albedo_map_interp", "nearest")),
                    ocean_glint_strength=float(cfg.earth.get("ocean_glint_strength", 0.0)),
                    ocean_glint_sigma_deg=float(cfg.earth.get("ocean_glint_sigma_deg", 6.0)),
                    ocean_glint_threshold=float(cfg.earth.get("ocean_glint_threshold", 0.12)),
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

    # Optional kernel manifest for reproducibility
    kernel_hash_mode = str(cfg.output.get("kernel_manifest_hash", "none")).lower()
    kernel_manifest = []
    for k in kernels:
        try:
            p = Path(k)
            nbytes = int(p.stat().st_size) if p.exists() else 0
            h = sha256_prefix(p, n=16) if kernel_hash_mode in ("sha256", "sha") else ""
        except Exception:
            nbytes = 0
            h = ""
        kernel_manifest.append((str(k), nbytes, h))

    mk_hash = ""
    try:
        mk_hash = sha256_prefix(Path(mk), n=16)
    except Exception:
        mk_hash = ""

    tsi_1au = float(cfg.output.get("tsi_w_m2", 1361.0))
    fsun_moon = tsi_1au * float(inv_solar_irradiance_scale(moon_pos, sun_pos))
    rad_fac = fsun_moon / np.pi

    img_rad_total = img_if_total * rad_fac
    img_rad_sun   = img_if_sun * rad_fac
    img_rad_earth = img_if_earth * rad_fac

    run_mode = "legacy_parallel" if args.legacy_parallel else ("advanced" if args.comparison_child else "single")
    earth_point_hdr = int(bool(cfg.earth.get("point_source", False)))
    header_cards = {
        "RUNMODE": (run_mode, "Render mode"),
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
        "EARTHPT": (earth_point_hdr, "Earth point 0/1"),
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
        "MFRAME": (moon_frame, "Moon-fixed frame"),
        "LLCONV": ("arctan2", "Lon=(-180,180], lat=asin(z/r)"),
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
        "SELON":    (img_selon_deg, "deg"),
        "SELAT":    (img_selat_deg, "deg"),
    }

    # Optional: write only a single layer (1-based index) to keep output small for long sequences.
    only_i = cfg.output.get("only_layer_index", None)
    try:
        only_i = int(only_i) if only_i is not None else None
    except Exception:
        only_i = None
    if (only_i is not None) and (only_i > 0):
        keys = list(layers.keys())
        if only_i > len(keys):
            raise ValueError(f"output.only_layer_index={only_i} but only {len(keys)} layers exist: {keys}")
        k = keys[only_i - 1]
        layers = {k: layers[k]}
        header_cards["ONLYLAY"] = (only_i, "Only Nth layer written (1-based)")
        header_cards["ONLYNAME"] = (k, "Name of ONLYLAY")



    write_fits_cube(
        out_path=out_path,
        layers=layers,
        header_cards=header_cards,
        cube_dtype=primary_dtype,
        kernel_manifest=kernel_manifest,
        meta_kernel_path=str(mk),
        meta_kernel_sha256_prefix=mk_hash,
    )

    print(f"Wrote cube: {out_path}  (layers: {', '.join(layers.keys())})")


if __name__ == "__main__":
    main()
