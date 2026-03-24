#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from PIL import Image


@dataclass(frozen=True)
class DayInfo:
    utc: str
    year: str
    date8: str
    doy: str


def jd_to_utc(jd: float) -> str:
    return Time(float(jd), format="jd", scale="utc").isot + "Z"


def utc_to_dayinfo(utc: str) -> DayInfo:
    t = datetime.fromisoformat(utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    return DayInfo(
        utc=t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        year=t.strftime("%Y"),
        date8=t.strftime("%Y%m%d"),
        doy=t.strftime("%j"),
    )


def frame_jds(start_jd: float, end_jd: float, step_hours: float) -> list[float]:
    if step_hours <= 0:
        raise ValueError("--step-hours must be > 0")
    step_days = float(step_hours) / 24.0
    vals: list[float] = []
    x = float(start_jd)
    end = float(end_jd)
    eps = step_days * 1.0e-6 + 1.0e-12
    while x <= end + eps:
        vals.append(x)
        x += step_days
    return vals


def first_glob(pattern: str) -> str | None:
    matches = sorted(str(p) for p in Path().glob(pattern))
    return matches[0] if matches else None


def ensure_eo_for_day(di: DayInfo, fetch_missing: bool) -> None:
    modis_cf_glob = f"DATA/MODIS/mod08_d3.a{di.year}{di.doy}.*_cloud_fraction.fits"
    modis_tau_glob = f"DATA/MODIS/mod08_d3.a{di.year}{di.doy}.*_cloud_tau.fits"
    nsidc_ice = f"DATA/NSIDC/ice_fraction_{di.date8}.fits"
    land_ice = f"DATA/MODIS/land_ice_mask_{di.year}.fits"

    modis_cf = first_glob(modis_cf_glob)
    modis_tau = first_glob(modis_tau_glob)
    missing_modis = (modis_cf is None) or (modis_tau is None)
    missing_nsidc = not Path(nsidc_ice).exists()
    missing_land = not Path(land_ice).exists()

    if not (missing_modis or missing_nsidc or missing_land):
        return

    if not fetch_missing:
        lines: list[str] = []
        lines.append(f"Missing EO inputs for UTC day {di.utc}.")
        lines.append("Run these commands:")
        if missing_modis:
            lines.extend([
                "uv run python tools/download_modis_cloud_granule.py \\",
                f"  --utc {di.utc} \\",
                "  --product MOD08_D3 \\",
                "  --out-dir DATA/MODIS",
                "uv run python tools/extract_modis_l3_cloud_maps.py \\",
                f"  --in-hdf 'DATA/MODIS/MOD08_D3.A{di.year}{di.doy}*.hdf' \\",
                "  --out-dir DATA/MODIS",
            ])
        if missing_nsidc:
            lines.extend([
                "uv run python tools/download_nsidc_g02202_daily.py \\",
                f"  --utc {di.utc} \\",
                "  --hemisphere both \\",
                "  --out-dir DATA/NSIDC",
                "uv run python tools/extract_nsidc_g02202_ice_map.py \\",
                f"  --north-nc DATA/NSIDC/sic_psn25_{di.date8}_*.nc \\",
                f"  --south-nc DATA/NSIDC/sic_pss25_{di.date8}_*.nc \\",
                f"  --out-fits {nsidc_ice}",
            ])
        if missing_land:
            lines.extend([
                "uv run python tools/download_modis_landcover_file.py \\",
                f"  --year {di.year} \\",
                "  --product MCD12C1 \\",
                "  --out-dir DATA/MODIS",
                "uv run python tools/extract_modis_landice_mask.py \\",
                f"  --in-hdf 'DATA/MODIS/MCD12C1.A{di.year}001*.hdf' \\",
                f"  --out-fits {land_ice}",
            ])
        raise SystemExit("\n".join(lines))

    env = os.environ.copy()
    env["UV_CACHE_DIR"] = "/tmp/uvcache"
    if missing_modis:
        print(f"Fetching MODIS daily clouds for {di.utc}")
        subprocess.run([
            sys.executable,
            "tools/download_modis_cloud_granule.py",
            "--utc", di.utc,
            "--product", "MOD08_D3",
            "--out-dir", "DATA/MODIS",
        ], check=True, env=env)
        subprocess.run([
            sys.executable,
            "tools/extract_modis_l3_cloud_maps.py",
            "--in-hdf", f"DATA/MODIS/MOD08_D3.A{di.year}{di.doy}*.hdf",
            "--out-dir", "DATA/MODIS",
        ], check=True, env=env)
    if missing_nsidc:
        print(f"Fetching NSIDC daily sea ice for {di.utc}")
        subprocess.run([
            sys.executable,
            "tools/download_nsidc_g02202_daily.py",
            "--utc", di.utc,
            "--hemisphere", "both",
            "--out-dir", "DATA/NSIDC",
        ], check=True, env=env)
        subprocess.run([
            sys.executable,
            "tools/extract_nsidc_g02202_ice_map.py",
            "--north-nc", f"DATA/NSIDC/sic_psn25_{di.date8}_*.nc",
            "--south-nc", f"DATA/NSIDC/sic_pss25_{di.date8}_*.nc",
            "--out-fits", nsidc_ice,
        ], check=True, env=env)
    if missing_land:
        print(f"Fetching MODIS static land ice for year {di.year}")
        subprocess.run([
            sys.executable,
            "tools/download_modis_landcover_file.py",
            "--year", di.year,
            "--product", "MCD12C1",
            "--out-dir", "DATA/MODIS",
        ], check=True, env=env)
        subprocess.run([
            sys.executable,
            "tools/extract_modis_landice_mask.py",
            "--in-hdf", f"DATA/MODIS/MCD12C1.A{di.year}001*.hdf",
            "--out-fits", land_ice,
        ], check=True, env=env)


def make_frame_png(frame: np.ndarray, out_png: Path, vmin: float, vmax: float) -> None:
    x = np.asarray(frame, dtype=np.float64)
    x = np.nan_to_num(x, nan=vmin, posinf=vmax, neginf=vmin)
    if vmax <= vmin:
        y = np.zeros_like(x, dtype=np.uint16)
    else:
        z = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)
        y = np.round(z * 65535.0).astype(np.uint16)
    Image.fromarray(y, mode="I;16").save(out_png)


def auto_vmax(frame: np.ndarray, percentile: float) -> float:
    vals = np.asarray(frame, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    vmax = float(np.percentile(vals, percentile))
    return vmax if vmax > 0.0 else 1.0


def encode_video(frame_glob: str, out_path: Path, fps: int, crf: int) -> None:
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frame_glob,
        "-vf", "format=yuv420p",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "slow",
        str(out_path),
    ], check=True)


def clear_stale_frames(frame_dir: Path) -> None:
    for frame in frame_dir.glob("frame_*.png"):
        frame.unlink()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render JD-range Moon/Earth movie with EO preflight and MP4 encoding")
    ap.add_argument("--config", default="scene.toml")
    ap.add_argument("--start-jd", type=float, required=True)
    ap.add_argument("--end-jd", type=float, required=True)
    ap.add_argument("--step-hours", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--alt-m", type=float, required=True)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--workdir", default="/tmp/synthmoon_movie_jd")
    ap.add_argument("--moon-mp4", default="OUTPUT/moon_movie_jd.mp4")
    ap.add_argument("--earth-mp4", default=None, help="If set, also encode an Earth video")
    ap.add_argument("--fetch-missing", action="store_true")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--moon-vmin", type=float, default=0.0)
    ap.add_argument("--moon-vmax", type=float, default=None)
    ap.add_argument("--moon-vmax-pct", type=float, default=99.9)
    ap.add_argument("--earth-vmin", type=float, default=0.0)
    ap.add_argument("--earth-vmax", type=float, default=None)
    ap.add_argument("--earth-vmax-pct", type=float, default=99.9)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found in PATH")
    if args.end_jd < args.start_jd:
        raise SystemExit("--end-jd must be >= --start-jd")

    jds = frame_jds(args.start_jd, args.end_jd, args.step_hours)
    if not jds:
        raise SystemExit("No frames requested")

    dayinfos = [utc_to_dayinfo(jd_to_utc(jd)) for jd in jds]
    unique_days: list[DayInfo] = []
    seen = set()
    for di in dayinfos:
        key = (di.year, di.date8, di.doy)
        if key not in seen:
            seen.add(key)
            unique_days.append(di)

    print(f"Preflight EO data for {len(unique_days)} UTC day(s)")
    for di in unique_days:
        ensure_eo_for_day(di, fetch_missing=bool(args.fetch_missing))

    workdir = Path(args.workdir)
    moon_dir = workdir / "moon_frames"
    earth_dir = workdir / "earth_frames"
    fits_dir = workdir / "fits"
    if workdir.exists() and not args.keep_frames:
        shutil.rmtree(workdir)
    moon_dir.mkdir(parents=True, exist_ok=True)
    fits_dir.mkdir(parents=True, exist_ok=True)
    clear_stale_frames(moon_dir)
    if args.earth_mp4:
        earth_dir.mkdir(parents=True, exist_ok=True)
        clear_stale_frames(earth_dir)

    moon_vmax = args.moon_vmax
    earth_vmax = args.earth_vmax

    for i, jd in enumerate(jds):
        moon_fits = fits_dir / f"moon_{i:05d}.fits"
        earth_fits = fits_dir / f"earth_{i:05d}.fits"
        cmd = [
            "bash", "tools/go_earth_moon_pair.sh",
            "--config", args.config,
            "--lon", str(args.lon),
            "--lat", str(args.lat),
            "--alt-m", str(args.alt_m),
            "--jd", f"{jd:.10f}",
            "--moon-out", str(moon_fits),
            "--earth-out", str(earth_fits),
        ]
        subprocess.run(cmd, check=True)

        moon_frame = np.asarray(fits.getdata(moon_fits), dtype=np.float64)
        if moon_frame.ndim != 2:
            raise RuntimeError(f"Moon frame {moon_fits} is not 2D: {moon_frame.shape}")
        if moon_vmax is None:
            moon_vmax = auto_vmax(moon_frame, args.moon_vmax_pct)
            print(f"Moon video scaling: vmin={args.moon_vmin:.6g} vmax={moon_vmax:.6g}")
        make_frame_png(moon_frame, moon_dir / f"frame_{i:05d}.png", args.moon_vmin, float(moon_vmax))

        if args.earth_mp4:
            earth_frame = np.asarray(fits.getdata(earth_fits), dtype=np.float64)
            if earth_frame.ndim != 2:
                raise RuntimeError(f"Earth frame {earth_fits} is not 2D: {earth_frame.shape}")
            if earth_vmax is None:
                earth_vmax = auto_vmax(earth_frame, args.earth_vmax_pct)
                print(f"Earth video scaling: vmin={args.earth_vmin:.6g} vmax={earth_vmax:.6g}")
            make_frame_png(earth_frame, earth_dir / f"frame_{i:05d}.png", args.earth_vmin, float(earth_vmax))

        if (i + 1) == 1 or (i + 1) == len(jds) or ((i + 1) % 10 == 0):
            print(f"[{i+1:4d}/{len(jds)}] JD={jd:.7f} UTC={dayinfos[i].utc}")

    moon_mp4 = Path(args.moon_mp4)
    moon_mp4.parent.mkdir(parents=True, exist_ok=True)
    encode_video(str(moon_dir / "frame_%05d.png"), moon_mp4, args.fps, args.crf)
    print(f"Wrote Moon MP4: {moon_mp4}")

    if args.earth_mp4:
        earth_mp4 = Path(args.earth_mp4)
        earth_mp4.parent.mkdir(parents=True, exist_ok=True)
        encode_video(str(earth_dir / "frame_%05d.png"), earth_mp4, args.fps, args.crf)
        print(f"Wrote Earth MP4: {earth_mp4}")

    if not args.keep_frames:
        shutil.rmtree(workdir)
        print(f"Removed temporary workdir: {workdir}")
    else:
        print(f"Kept frames and FITS in: {workdir}")


if __name__ == "__main__":
    main()
