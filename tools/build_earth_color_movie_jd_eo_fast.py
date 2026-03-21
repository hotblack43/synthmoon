#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.time import Time
from PIL import Image

try:
    from synthmoon.earth_rgb_fast import EarthRGBFastRenderer
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from synthmoon.earth_rgb_fast import EarthRGBFastRenderer
try:
    from tools.plot_earth_rgb_simple_csv import generate_plots
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.plot_earth_rgb_simple_csv import generate_plots


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


def ensure_eo_for_day(di: DayInfo, fetch_missing: bool) -> tuple[str, str, str, str]:
    modis_cf_glob = f"DATA/MODIS/mod08_d3.a{di.year}{di.doy}.*_cloud_fraction.fits"
    modis_tau_glob = f"DATA/MODIS/mod08_d3.a{di.year}{di.doy}.*_cloud_tau.fits"
    nsidc_ice = f"DATA/NSIDC/ice_fraction_{di.date8}.fits"
    land_ice = f"DATA/MODIS/land_ice_mask_{di.year}.fits"

    modis_cf = first_glob(modis_cf_glob)
    modis_tau = first_glob(modis_tau_glob)
    missing_modis = (modis_cf is None) or (modis_tau is None)
    missing_nsidc = not Path(nsidc_ice).exists()
    missing_land = not Path(land_ice).exists()

    if missing_modis or missing_nsidc or missing_land:
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
            subprocess.run([sys.executable, "tools/download_modis_cloud_granule.py", "--utc", di.utc, "--product", "MOD08_D3", "--out-dir", "DATA/MODIS"], check=True, env=env)
            subprocess.run([sys.executable, "tools/extract_modis_l3_cloud_maps.py", "--in-hdf", f"DATA/MODIS/MOD08_D3.A{di.year}{di.doy}*.hdf", "--out-dir", "DATA/MODIS"], check=True, env=env)
        if missing_nsidc:
            print(f"Fetching NSIDC daily sea ice for {di.utc}")
            subprocess.run([sys.executable, "tools/download_nsidc_g02202_daily.py", "--utc", di.utc, "--hemisphere", "both", "--out-dir", "DATA/NSIDC"], check=True, env=env)
            subprocess.run([sys.executable, "tools/extract_nsidc_g02202_ice_map.py", "--north-nc", f"DATA/NSIDC/sic_psn25_{di.date8}_*.nc", "--south-nc", f"DATA/NSIDC/sic_pss25_{di.date8}_*.nc", "--out-fits", nsidc_ice], check=True, env=env)
        if missing_land:
            print(f"Fetching MODIS static land ice for year {di.year}")
            subprocess.run([sys.executable, "tools/download_modis_landcover_file.py", "--year", di.year, "--product", "MCD12C1", "--out-dir", "DATA/MODIS"], check=True, env=env)
            subprocess.run([sys.executable, "tools/extract_modis_landice_mask.py", "--in-hdf", f"DATA/MODIS/MCD12C1.A{di.year}001*.hdf", "--out-fits", land_ice], check=True, env=env)
        modis_cf = first_glob(modis_cf_glob)
        modis_tau = first_glob(modis_tau_glob)

    if modis_cf is None or modis_tau is None:
        raise RuntimeError(f"Missing MODIS cloud maps after preflight for {di.utc}")
    return modis_cf, modis_tau, nsidc_ice, land_ice


def build_temp_config(src_path: Path, dst_path: Path, *, modis_cf: str, modis_tau: str, nsidc_ice: str, land_ice: str) -> None:
    txt = src_path.read_text(encoding="utf-8")
    lines = txt.splitlines(keepends=True)
    section = ""

    def set_kv(key: str, val: str, line: str) -> str:
        m = re.match(r"^(\s*" + re.escape(key) + r"\s*=\s*)(.*?)(\s*(#.*)?)\n?$", line)
        if not m:
            return line
        newline = "\n" if line.endswith("\n") else ""
        return f"{m.group(1)}{val}{m.group(3)}{newline}"

    out: list[str] = []
    for line in lines:
        sm = re.match(r"^\s*\[([^\]]+)\]\s*$", line.strip())
        if sm:
            section = sm.group(1).strip().lower()
            out.append(line)
            continue
        if section == "earth":
            for key, val in [
                ("cloud_fraction_map_fits", f'"{modis_cf}"'),
                ("cloud_tau_map_fits", f'"{modis_tau}"'),
                ("cloud_map_lon_mode", '"-180_180"'),
                ("ice_fraction_map_fits", f'"{nsidc_ice}"'),
                ("ice_map_lon_mode", '"-180_180"'),
                ("ice_fraction_blend", "true"),
                ("land_ice_mask_fits", f'"{land_ice}"'),
                ("land_ice_mask_lon_mode", '"-180_180"'),
                ("land_ice_mask_blend", "true"),
                ("seasonal_ice_enable", "false"),
            ]:
                line = set_kv(key, val, line)
        out.append(line)
    dst_path.write_text("".join(out), encoding="utf-8")


def auto_scale(rgb: np.ndarray, pct: float) -> float:
    valid = np.isfinite(rgb).all(axis=-1)
    if not np.any(valid):
        return 1.0
    val = float(np.nanpercentile(rgb[valid], pct))
    return max(val, 1.0e-12)


def write_rgb_png(rgb: np.ndarray, out_path: Path, scale: float, pad_frac: float = 0.0) -> None:
    x = np.clip(np.asarray(rgb, dtype=np.float64) / float(scale), 0.0, 1.0)
    x = np.power(x, 1.0 / 2.2)
    x[~np.isfinite(x)] = 0.0
    img8 = np.clip(np.rint(255.0 * x), 0, 255).astype(np.uint8)
    p = max(float(pad_frac), 0.0)
    if p > 0.0:
        ny, nx, nc = img8.shape
        pad_x = int(np.rint(nx * p))
        pad_y = int(np.rint(ny * p))
        canvas = np.zeros((ny + 2 * pad_y, nx + 2 * pad_x, nc), dtype=np.uint8)
        canvas[pad_y:pad_y + ny, pad_x:pad_x + nx, :] = img8
        img8 = canvas
    Image.fromarray(img8, mode="RGB").save(out_path)


def encode_video(frame_glob: str, out_path: Path, fps: int, crf: int) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(int(fps)),
            "-i", frame_glob,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(int(crf)),
            str(out_path),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render an EO-aware Earth-only color movie over a JD range using the in-process fast RGB renderer.")
    ap.add_argument("--config", default="scene.toml")
    ap.add_argument("--start-jd", type=float, required=True)
    ap.add_argument("--end-jd", type=float, required=True)
    ap.add_argument("--step-hours", type=float, required=True)
    ap.add_argument("--nx", type=int, default=1024)
    ap.add_argument("--ny", type=int, default=1024)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--workdir", default="/tmp/synthmoon_earth_movie_eo_fast")
    ap.add_argument("--out-mp4", default="OUTPUT/earth_movie_jd_color_eo_fast.mp4")
    ap.add_argument("--rgb-sums-csv", default=None, help="Optional CSV path for per-frame JD,R,G,B sums. Defaults to <out-mp4 stem>_rgb.csv.")
    ap.add_argument("--plot-out-prefix", default=None, help="Optional prefix for ratio/colour-index plots. Defaults to the MP4 path without suffix.")
    ap.add_argument("--no-rgb-plots", action="store_true", help="Do not generate RGB ratio/colour-index plots after the movie completes.")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--fetch-missing", action="store_true")
    ap.add_argument("--scale-pct", type=float, default=99.7)
    ap.add_argument("--scale-abs", type=float, default=None, help="Absolute radiance scale for RGB channels.")
    ap.add_argument("--pad-frac", type=float, default=0.10, help="Black padding fraction to add on each side of the frame.")
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

    workdir = Path(args.workdir)
    png_dir = workdir / "png"
    cfg_dir = workdir / "cfg"
    if workdir.exists() and not args.keep_frames:
        shutil.rmtree(workdir)
    png_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    src_cfg = Path(args.config)
    scale = float(args.scale_abs) if args.scale_abs is not None else None
    out_mp4 = Path(args.out_mp4)
    if args.rgb_sums_csv:
        rgb_csv_path = Path(args.rgb_sums_csv)
    else:
        rgb_csv_path = out_mp4.with_name(out_mp4.stem + "_rgb.csv")
    if args.plot_out_prefix:
        plot_out_prefix = Path(args.plot_out_prefix)
    else:
        plot_out_prefix = out_mp4.with_suffix("")
    rgb_csv_file = None
    rgb_csv_writer = None
    rgb_csv_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_csv_file = rgb_csv_path.open("w", newline="", encoding="utf-8")
    rgb_csv_writer = csv.writer(rgb_csv_file)
    rgb_csv_writer.writerow(["jd", "sum_r", "sum_g", "sum_b"])

    dayinfos = [utc_to_dayinfo(jd_to_utc(jd)) for jd in jds]
    unique_days: list[DayInfo] = []
    seen = set()
    for di in dayinfos:
        key = (di.year, di.date8, di.doy)
        if key not in seen:
            seen.add(key)
            unique_days.append(di)

    day_cfgs: dict[tuple[str, str, str], Path] = {}
    renderers: dict[Path, EarthRGBFastRenderer] = {}

    print(f"Preflight EO data for {len(unique_days)} UTC day(s)")
    for di in unique_days:
        modis_cf, modis_tau, nsidc_ice, land_ice = ensure_eo_for_day(di, fetch_missing=bool(args.fetch_missing))
        key = (di.year, di.date8, di.doy)
        cfg_path = cfg_dir / f"scene_{di.date8}.toml"
        build_temp_config(src_cfg, cfg_path, modis_cf=modis_cf, modis_tau=modis_tau, nsidc_ice=nsidc_ice, land_ice=land_ice)
        day_cfgs[key] = cfg_path

    try:
        for i, jd in enumerate(jds):
            utc = jd_to_utc(jd)
            di = dayinfos[i]
            cfg_path = day_cfgs[(di.year, di.date8, di.doy)]
            if cfg_path not in renderers:
                renderers[cfg_path] = EarthRGBFastRenderer(cfg_path)
            frame = renderers[cfg_path].render_rgb(jd=jd, nx=int(args.nx), ny=int(args.ny))
            rgb = np.asarray(frame["rgb"], dtype=np.float64)
            if scale is None:
                scale = auto_scale(rgb, args.scale_pct)
                print(f"Earth RGB scaling: scale={scale:.6g} (p{args.scale_pct:g} of first frame)")
            write_rgb_png(rgb, png_dir / f"frame_{i:05d}.png", scale, pad_frac=args.pad_frac)
            rgb_csv_writer.writerow([f"{float(frame['jd']):.10f}", f"{float(frame['sum_r']):.15g}", f"{float(frame['sum_g']):.15g}", f"{float(frame['sum_b']):.15g}"])
            if (i + 1) == 1 or (i + 1) == len(jds) or ((i + 1) % 10 == 0):
                print(f"[{i+1:4d}/{len(jds)}] JD={jd:.7f} UTC={utc} cfg={cfg_path.name}")
    finally:
        if rgb_csv_file is not None:
            rgb_csv_file.close()

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    encode_video(str(png_dir / "frame_%05d.png"), out_mp4, args.fps, args.crf)
    print(f"Wrote Earth MP4: {out_mp4}")
    print(f"Wrote Earth RGB sums CSV: {rgb_csv_path}")

    if not args.no_rgb_plots:
        ratio_path, mag_path = generate_plots(rgb_csv_path, plot_out_prefix)
        print(f"Wrote Earth RGB ratio plot: {ratio_path}")
        print(f"Wrote Earth RGB colour-index plot: {mag_path}")

    if not args.keep_frames:
        shutil.rmtree(workdir)
        print(f"Removed temporary workdir: {workdir}")
    else:
        print(f"Kept PNG frames and cfg files in: {workdir}")


if __name__ == "__main__":
    main()
