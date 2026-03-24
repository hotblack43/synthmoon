#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from PIL import Image


def jd_to_utc(jd: float) -> str:
    return Time(float(jd), format="jd", scale="utc").isot + "Z"


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render an Earth-only color movie over a JD range.")
    ap.add_argument("--config", default="scene.toml")
    ap.add_argument("--start-jd", type=float, required=True)
    ap.add_argument("--end-jd", type=float, required=True)
    ap.add_argument("--step-hours", type=float, required=True)
    ap.add_argument("--nx", type=int, default=1024)
    ap.add_argument("--ny", type=int, default=1024)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--workdir", default="/tmp/synthmoon_earth_movie")
    ap.add_argument("--out-mp4", default="OUTPUT/earth_movie_jd_color.mp4")
    ap.add_argument("--rgb-sums-csv", default=None, help="Optional CSV path for per-frame JD,R,G,B sums.")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--scale-pct", type=float, default=99.7)
    ap.add_argument("--scale-abs", type=float, default=None, help="Absolute radiance scale for RGB channels.")
    ap.add_argument("--pad-frac", type=float, default=0.10, help="Black padding fraction to add on each side of the frame.")
    return ap.parse_args()


def auto_scale(rgb: np.ndarray, pct: float) -> float:
    valid = np.isfinite(rgb).all(axis=-1)
    if not np.any(valid):
        return 1.0
    val = float(np.nanpercentile(rgb[valid], pct))
    return max(val, 1.0e-12)


def layer_index_from_header(header: fits.Header, layer_name: str) -> int:
    target = str(layer_name).strip().upper()
    n_layers = int(header.get("NLAYERS", 0) or 0)
    for i in range(1, n_layers + 1):
        if str(header.get(f"LAY{i}", "")).strip().upper() == target:
            return i - 1
    raise KeyError(f"Layer {layer_name} not found in FITS header")


def channel_sums(cube: np.ndarray, header: fits.Header) -> tuple[float, float, float]:
    sums: list[float] = []
    for key, layer_name in [("SUMRADR", "RAD_R"), ("SUMRADG", "RAD_G"), ("SUMRADB", "RAD_B")]:
        val = header.get(key)
        if val is None:
            idx = layer_index_from_header(header, layer_name)
            val = float(np.nansum(cube[idx]))
        sums.append(float(val))
    return sums[0], sums[1], sums[2]


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
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(int(fps)),
        "-i", frame_glob,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(int(crf)),
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


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
    fits_dir = workdir / "fits"
    png_dir = workdir / "png"
    if workdir.exists() and not args.keep_frames:
        shutil.rmtree(workdir)
    fits_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    scale = float(args.scale_abs) if args.scale_abs is not None else None
    rgb_csv_path = Path(args.rgb_sums_csv) if args.rgb_sums_csv else None
    rgb_csv_file = None
    rgb_csv_writer = None
    if rgb_csv_path is not None:
        rgb_csv_path.parent.mkdir(parents=True, exist_ok=True)
        rgb_csv_file = rgb_csv_path.open("w", newline="", encoding="utf-8")
        rgb_csv_writer = csv.writer(rgb_csv_file)
        rgb_csv_writer.writerow(["jd", "sum_r", "sum_g", "sum_b"])

    env = dict(os.environ)
    env["UV_CACHE_DIR"] = "/tmp/uvcache"

    try:
        for i, jd in enumerate(jds):
            utc = jd_to_utc(jd)
            fits_path = fits_dir / f"earth_{i:05d}.fits"
            subprocess.run(
                [
                    "uv", "run", "python", "tools/render_earth_fits.py",
                    "--config", args.config,
                    "--jd", f"{jd:.10f}",
                    "--nx", str(int(args.nx)),
                    "--ny", str(int(args.ny)),
                    "--out", str(fits_path),
                ],
                check=True,
                env=env,
            )

            with fits.open(fits_path) as hdul:
                cube = np.asarray(hdul[0].data, dtype=np.float64)
                header = hdul[0].header
            if cube.ndim != 3 or cube.shape[0] < 5:
                raise RuntimeError(f"Unexpected Earth cube shape for {fits_path}: {cube.shape}")
            idx_r = layer_index_from_header(header, "RAD_R")
            idx_g = layer_index_from_header(header, "RAD_G")
            idx_b = layer_index_from_header(header, "RAD_B")
            rgb = np.stack([cube[idx_r], cube[idx_g], cube[idx_b]], axis=-1)
            if scale is None:
                scale = auto_scale(rgb, args.scale_pct)
                print(f"Earth RGB scaling: scale={scale:.6g} (p{args.scale_pct:g} of first frame)")
            write_rgb_png(rgb, png_dir / f"frame_{i:05d}.png", scale, pad_frac=args.pad_frac)
            if rgb_csv_writer is not None:
                sum_r, sum_g, sum_b = channel_sums(cube, header)
                rgb_csv_writer.writerow([f"{jd:.10f}", f"{sum_r:.15g}", f"{sum_g:.15g}", f"{sum_b:.15g}"])
            del cube
            if not args.keep_frames:
                fits_path.unlink(missing_ok=True)

            if (i + 1) == 1 or (i + 1) == len(jds) or ((i + 1) % 10 == 0):
                print(f"[{i+1:4d}/{len(jds)}] JD={jd:.7f} UTC={utc}")
    finally:
        if rgb_csv_file is not None:
            rgb_csv_file.close()

    out_mp4 = Path(args.out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    encode_video(str(png_dir / "frame_%05d.png"), out_mp4, args.fps, args.crf)
    print(f"Wrote Earth MP4: {out_mp4}")

    if not args.keep_frames:
        shutil.rmtree(workdir)
        print(f"Removed temporary workdir: {workdir}")
    else:
        print(f"Kept frames and FITS in: {workdir}")
    if rgb_csv_path is not None:
        print(f"Wrote Earth RGB sums CSV: {rgb_csv_path}")


if __name__ == "__main__":
    main()
