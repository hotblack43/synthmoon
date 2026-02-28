#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image


def parse_utc(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fmt_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_frame_png(frame: np.ndarray, out_png: Path, vmin: float, vmax: float) -> None:
    x = np.asarray(frame, dtype=np.float64)
    x = np.nan_to_num(x, nan=vmin, posinf=vmax, neginf=vmin)
    if vmax <= vmin:
        y = np.zeros_like(x, dtype=np.uint16)
    else:
        z = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)
        y = np.round(z * 65535.0).astype(np.uint16)
    Image.fromarray(y, mode="I;16").save(out_png)


def run() -> None:
    ap = argparse.ArgumentParser(description="Render hourly synthmoon frames and encode MP4/MOV directly.")
    ap.add_argument("--config", default="scene.toml")
    ap.add_argument("--start-utc", required=True)
    ap.add_argument("--hours", type=int, default=672)
    ap.add_argument("--step-hours", type=int, default=1)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out-mp4", default="OUTPUT/movie_28d_hourly_iftotal.mp4")
    ap.add_argument("--out-mov", default="OUTPUT/movie_28d_hourly_iftotal.mov")
    ap.add_argument("--workdir", default="/tmp/synthmoon_movie_frames")
    ap.add_argument("--tmp-fits", default="/tmp/synthmoon_movie_frame.fits")
    ap.add_argument("--vmin", type=float, default=0.0, help="Display min for video scaling.")
    ap.add_argument("--vmax", type=float, default=None, help="Display max for video scaling. Default: p99.9 of first frame.")
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found in PATH")

    cfg = Path(args.config)
    if not cfg.exists():
        raise SystemExit(f"Missing config: {cfg}")

    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    tmp_fits = Path(args.tmp_fits)
    start = parse_utc(args.start_utc)
    n = int(args.hours)
    step = int(args.step_hours)

    vmax = args.vmax
    for i in range(n):
        t = start + timedelta(hours=i * step)
        utc = fmt_utc(t)
        cmd = [
            sys.executable,
            "-m",
            "synthmoon.run_v0",
            "--config",
            str(cfg),
            "--utc",
            utc,
            "--out",
            str(tmp_fits),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        frame = np.asarray(fits.getdata(tmp_fits))
        if frame.ndim != 2:
            raise RuntimeError(f"Expected 2D frame, got {frame.shape} at frame {i}")

        if vmax is None:
            vals = frame[np.isfinite(frame)]
            vmax = float(np.percentile(vals, 99.9)) if vals.size else 1.0
            if vmax <= args.vmin:
                vmax = args.vmin + 1.0
            print(f"Scaling for video: vmin={args.vmin:.6g}, vmax={vmax:.6g}")

        out_png = workdir / f"frame_{i:04d}.png"
        make_frame_png(frame, out_png, vmin=float(args.vmin), vmax=float(vmax))

        if (i + 1) % 24 == 0 or i == 0 or i == n - 1:
            print(f"[{i+1:4d}/{n}] {utc}")

    out_mp4 = Path(args.out_mp4)
    out_mov = Path(args.out_mov)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    out_mov.parent.mkdir(parents=True, exist_ok=True)

    seq = str(workdir / "frame_%04d.png")
    ff_common = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", seq,
        "-vf", "format=yuv420p",
        "-c:v", "libx264",
        "-crf", str(args.crf),
        "-preset", "slow",
    ]
    subprocess.run(ff_common + [str(out_mp4)], check=True)
    subprocess.run(ff_common + [str(out_mov)], check=True)

    print(f"Wrote MP4: {out_mp4}")
    print(f"Wrote MOV: {out_mov}")
    print(f"Frames kept in: {workdir}")


if __name__ == "__main__":
    run()

