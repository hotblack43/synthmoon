#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Build hourly movie FITS cube from synthmoon single-image outputs.")
    ap.add_argument("--config", default="/tmp/scene_movie.toml", help="Config for advanced single-image render.")
    ap.add_argument("--start-utc", required=True, help="Start UTC, e.g. 2006-02-13T06:18:45Z")
    ap.add_argument("--hours", type=int, default=672, help="Frame count (default 672 = 28d hourly).")
    ap.add_argument("--step-hours", type=int, default=1, help="Step in hours.")
    ap.add_argument("--out", default="OUTPUT/movie_28d_hourly_iftotal.fits", help="Output movie cube FITS.")
    ap.add_argument("--tmp-frame", default="/tmp/synthmoon_movie_frame.fits", help="Temporary frame path.")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="Movie cube dtype.")
    args = ap.parse_args()

    cfg = Path(args.config)
    if not cfg.exists():
        raise SystemExit(f"Missing config: {cfg}")

    start = parse_utc(args.start_utc)
    n = int(args.hours)
    step = int(args.step_hours)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_frame = Path(args.tmp_frame)

    cube = None
    utc_list: list[str] = []
    jd_list: list[float] = []

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
            str(tmp_frame),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with fits.open(tmp_frame) as h:
            frame = np.asarray(h[0].data)
            hdr = h[0].header

        if frame.ndim != 2:
            raise RuntimeError(f"Expected 2D frame, got shape={frame.shape} at i={i}")

        if cube is None:
            ny, nx = frame.shape
            dtype = np.float32 if args.dtype == "float32" else np.float64
            cube = np.empty((n, ny, nx), dtype=dtype)

        cube[i] = frame.astype(cube.dtype, copy=False)
        utc_list.append(str(hdr.get("DATE-OBS", utc)))
        jd_list.append(float(hdr.get("JD-OBS", np.nan)))

        if (i + 1) % 24 == 0 or i == 0 or i == n - 1:
            print(f"[{i+1:4d}/{n}] {utc}")

    assert cube is not None

    h = fits.Header()
    h["RUNMODE"] = ("movie_hourly", "Hourly movie cube")
    h["BUNIT"] = ("I/F", "Radiance factor")
    h["NFRAMES"] = (int(n), "Number of frames")
    h["DT_HR"] = (int(step), "Step hours")
    h["UTC0"] = (utc_list[0], "First frame UTC")
    h["UTCN"] = (utc_list[-1], "Last frame UTC")
    h["JD0"] = (float(jd_list[0]), "First frame JD")
    h["JDN"] = (float(jd_list[-1]), "Last frame JD")
    h["AXIS3"] = ("time", "Frame axis")
    h.add_history("Frame i uses UTC = UTC0 + i*DT_HR hours.")
    h.add_history(f"Rendered from config: {cfg}")

    fits.PrimaryHDU(data=cube, header=h).writeto(out_path, overwrite=True, output_verify="silentfix")
    print(f"Wrote movie cube: {out_path} shape={cube.shape} dtype={cube.dtype}")


if __name__ == "__main__":
    main()

