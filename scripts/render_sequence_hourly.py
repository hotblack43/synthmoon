#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import tomllib  # py>=3.11
except Exception:
    import tomli as tomllib  # type: ignore


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


def safe_stamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="scene.toml", help="Path to scene.toml")
    ap.add_argument("--outdir", default="OUTPUT/SEQ_24H", help="Directory for output FITS files")
    ap.add_argument("--hours", type=int, default=24, help="Number of frames to render (default 24)")
    ap.add_argument("--step-hours", type=int, default=1, help="Time step in hours (default 1)")
    ap.add_argument("--start-utc", default=None,
                    help="Override start time (ISO8601). If omitted, uses [time].utc from config.")
    ap.add_argument("--only-layer-index", type=int, default=5,
                    help="Write only this layer index (1-based). Use 0 to write all layers.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Missing config: {cfg_path}")

    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)

    start_utc = args.start_utc or cfg.get("time", {}).get("utc", None)
    if not start_utc:
        raise SystemExit("No start UTC found. Set [time].utc in scene.toml or pass --start-utc")

    t0 = parse_utc(start_utc)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hours = int(args.hours)
    step = int(args.step_hours)

    print(f"Config : {cfg_path}")
    print(f"Start  : {fmt_utc(t0)}")
    print(f"Frames : {hours}  step={step}h")
    print(f"Outdir : {outdir}")
    if args.only_layer_index and args.only_layer_index > 0:
        print(f"Output : only layer index {args.only_layer_index}")
    else:
        print("Output : full cube")

    for i in range(hours):
        ti = t0 + timedelta(hours=i * step)
        utc_i = fmt_utc(ti)
        stamp = safe_stamp(ti)
        out_fits = outdir / f"synthmoon_{stamp}_L{args.only_layer_index if args.only_layer_index>0 else 'ALL'}.fits"

        cmd = [
            sys.executable, "-m", "synthmoon.run_v0",
            "--config", str(cfg_path),
            "--utc", utc_i,
            "--out", str(out_fits),
        ]
        if args.only_layer_index and args.only_layer_index > 0:
            cmd += ["--only-layer-index", str(args.only_layer_index)]

        print(f"[{i+1:02d}/{hours:02d}] {utc_i} -> {out_fits.name}")
        subprocess.run(cmd, check=True)

    print("Done.")


if __name__ == "__main__":
    main()

