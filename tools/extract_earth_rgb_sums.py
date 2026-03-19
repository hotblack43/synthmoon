#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from astropy.io import fits


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract JD and RGB radiance sums from Earth FITS products into a CSV table."
    )
    ap.add_argument(
        "--fits-dir",
        action="append",
        required=True,
        help="Directory containing Earth FITS files. Repeat for multiple directories.",
    )
    ap.add_argument(
        "--atm-flag",
        action="append",
        required=True,
        help="Atmosphere flag (for example 1 or 0) matching each --fits-dir in order.",
    )
    ap.add_argument(
        "--glob",
        default="earth_*.fits",
        help="Filename glob inside each fits directory (default: earth_*.fits).",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output CSV path.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.fits_dir) != len(args.atm_flag):
        raise SystemExit("Repeat --atm-flag once for each --fits-dir.")

    rows: list[tuple[float, float, float, float, int, str]] = []
    for fits_dir_raw, atm_flag_raw in zip(args.fits_dir, args.atm_flag):
        fits_dir = Path(fits_dir_raw)
        atm_flag = int(atm_flag_raw)
        if not fits_dir.is_dir():
            raise SystemExit(f"Not a directory: {fits_dir}")

        for path in sorted(fits_dir.glob(args.glob)):
            hdr = fits.getheader(path)
            try:
                jd = float(hdr["JD-OBS"])
                r = float(hdr["SUMRADR"])
                g = float(hdr["SUMRADG"])
                b = float(hdr["SUMRADB"])
            except KeyError as exc:
                raise SystemExit(f"Missing required header key {exc} in {path}") from exc
            rows.append((jd, r, g, b, atm_flag, str(path)))

    rows.sort(key=lambda x: (x[0], x[4], x[5]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["jd", "sum_r", "sum_g", "sum_b", "atm_flag"])
        for jd, r, g, b, atm_flag, _src in rows:
            w.writerow([f"{jd:.7f}", f"{r:.15g}", f"{g:.15g}", f"{b:.15g}", atm_flag])

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
