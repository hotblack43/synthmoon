#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"


def _isoz(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_utc(s: str) -> dt.datetime:
    t = s.strip().replace("Z", "+00:00")
    x = dt.datetime.fromisoformat(t)
    if x.tzinfo is None:
        x = x.replace(tzinfo=dt.timezone.utc)
    return x.astimezone(dt.timezone.utc)


def _find_download_url(entry: dict) -> str | None:
    for ln in entry.get("links", []):
        href = str(ln.get("href", "")).strip()
        rel = str(ln.get("rel", "")).strip()
        title = str(ln.get("title", "")).lower()
        if not href or ln.get("inherited", False):
            continue
        if href.endswith((".hdf", ".h5", ".nc")):
            return href
        if "data#" in rel or "download" in title:
            return href
    return None


def _http_get_json(url: str) -> dict:
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "synthmoon-modis-landcover/0.1"})
    with urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))


def _http_download(url: str, out_path: Path, bearer_token: str | None) -> None:
    headers = {"User-Agent": "synthmoon-modis-landcover/0.1"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    req = Request(url, headers=headers)
    with urlopen(req) as r:
        data = r.read()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download one annual MODIS land-cover file (MCD12C1/MCD12Q1) nearest a year.")
    ap.add_argument("--year", type=int, default=None, help="Target year, e.g. 2011")
    ap.add_argument("--utc", default=None, help="Alternative to --year; target UTC from which the year is taken")
    ap.add_argument("--product", default="MCD12C1", choices=["MCD12C1", "MCD12Q1"], help="MODIS land-cover product")
    ap.add_argument("--version", default="061", help="Collection version filter (default: 061)")
    ap.add_argument("--max-results", type=int, default=50, help="CMR page_size")
    ap.add_argument("--out-dir", default="DATA/MODIS", help="Download directory")
    ap.add_argument("--token", default=None, help="Earthdata bearer token; defaults to EARTHDATA_TOKEN env var")
    args = ap.parse_args()

    if args.year is None and args.utc is None:
        raise SystemExit("One of --year or --utc is required.")
    year = int(args.year) if args.year is not None else int(_parse_utc(str(args.utc)).year)
    t0 = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
    t1 = dt.datetime(year + 1, 1, 1, tzinfo=dt.timezone.utc)

    q = {
        "short_name": args.product,
        "temporal": f"{_isoz(t0)},{_isoz(t1)}",
        "page_size": int(args.max_results),
        "sort_key[]": "start_date",
    }
    if args.version:
        q["version"] = str(args.version).strip()
    payload = _http_get_json(f"{CMR_GRANULES_URL}?{urlencode(q, doseq=True)}")
    entries = payload.get("feed", {}).get("entry", [])
    if not entries:
        raise SystemExit(f"No {args.product} entries found for year {year}")

    target = dt.datetime(year, 7, 2, tzinfo=dt.timezone.utc)
    best = None
    best_dt = None
    for e in entries:
        ts = e.get("time_start")
        if not ts:
            continue
        try:
            et = _parse_utc(ts)
        except Exception:
            continue
        dsec = abs((et - target).total_seconds())
        if best is None or dsec < best_dt:
            best = e
            best_dt = dsec
    if best is None:
        raise SystemExit("Found entries, but none with parseable time_start.")

    durl = _find_download_url(best)
    if not durl:
        raise SystemExit("No downloadable link found in selected land-cover entry.")

    gid = str(best.get("producer_granule_id") or best.get("title") or "granule")
    fname = Path(durl.split("?", 1)[0]).name or f"{gid}.dat"
    out_path = Path(args.out_dir) / fname
    token = args.token or os.getenv("EARTHDATA_TOKEN", "").strip()
    if not token:
        print("Warning: no Earthdata token supplied. Download may fail.", file=sys.stderr)

    _http_download(durl, out_path, token if token else None)
    print(f"Selected granule: {gid}")
    print(f"time_start: {best.get('time_start')}")
    print(f"distance_to_target_s: {best_dt:.1f}")
    print(f"url: {durl}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
