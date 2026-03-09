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


def _parse_utc(s: str) -> dt.datetime:
    t = s.strip().replace("Z", "+00:00")
    x = dt.datetime.fromisoformat(t)
    if x.tzinfo is None:
        x = x.replace(tzinfo=dt.timezone.utc)
    return x.astimezone(dt.timezone.utc)


def _isoz(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _find_download_url(entry: dict) -> str | None:
    links = entry.get("links", [])
    for ln in links:
        href = str(ln.get("href", "")).strip()
        rel = str(ln.get("rel", "")).strip()
        title = str(ln.get("title", "")).lower()
        if not href:
            continue
        if ln.get("inherited", False):
            continue
        if href.endswith((".hdf", ".h5", ".nc")):
            return href
        if "data#" in rel or "download" in title:
            return href
    return None


def _http_get_json(url: str) -> dict:
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "synthmoon-modis-downloader/0.1"})
    with urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))


def _http_download(url: str, out_path: Path, bearer_token: str | None) -> None:
    headers = {"User-Agent": "synthmoon-modis-downloader/0.1"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    req = Request(url, headers=headers)
    with urlopen(req) as r:
        data = r.read()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download one MODIS cloud granule/file nearest a UTC using NASA CMR.")
    ap.add_argument("--utc", required=True, help="Target UTC, e.g. 2011-07-15T12:00:00Z")
    ap.add_argument(
        "--product",
        default="MOD06_L2",
        choices=["MOD06_L2", "MYD06_L2", "MOD08_D3", "MYD08_D3"],
        help="MODIS product short name",
    )
    ap.add_argument("--version", default=None, help="Optional collection version filter (e.g. 061).")
    ap.add_argument("--window-min", type=int, default=60, help="Search half-window in minutes around UTC")
    ap.add_argument("--max-results", type=int, default=50, help="CMR page_size")
    ap.add_argument("--out-dir", default="DATA/MODIS", help="Download directory")
    ap.add_argument(
        "--token",
        default=None,
        help="Earthdata bearer token. If omitted, uses EARTHDATA_TOKEN env var.",
    )
    args = ap.parse_args()

    target = _parse_utc(args.utc)
    is_daily = str(args.product).endswith("_D3")
    if is_daily:
        day0 = target.replace(hour=0, minute=0, second=0, microsecond=0)
        t0 = day0
        t1 = day0 + dt.timedelta(days=1)
    else:
        t0 = target - dt.timedelta(minutes=int(args.window_min))
        t1 = target + dt.timedelta(minutes=int(args.window_min))
    temporal = f"{_isoz(t0)},{_isoz(t1)}"

    q = {
        "short_name": args.product,
        "temporal": temporal,
        "page_size": int(args.max_results),
        "sort_key[]": "start_date",
    }
    if args.version is not None and str(args.version).strip():
        q["version"] = str(args.version).strip()
    url = f"{CMR_GRANULES_URL}?{urlencode(q, doseq=True)}"
    payload = _http_get_json(url)
    entries = payload.get("feed", {}).get("entry", [])
    if not entries:
        raise SystemExit(f"No {args.product} granules in {temporal}")

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
        raise SystemExit("No downloadable link found in selected granule entry.")

    gid = str(best.get("producer_granule_id") or best.get("title") or "granule")
    fname = Path(durl.split("?")[0]).name
    if not fname:
        fname = f"{gid}.dat"
    out_dir = Path(args.out_dir)
    out_path = out_dir / fname

    token = args.token or os.getenv("EARTHDATA_TOKEN", "").strip()
    if not token:
        print("Warning: no Earthdata token supplied. Download may fail for protected endpoints.", file=sys.stderr)

    _http_download(durl, out_path, token if token else None)

    print(f"Selected granule: {gid}")
    print(f"time_start: {best.get('time_start')}")
    print(f"distance_to_target_s: {best_dt:.1f}")
    print(f"url: {durl}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
