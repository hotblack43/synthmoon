#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from html import unescape
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


BASE_URL = "https://noaadata.apps.nsidc.org/NOAA/G02202_V5/"
USER_AGENT = "synthmoon-nsidc-downloader/0.2"
HTML_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
REV_RE = re.compile(r"_v(\d{2})r(\d{2})", re.IGNORECASE)


def _parse_utc(s: str) -> dt.datetime:
    t = s.strip().replace("Z", "+00:00")
    x = dt.datetime.fromisoformat(t)
    if x.tzinfo is None:
        x = x.replace(tzinfo=dt.timezone.utc)
    return x.astimezone(dt.timezone.utc)


def _isoz(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _request_headers(token: str | None) -> dict[str, str]:
    hdr = {"User-Agent": USER_AGENT}
    if token:
        hdr["Authorization"] = f"Bearer {token}"
    return hdr


def _http_get_text(url: str, token: str | None) -> str:
    req = Request(url, headers=_request_headers(token))
    with urlopen(req) as r:
        return r.read().decode("utf-8", errors="replace")


def _http_download(url: str, out_path: Path, token: str | None) -> None:
    req = Request(url, headers=_request_headers(token))
    with urlopen(req) as r:
        data = r.read()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)


def _candidate_listing_urls(day: dt.datetime, hemisphere: str) -> list[str]:
    month_num = day.strftime("%m")
    month_abbr = day.strftime("%b")
    month_full = day.strftime("%B")
    year = day.strftime("%Y")
    base = f"{BASE_URL}{hemisphere}/daily/"
    return [
        f"{base}{year}/",
        f"{base}{year}/{month_num}/",
        f"{base}{year}/{month_num}_{month_abbr}/",
        f"{base}{year}/{month_num}_{month_abbr.upper()}/",
        f"{base}{year}/{month_num}_{month_full}/",
        f"{base}{year}/{month_num}_{month_full.upper()}/",
    ]


def _extract_links(html: str, base_url: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for href in HTML_HREF_RE.findall(html):
        href = unescape(href).strip()
        if not href or href.startswith("#"):
            continue
        full = urljoin(base_url, href)
        if full in seen:
            continue
        seen.add(full)
        out.append(full)
    return out


def _revision_key(name: str) -> tuple[int, int, str]:
    m = REV_RE.search(name)
    if not m:
        return (-1, -1, name)
    return (int(m.group(1)), int(m.group(2)), name)


def _pick_daily_file(links: list[str], day: dt.datetime, hemisphere: str) -> str | None:
    tag = day.strftime("%Y%m%d")
    hemi_tag = "sic_psn25_" if hemisphere == "north" else "sic_pss25_"
    files = []
    for link in links:
        name = Path(link.split("?", 1)[0]).name
        low = name.lower()
        if tag not in low:
            continue
        if not low.endswith((".nc", ".nc4", ".bin")):
            continue
        if hemi_tag not in low:
            continue
        files.append(link)
    if not files:
        return None
    files.sort(key=lambda x: _revision_key(Path(x.split("?", 1)[0]).name.lower()), reverse=True)
    return files[0]


def _find_daily_url(day: dt.datetime, hemisphere: str, token: str | None) -> tuple[str, str]:
    errors: list[str] = []
    for listing_url in _candidate_listing_urls(day, hemisphere):
        try:
            html = _http_get_text(listing_url, token)
        except HTTPError as e:
            errors.append(f"{listing_url} -> HTTP {e.code}")
            continue
        except URLError as e:
            errors.append(f"{listing_url} -> {e.reason}")
            continue
        links = _extract_links(html, listing_url)
        pick = _pick_daily_file(links, day, hemisphere)
        if pick is not None:
            return pick, listing_url
        errors.append(f"{listing_url} -> no matching daily file")
    raise SystemExit(
        f"Could not find a G02202 daily file for {hemisphere} on {_isoz(day)}.\n"
        + "\n".join(errors)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Download NSIDC/NOAA G02202_V5 daily sea-ice files from NOAA@NSIDC HTTPS.")
    ap.add_argument("--utc", required=True, help="Target UTC, e.g. 2011-07-06T06:21:47Z")
    ap.add_argument(
        "--hemisphere",
        choices=("north", "south", "both"),
        default="both",
        help="Which hemisphere file(s) to download",
    )
    ap.add_argument("--out-dir", default="DATA/NSIDC", help="Download directory")
    ap.add_argument("--token", default=None, help="Earthdata bearer token; defaults to EARTHDATA_TOKEN env var")
    args = ap.parse_args()

    token = args.token or os.getenv("EARTHDATA_TOKEN", "").strip() or None
    if token is None:
        print("Warning: no Earthdata token supplied. Download may fail.", file=sys.stderr)

    target = _parse_utc(args.utc)
    hemispheres = ("north", "south") if args.hemisphere == "both" else (args.hemisphere,)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for hemi in hemispheres:
        daily_url, listing_url = _find_daily_url(target, hemi, token)
        fname = Path(daily_url.split("?", 1)[0]).name
        out_path = out_dir / fname
        _http_download(daily_url, out_path, token)
        print(f"hemisphere: {hemi}")
        print(f"listing: {listing_url}")
        print(f"url: {daily_url}")
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
