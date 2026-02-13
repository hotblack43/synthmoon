#!/usr/bin/env python3
"""
Download a minimal set of NAIF generic kernels needed for Earth/Moon/Sun geometry.

Writes into KERNELS/ with subdirectories lsk/, pck/, spk/planets/ and generates a meta-kernel KERNELS/generic.tm.

Kernels:
- lsk/naif0012.tls
- pck/pck00011.tpc
- spk/planets/de442s.bsp

Run:
  uv run python scripts/download_kernels.py
"""
from __future__ import annotations
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"

URLS = [
    f"{BASE}/lsk/naif0012.tls",
    f"{BASE}/pck/pck00011.tpc",
    f"{BASE}/spk/planets/de442s.bsp",
]

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"OK (exists): {dest}")
        return
    ensure_parent(dest)
    print(f"Downloading: {url}")
    urlretrieve(url, dest)
    print(f"Wrote: {dest} ({dest.stat().st_size/1e6:.1f} MB)")

def write_meta_kernel(kernels_dir: Path, mk_path: Path) -> None:
    # Defines $KERNELS so paths like '$KERNELS/lsk/naif0012.tls' work.
    text = """KPL/MK

\\begindata

PATH_VALUES  = ( 'KERNELS' )
PATH_SYMBOLS = ( 'KERNELS' )

KERNELS_TO_LOAD = (
   '$KERNELS/lsk/naif0012.tls'
   '$KERNELS/pck/pck00011.tpc'
   '$KERNELS/spk/planets/de442s.bsp'
)

\\begintext

"""
    ensure_parent(mk_path)
    mk_path.write_text(text, encoding="utf-8")
    print(f"Wrote meta-kernel: {mk_path}")

def main() -> None:
    kernels_dir = Path("KERNELS")
    dests = [
        kernels_dir / "lsk" / "naif0012.tls",
        kernels_dir / "pck" / "pck00011.tpc",
        kernels_dir / "spk" / "planets" / "de442s.bsp",
    ]
    for url, dest in zip(URLS, dests):
        download(url, dest)
    write_meta_kernel(kernels_dir, kernels_dir / "generic.tm")

if __name__ == "__main__":
    main()
