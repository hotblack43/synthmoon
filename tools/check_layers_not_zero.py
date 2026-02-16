#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

# Layers allowed to be all-zero by design in some configurations.
# Keep this list small so unexpected regressions are still caught.
ZERO_LAYER_WHITELIST = {
    "IF_EARTH",
    "RAD_EAR",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run one synthmoon render and fail if any produced FITS layer is identically zero "
            "(unless whitelisted)."
        )
    )
    ap.add_argument("--config", default="scene.toml", help="Path to scene TOML")
    ap.add_argument("--utc", default=None, help="Optional UTC override passed to renderer")
    ap.add_argument(
        "--require-earthlight-layers-nonzero",
        action="store_true",
        help="Require IF_EARTH (layer 4) and RAD_EAR (layer 7) to be non-zero.",
    )
    return ap.parse_args()


def layer_names_from_header(hdr: fits.Header, nz: int) -> list[str]:
    names: list[str] = []
    for i in range(1, nz + 1):
        names.append(str(hdr.get(f"LAY{i}", f"LAYER{i}")))
    return names


def read_include_earthlight(cfg_path: Path) -> bool:
    with cfg_path.open("rb") as f:
        raw = tomllib.load(f)
    return bool(raw.get("illumination", {}).get("include_earthlight", True))


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = Path(args.config)
    if not cfg.is_absolute():
        cfg = repo_root / cfg
    if not cfg.exists():
        print(f"ERROR: config not found: {cfg}", file=sys.stderr)
        return 2
    include_earthlight = read_include_earthlight(cfg)

    with tempfile.TemporaryDirectory(prefix="synthmoon_layercheck_") as tmpdir:
        out_path = Path(tmpdir) / "layer_check.fits"
        cmd = [
            sys.executable,
            "-m",
            "synthmoon.run_v0",
            "--config",
            str(cfg),
            "--out",
            str(out_path),
        ]
        if args.utc:
            cmd += ["--utc", str(args.utc)]
        subprocess.run(cmd, check=True, cwd=repo_root)

        with fits.open(out_path, memmap=False) as hdul:
            data = np.asarray(hdul[0].data)
            hdr = hdul[0].header

    if data.ndim == 2:
        data = data[np.newaxis, ...]
        names = [str(hdr.get("LAY1", "PRIMARY"))]
    elif data.ndim == 3:
        names = layer_names_from_header(hdr, data.shape[0])
    else:
        print(f"ERROR: expected 2D or 3D primary FITS data, got shape={data.shape}", file=sys.stderr)
        return 2

    zero_layers: list[str] = []
    for i, name in enumerate(names):
        if np.all(data[i] == 0):
            zero_layers.append(name)

    print("Produced layers:", ", ".join(names))

    offending = [name for name in zero_layers if name not in ZERO_LAYER_WHITELIST]
    if include_earthlight and args.require_earthlight_layers_nonzero:
        for must_have in ("IF_EARTH", "RAD_EAR"):
            if must_have in names and must_have in zero_layers and must_have not in offending:
                offending.append(must_have)

    if offending:
        reason = "non-whitelisted layers are identically zero"
        if include_earthlight and args.require_earthlight_layers_nonzero:
            reason += " (earthlight is enabled, so IF_EARTH/RAD_EAR must be non-zero)"
        print(
            "FAIL: " + reason + ": "
            + ", ".join(offending)
            + f". Whitelist={sorted(ZERO_LAYER_WHITELIST)}",
            file=sys.stderr,
        )
        return 1

    print("PASS: no non-whitelisted layer is identically zero.")
    if zero_layers:
        print("Zero layers accepted by whitelist:", ", ".join(zero_layers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
