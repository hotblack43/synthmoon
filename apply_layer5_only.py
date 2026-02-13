#!/usr/bin/env python3
# apply_layer5_only.py
#
# Edits your existing SYNTHMOON runner in-place to add:
#   --layer5-only  (write only layer 5 to FITS cube; default writes all layers)
#
# It tries to find the runner file automatically under ./synthmoon/.
# It makes a .bak backup of any modified file.
#
# Usage (from repo root):
#   python apply_layer5_only.py
#
# Then run:
#   PYTHONPATH=$PWD uv run python -m synthmoon.run_v0 --config scene.toml --out out.fits --layer5-only

from __future__ import annotations
import re
import sys
from pathlib import Path

FLAG_BLOCK = """
    p.add_argument(
        "--layer5-only",
        action="store_true",
        help="Write only layer 5 (1-based) to the FITS cube output (default writes all layers).",
    )
""".lstrip("\n")

HELPER_BLOCK = """
def _select_only_layer5(cube, layer_names):
    \"\"\"
    Return (cube5, names5) where cube5 contains only layer 5 (1-based).
    Supports cube shapes (ny, nx, nlayers) or (nlayers, ny, nx).
    \"\"\"
    if cube is None:
        raise ValueError("cube is None")
    if layer_names is None:
        layer_names = []

    # (ny, nx, nlayers)
    if getattr(cube, "ndim", None) == 3 and cube.shape[-1] >= 1:
        nl = cube.shape[-1]
        axis_last = True
    # (nlayers, ny, nx)
    elif getattr(cube, "ndim", None) == 3 and cube.shape[0] >= 1:
        nl = cube.shape[0]
        axis_last = False
    else:
        raise ValueError(f"Unexpected cube shape: {getattr(cube, 'shape', None)}")

    if nl < 5:
        raise ValueError(f"Requested layer 5, but cube has only {nl} layer(s).")

    idx0 = 4  # layer 5 in 0-based indexing
    if axis_last:
        cube5 = cube[..., idx0:idx0+1]
    else:
        cube5 = cube[idx0:idx0+1, ...]

    if len(layer_names) >= 5:
        names5 = [layer_names[4]]
    else:
        names5 = ["LAYER5"]
    return cube5, names5
""".lstrip("\n")

def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def info(msg: str) -> None:
    print(msg)

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def write_text(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def find_runner_candidates(repo_root: Path) -> list[Path]:
    synth = repo_root / "synthmoon"
    if not synth.is_dir():
        return []
    cands: list[Path] = []
    for p in synth.rglob("*.py"):
        try:
            t = read_text(p)
        except Exception:
            continue
        if ("argparse" in t) and ("def main" in t) and ("--config" in t or "config" in t) and ("--out" in t or "out" in t):
            cands.append(p)
    return cands

def choose_best_candidate(cands: list[Path]) -> Path | None:
    if not cands:
        return None
    for name in ("run_v0.py", "run.py", "__main__.py"):
        for p in cands:
            if p.name == name:
                return p
    cands_sorted = sorted(cands, key=lambda p: (len(p.parts), str(p)))
    return cands_sorted[0]

def insert_flag_into_parser(t: str) -> str:
    if "--layer5-only" in t:
        return t

    m = re.search(r"def\s+build_arg_parser\s*\([^)]*\)\s*->\s*argparse\.ArgumentParser\s*:\s*(?:\n|\r\n)", t)
    if not m:
        m = re.search(r"def\s+build_arg_parser\s*\([^)]*\)\s*:\s*(?:\n|\r\n)", t)
    if not m:
        return t

    start = m.start()
    window = t[start:]
    rm = re.search(r"^(\s*)return\s+p\s*$", window, flags=re.MULTILINE)
    if not rm:
        return t

    indent = rm.group(1)
    block = "\n".join(indent + line if line.strip() else line for line in FLAG_BLOCK.splitlines()) + "\n\n"
    insert_at = start + rm.start()
    return t[:insert_at] + block + t[insert_at:]

def insert_helper_before_main(t: str) -> str:
    if "def _select_only_layer5" in t:
        return t
    mm = re.search(r"^(\s*)def\s+main\s*\(", t, flags=re.MULTILINE)
    if not mm:
        return t
    insert_at = mm.start()
    return t[:insert_at] + HELPER_BLOCK + "\n\n" + t[insert_at:]

def insert_filter_after_render(t: str) -> str:
    if "args.layer5_only" in t:
        return t

    pat = r"^(\s*)(cube\s*,\s*layer_names\s*(?:,\s*[^=]+)?=\s*render_[A-Za-z0-9_]+\([^\n]*\))\s*$"
    m = re.search(pat, t, flags=re.MULTILINE)
    if not m:
        pat2 = r"^(\s*)(cube\s*,\s*layer_names\s*(?:,\s*[^=]+)?=\s*[A-Za-z0-9_]+\([^\n]*\))\s*$"
        m = re.search(pat2, t, flags=re.MULTILINE)
    if not m:
        return t

    indent = m.group(1)
    insert_at = m.end()
    block = (
        "\n"
        f"{indent}# Optional: only keep layer 5\n"
        f"{indent}if args.layer5_only:\n"
        f"{indent}    cube, layer_names = _select_only_layer5(cube, layer_names)\n"
    )
    return t[:insert_at] + block + t[insert_at:]

def apply_to_file(p: Path) -> bool:
    orig = read_text(p)
    t = orig
    t = insert_flag_into_parser(t)
    t = insert_helper_before_main(t)
    t = insert_filter_after_render(t)

    if t == orig:
        return False

    if ("--layer5-only" not in t) or ("def _select_only_layer5" not in t) or ("args.layer5_only" not in t):
        die(f"Internal error: edits incomplete for {p}")

    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        write_text(bak, orig)
    write_text(p, t)
    return True

def main() -> int:
    repo_root = Path(".").resolve()
    cands = find_runner_candidates(repo_root)
    if not cands:
        die("Could not find any runner candidates under ./synthmoon/. Are you in the repo root?")
    target = choose_best_candidate(cands)
    if target is None:
        die("Could not choose a runner file automatically.")
    info(f"Target runner file: {target}")
    changed = apply_to_file(target)
    if changed:
        info("✅ Applied edits (backup saved as .bak).")
        info("Run e.g.:")
        info("  PYTHONPATH=$PWD uv run python -m synthmoon.run_v0 --config scene.toml --out out_layer5.fits --layer5-only")
    else:
        info("No changes made (maybe already applied?).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
