#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot Earth RGB ratios and colour indices from a simple CSV.")
    ap.add_argument("--csv", required=True, help="Input CSV with columns jd,sum_r,sum_g,sum_b")
    ap.add_argument("--out-prefix", required=True, help="Output file prefix")
    return ap.parse_args()


def load_rows(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cols = {"jd": [], "sum_r": [], "sum_g": [], "sum_b": []}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"jd", "sum_r", "sum_g", "sum_b"}
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required - fieldnames)
        if missing:
            raise SystemExit("CSV is missing required columns: " + ", ".join(missing))
        for row in reader:
            cols["jd"].append(float(row["jd"]))
            cols["sum_r"].append(float(row["sum_r"]))
            cols["sum_g"].append(float(row["sum_g"]))
            cols["sum_b"].append(float(row["sum_b"]))
    order = np.argsort(np.asarray(cols["jd"], dtype=float))
    jd = np.asarray(cols["jd"], dtype=float)[order]
    r = np.asarray(cols["sum_r"], dtype=float)[order]
    g = np.asarray(cols["sum_g"], dtype=float)[order]
    b = np.asarray(cols["sum_b"], dtype=float)[order]
    return jd, r, g, b


def common_limits(series: list[np.ndarray]) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(x, dtype=float) for x in series])
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    pad = 0.05 * (hi - lo) if hi > lo else 0.05 * max(abs(lo), 1.0)
    return lo - pad, hi + pad


def plot_three_panels(
    jd: np.ndarray,
    ys: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    title: str,
    out_path: Path,
    free_scale_indices: set[int] | None = None,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), sharex=True)
    ylo, yhi = common_limits(ys)
    fig.suptitle(title)
    free_scale_indices = free_scale_indices or set()
    for i, (ax, y, label, color) in enumerate(zip(axes, ys, labels, colors)):
        ax.plot(jd, y, color=color, linewidth=1.7)
        ax.set_ylabel(label)
        if i not in free_scale_indices:
            ax.set_ylim(ylo, yhi)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Julian Day (UTC)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_plots(csv_path: Path, out_prefix: Path) -> tuple[Path, Path]:
    jd, r, g, b = load_rows(Path(csv_path))
    tiny = 1.0e-300
    rg = r / np.maximum(g, tiny)
    rb = r / np.maximum(b, tiny)
    gb = g / np.maximum(b, tiny)

    ratio_path = out_prefix.with_name(out_prefix.name + "_ratio_combined.png")
    plot_three_panels(
        jd,
        [rg, rb, gb],
        ["R/G", "R/B", "G/B"],
        ["#e84a5f", "#15607a", "#f2a93b"],
        "Earth RGB Channel Ratios vs Julian Day",
        ratio_path,
    )
    mag_path = out_prefix.with_name(out_prefix.name + "_mag_combined.png")
    plot_three_panels(
        jd,
        [
            -2.5 * np.log10(np.maximum(rg, tiny)),
            -2.5 * np.log10(np.maximum(rb, tiny)),
            -2.5 * np.log10(np.maximum(np.maximum(b, tiny) / np.maximum(g, tiny), tiny)),
        ],
        [r"$m_R - m_G$", r"$m_R - m_B$", r"$m_B - m_G$"],
        ["#e84a5f", "#1b9db7", "#f2a93b"],
        "Earth RGB Channel Ratios vs Julian Day (Colour Index)",
        mag_path,
        free_scale_indices={2},
    )
    return ratio_path, mag_path


def main() -> None:
    args = parse_args()
    ratio_path, mag_path = generate_plots(Path(args.csv), Path(args.out_prefix))
    print(f"Wrote: {ratio_path}")
    print(f"Wrote: {mag_path}")


if __name__ == "__main__":
    main()
