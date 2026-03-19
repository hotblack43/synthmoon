#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot Earth RGB channel ratios against JD from a CSV table."
    )
    ap.add_argument("--csv", required=True, help="Input CSV from extract_earth_rgb_sums.py")
    ap.add_argument(
        "--out-prefix",
        required=True,
        help=(
            "Output prefix for PNG files. The script writes "
            "_ratio_combined.png, _ratio_split.png, _logratio_combined.png, and _logratio_split.png."
        ),
    )
    ap.add_argument(
        "--title",
        default="Earth RGB Channel Ratios vs Julian Day",
        help="Figure title prefix",
    )
    return ap.parse_args()


def load_rows(path: Path) -> dict[int, dict[str, np.ndarray]]:
    grouped: dict[int, dict[str, list[float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            atm = int(row["atm_flag"])
            bucket = grouped.setdefault(atm, {"jd": [], "r": [], "g": [], "b": []})
            bucket["jd"].append(float(row["jd"]))
            bucket["r"].append(float(row["sum_r"]))
            bucket["g"].append(float(row["sum_g"]))
            bucket["b"].append(float(row["sum_b"]))
    out: dict[int, dict[str, np.ndarray]] = {}
    for atm, bucket in grouped.items():
        out[atm] = {k: np.asarray(v, dtype=np.float64) for k, v in bucket.items()}
    return out


def rgb_ratios(series: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    r = np.maximum(series["r"], 1.0e-300)
    g = np.maximum(series["g"], 1.0e-300)
    b = np.maximum(series["b"], 1.0e-300)
    return {
        "R/G": r / g,
        "R/B": r / b,
        "G/B": g / b,
    }


def rgb_log_ratios(series: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ratios = rgb_ratios(series)
    return {name: np.log(vals) for name, vals in ratios.items()}


def rgb_magnitude_colors(series: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ratios = rgb_ratios(series)
    return {name: -2.5 * np.log10(vals) for name, vals in ratios.items()}


def style_axes(ax: plt.Axes, ylabel: str, reference: float) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.axhline(reference, color="0.35", linewidth=0.9, alpha=0.7)


def common_ylim(data_series: list[np.ndarray], reference: float) -> tuple[float, float]:
    vals = np.concatenate([np.ravel(x) for x in data_series if x.size > 0])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (reference - 1.0, reference + 1.0)
    ymin = float(np.min(vals))
    ymax = float(np.max(vals))
    if ymin == ymax:
        pad = max(abs(ymin) * 0.05, 1.0e-6)
        return (ymin - pad, ymax + pad)
    span = ymax - ymin
    pad = 0.05 * span
    ymin = min(ymin - pad, reference)
    ymax = max(ymax + pad, reference)
    return (ymin, ymax)


def panel_label(pair_name: str, ylabel: str) -> str:
    if ylabel == "Colour Index":
        if pair_name == "R/G":
            return r"$m_R - m_G$"
        if pair_name == "R/B":
            return r"$m_R - m_B$"
        if pair_name == "G/B":
            return r"$m_G - m_B$"
        return pair_name
    if ylabel == "Ratio":
        return pair_name
    return f"ln({pair_name})"


def make_combined_plot(
    data: dict[int, dict[str, np.ndarray]],
    out_path: Path,
    title: str,
    transform,
    ylabel: str,
    reference: float,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    pair_colors = {"R/G": "#d1495b", "R/B": "#00798c", "G/B": "#edae49"}
    atm_styles = {
        1: ("Atmosphere on", "-", 2.0, 0.95),
        0: ("Atmosphere off", "--", 1.6, 0.9),
    }
    all_series: list[np.ndarray] = []
    for atm_flag in [1, 0]:
        if atm_flag not in data:
            continue
        vals = transform(data[atm_flag])
        for pair_name in ["R/G", "R/B", "G/B"]:
            all_series.append(vals[pair_name])
    ylim = common_ylim(all_series, reference)

    for ax, pair_name in zip(axes, ["R/G", "R/B", "G/B"]):
        for atm_flag in [1, 0]:
            if atm_flag not in data:
                continue
            vals = transform(data[atm_flag])
            label, linestyle, linewidth, alpha = atm_styles[atm_flag]
            ax.plot(
                data[atm_flag]["jd"],
                vals[pair_name],
                color=pair_colors[pair_name],
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
            )
        style_axes(ax, panel_label(pair_name, ylabel), reference)
        ax.set_ylim(*ylim)
        ax.legend(loc="best", frameon=True)

    axes[-1].set_xlabel("Julian Day (UTC)")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_split_plot(
    data: dict[int, dict[str, np.ndarray]],
    out_path: Path,
    title: str,
    transform,
    ylabel: str,
    reference: float,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    line_colors = {"R/G": "#c1121f", "R/B": "#003049", "G/B": "#669bbc"}
    atm_titles = {1: "Atmosphere on", 0: "Atmosphere off"}
    all_series: list[np.ndarray] = []
    for atm_flag in [1, 0]:
        if atm_flag not in data:
            continue
        vals = transform(data[atm_flag])
        for pair_name in ["R/G", "R/B", "G/B"]:
            all_series.append(vals[pair_name])
    ylim = common_ylim(all_series, reference)

    for ax, atm_flag in zip(axes, [1, 0]):
        if atm_flag not in data:
            ax.set_visible(False)
            continue
        vals = transform(data[atm_flag])
        jd = data[atm_flag]["jd"]
        for pair_name in ["R/G", "R/B", "G/B"]:
            ax.plot(
                jd,
                vals[pair_name],
                color=line_colors[pair_name],
                linewidth=1.8,
                alpha=0.95,
                label=panel_label(pair_name, ylabel),
            )
        style_axes(ax, ylabel, reference)
        ax.set_ylim(*ylim)
        ax.set_title(atm_titles[atm_flag])
        ax.legend(loc="best", ncol=3, frameon=True)

    axes[-1].set_xlabel("Julian Day (UTC)")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    data = load_rows(csv_path)
    if not data:
        raise SystemExit(f"No rows found in {csv_path}")

    make_combined_plot(
        data,
        out_prefix.with_name(out_prefix.name + "_ratio_combined.png"),
        args.title,
        rgb_ratios,
        "Ratio",
        1.0,
    )
    make_split_plot(
        data,
        out_prefix.with_name(out_prefix.name + "_ratio_split.png"),
        args.title,
        rgb_ratios,
        "Ratio",
        1.0,
    )
    make_combined_plot(
        data,
        out_prefix.with_name(out_prefix.name + "_mag_combined.png"),
        args.title + " (Colour Index)",
        rgb_magnitude_colors,
        "Colour Index",
        0.0,
    )
    make_split_plot(
        data,
        out_prefix.with_name(out_prefix.name + "_mag_split.png"),
        args.title + " (Colour Index)",
        rgb_magnitude_colors,
        "Colour Index",
        0.0,
    )

    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_ratio_combined.png')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_ratio_split.png')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_mag_combined.png')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_mag_split.png')}")


if __name__ == "__main__":
    main()
