#!/usr/bin/env python3
"""Generate plots for paired long-rollout VBench analysis.

Run after:

  python experiments/keyframe_budget/scripts/aggregate_long_rollout_vbench.py
  python experiments/keyframe_budget/scripts/analyze_long_rollout_vbench_paired.py

Example:

  python experiments/keyframe_budget/scripts/plot_long_rollout_vbench_paired.py

Outputs are written to:

  outputs/long_keyframe_budget/vbench_analysis/paired_analysis/plots
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


FULL_METRICS = (
    "aesthetic_quality",
    "background_consistency",
    "imaging_quality",
    "motion_smoothness",
    "overall_consistency",
    "subject_consistency",
)

SUFFIX_METRICS = (
    "aesthetic_quality",
    "motion_smoothness",
    "subject_consistency",
)


METRIC_LABELS = {
    "aesthetic_quality": "Aesthetic Quality",
    "background_consistency": "Background Consistency",
    "dynamic_degree": "Dynamic Degree",
    "imaging_quality": "Imaging Quality",
    "motion_smoothness": "Motion Smoothness",
    "overall_consistency": "Overall Consistency",
    "subject_consistency": "Subject Consistency",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(row: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    return float(value)


def i(row: Mapping[str, Any], key: str, default: int = -1) -> int:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    return int(float(value))


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def ensure_matplotlib() -> Tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "This plotting script requires matplotlib and numpy. "
            "Run it in the Jean Zay environment where plotting dependencies are available."
        ) from exc
    return plt, np


def savefig(plt: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def plot_full_baseline_summary(rows: Sequence[Mapping[str, str]], out_dir: Path) -> None:
    plt, np = ensure_matplotlib()
    rows = [r for r in rows if r["metric"] in FULL_METRICS]
    rows = sorted(rows, key=lambda r: FULL_METRICS.index(r["metric"]))

    labels = [metric_label(r["metric"]) for r in rows]
    means = [f(r, "mean_delta") for r in rows]
    lows = [f(r, "mean_ci95_low") for r in rows]
    highs = [f(r, "mean_ci95_high") for r in rows]
    yerr = [[m - lo for m, lo in zip(means, lows)], [hi - m for m, hi in zip(means, highs)]]

    colors = ["#2f7f5f" if m > 0 else "#b64a3a" for m in means]
    x = np.arange(len(labels))
    plt.figure(figsize=(11, 4.8))
    plt.bar(x, means, yerr=yerr, capsize=4, color=colors, edgecolor="#242424", linewidth=0.8)
    plt.axhline(0, color="#222222", linewidth=1.0)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Paired delta: all_heavy - all_fast")
    plt.title("Full-video baseline anomaly: all-heavy is not a reliable upper anchor")
    plt.grid(axis="y", alpha=0.25)
    savefig(plt, out_dir / "full_baseline_heavy_minus_fast.png")


def plot_single_heavy_curves(rows: Sequence[Mapping[str, str]], out_dir: Path) -> None:
    plt, _ = ensure_matplotlib()
    for metric in FULL_METRICS:
        metric_rows = sorted([r for r in rows if r["metric"] == metric], key=lambda r: i(r, "heavy_idx"))
        if not metric_rows:
            continue
        xs = [i(r, "heavy_idx") for r in metric_rows]
        means = [f(r, "mean_delta") for r in metric_rows]
        lows = [f(r, "mean_ci95_low") for r in metric_rows]
        highs = [f(r, "mean_ci95_high") for r in metric_rows]

        plt.figure(figsize=(11, 4.2))
        plt.fill_between(xs, lows, highs, color="#d8c7aa", alpha=0.45, label="95% bootstrap CI")
        plt.plot(xs, means, marker="o", color="#1f5f6f", linewidth=1.8, markersize=4, label="mean paired delta")
        plt.axhline(0, color="#222222", linestyle="--", linewidth=1.0)
        plt.xlabel("Single heavy chunk index")
        plt.ylabel("Paired delta vs all_fast")
        plt.title(f"Full-video single-heavy effect: {metric_label(metric)}")
        plt.grid(alpha=0.25)
        plt.legend()
        savefig(plt, out_dir / f"full_single_heavy_curve_{metric}.png")


def plot_single_heavy_metric_overview(rows: Sequence[Mapping[str, str]], best_rows: Sequence[Mapping[str, str]], out_dir: Path) -> None:
    plt, np = ensure_matplotlib()
    by_metric = {r["metric"]: r for r in rows if r["metric"] in FULL_METRICS}
    best_by_metric = {r["metric"]: r for r in best_rows if r["metric"] in FULL_METRICS}
    labels = [metric_label(m) for m in FULL_METRICS]
    all_single = [f(by_metric[m], "mean_delta") for m in FULL_METRICS]
    best = [f(best_by_metric[m], "mean_delta") for m in FULL_METRICS]

    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(11, 4.8))
    plt.bar(x - width / 2, all_single, width, label="Mean over all single-heavy chunks", color="#8ab0ab")
    plt.bar(x + width / 2, best, width, label="Mean best chunk per prompt/seed", color="#d79255")
    plt.axhline(0, color="#222222", linewidth=1.0)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Paired delta vs all_fast")
    plt.title("Sparse compute signal: average placement is weak, best placement is strong")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    savefig(plt, out_dir / "full_single_heavy_average_vs_best.png")


def plot_best_chunk_histogram(rows: Sequence[Mapping[str, str]], out_dir: Path) -> None:
    plt, np = ensure_matplotlib()
    for metric in FULL_METRICS:
        metric_rows = [r for r in rows if r["metric"] == metric]
        if not metric_rows:
            continue
        counts = {i(r, "heavy_idx"): i(r, "best_count") for r in metric_rows}
        xs = list(range(40))
        ys = [counts.get(x, 0) for x in xs]
        plt.figure(figsize=(11, 3.8))
        plt.bar(xs, ys, color="#6d8ea0", edgecolor="#24343c", linewidth=0.4)
        plt.xlabel("Best single-heavy chunk index for a prompt/seed")
        plt.ylabel("Count out of 72")
        plt.title(f"Per-sample best chunk distribution: {metric_label(metric)}")
        plt.xticks(np.arange(0, 40, 4))
        plt.grid(axis="y", alpha=0.25)
        savefig(plt, out_dir / f"full_best_chunk_histogram_{metric}.png")


def plot_oracle_alignment(rows: Sequence[Mapping[str, str]], out_dir: Path) -> None:
    plt, np = ensure_matplotlib()
    rows = [r for r in rows if r["metric"] in FULL_METRICS]
    rows = sorted(rows, key=lambda r: FULL_METRICS.index(r["metric"]))
    labels = [metric_label(r["metric"]) for r in rows]
    spearman = [f(r, "mean_spearman_delta_vs_internal_gain") for r in rows]
    recall = [f(r, "mean_top10_recall_internal_oracle") for r in rows]
    x = np.arange(len(labels))

    plt.figure(figsize=(11, 4.8))
    plt.plot(x, spearman, marker="o", linewidth=2.0, color="#955f61", label="Mean Spearman vs internal gain")
    plt.plot(x, recall, marker="s", linewidth=2.0, color="#4d7c8a", label="Mean top-10 recall")
    plt.axhline(0, color="#222222", linestyle="--", linewidth=1.0)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Alignment")
    plt.title("Existing internal proxy oracle does not align with VBench key chunks")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    savefig(plt, out_dir / "full_internal_oracle_alignment.png")


def plot_suffix_heatmaps(rows: Sequence[Mapping[str, str]], out_dir: Path) -> None:
    plt, np = ensure_matplotlib()
    for metric in SUFFIX_METRICS:
        metric_rows = [r for r in rows if r["metric"] == metric]
        if not metric_rows:
            continue
        starts = sorted({i(r, "suffix_start") for r in metric_rows})
        heavy_indices = sorted({i(r, "heavy_idx") for r in metric_rows})
        value = {(i(r, "heavy_idx"), i(r, "suffix_start")): f(r, "mean_delta") for r in metric_rows}
        mat = np.full((len(heavy_indices), len(starts)), np.nan, dtype=float)
        for yi, heavy_idx in enumerate(heavy_indices):
            for xi, suffix_start in enumerate(starts):
                mat[yi, xi] = value.get((heavy_idx, suffix_start), np.nan)

        vmax = np.nanmax(np.abs(mat))
        if not math.isfinite(float(vmax)) or vmax == 0:
            vmax = 1.0
        plt.figure(figsize=(9, 5.8))
        im = plt.imshow(mat, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, label="Mean paired delta vs all_fast suffix")
        plt.xticks(range(len(starts)), starts, rotation=45)
        plt.yticks(range(len(heavy_indices)), heavy_indices)
        plt.xlabel("Suffix start chunk")
        plt.ylabel("Heavy chunk index")
        plt.title(f"Sparse suffix intervention map: {metric_label(metric)}")
        savefig(plt, out_dir / f"suffix_heatmap_{metric}.png")


def plot_suffix_relative_position(rows: Sequence[Mapping[str, str]], out_dir: Path) -> None:
    plt, np = ensure_matplotlib()
    positions = ["upstream", "same", "downstream"]
    metrics = [m for m in SUFFIX_METRICS if any(r["metric"] == m for r in rows)]
    x = np.arange(len(metrics))
    width = 0.25

    plt.figure(figsize=(10, 4.6))
    for offset, pos in enumerate(positions):
        vals = []
        for metric in metrics:
            row = next((r for r in rows if r["metric"] == metric and r["relative_position"] == pos), None)
            vals.append(f(row, "mean_delta") if row else float("nan"))
        plt.bar(x + (offset - 1) * width, vals, width, label=pos)
    plt.axhline(0, color="#222222", linewidth=1.0)
    plt.xticks(x, [metric_label(m) for m in metrics], rotation=25, ha="right")
    plt.ylabel("Mean paired delta vs all_fast suffix")
    plt.title("Suffix effects by relative intervention position")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    savefig(plt, out_dir / "suffix_relative_position_summary.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis_dir",
        type=Path,
        default=Path("outputs/long_keyframe_budget/vbench_analysis/paired_analysis"),
    )
    parser.add_argument("--plots_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir
    plots_dir = args.plots_dir or analysis_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    full_baseline = read_csv(analysis_dir / "full_baseline_summary.csv")
    full_by_chunk = read_csv(analysis_dir / "full_single_heavy_by_chunk.csv")
    full_by_metric = read_csv(analysis_dir / "full_single_heavy_by_metric.csv")
    full_best_summary = read_csv(analysis_dir / "full_sample_best_summary.csv")
    full_best_hist = read_csv(analysis_dir / "full_best_chunk_histogram.csv")
    oracle_alignment = read_csv(analysis_dir / "full_internal_oracle_alignment_summary.csv")
    suffix_by_chunk = read_csv(analysis_dir / "suffix_single_heavy_by_suffix_chunk.csv")
    suffix_relative = read_csv(analysis_dir / "suffix_single_heavy_by_relative_position.csv")

    plot_full_baseline_summary(full_baseline, plots_dir)
    plot_single_heavy_curves(full_by_chunk, plots_dir)
    plot_single_heavy_metric_overview(full_by_metric, full_best_summary, plots_dir)
    plot_best_chunk_histogram(full_best_hist, plots_dir)
    plot_oracle_alignment(oracle_alignment, plots_dir)
    plot_suffix_heatmaps(suffix_by_chunk, plots_dir)
    plot_suffix_relative_position(suffix_relative, plots_dir)

    print(f"[plot-paired-vbench] wrote plots to: {plots_dir}")


if __name__ == "__main__":
    main()
