#!/usr/bin/env python3
"""Analyze the compression-transition hypothesis and generate plots."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PREDICTORS = ["D_in", "D_out", "D_max", "A_in", "A_out", "A_max", "R_in", "R_out", "R_max", "Q"]
TEMPORAL_METRICS = ["background_consistency", "subject_consistency", "motion_smoothness", "overall_consistency"]
QUALITY_METRICS = ["imaging_quality", "aesthetic_quality"]


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def finite_pair(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    return df[[x, y]].replace([math.inf, -math.inf], math.nan).dropna()


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    return float(x.rank(method="average").corr(y.rank(method="average"), method="pearson"))


def bootstrap_spearman(
    data: pd.DataFrame,
    x: str,
    y: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    if len(data) < 3 or n_bootstrap <= 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values: List[float] = []
    for _ in range(n_bootstrap):
        sample = data.iloc[rng.integers(0, len(data), size=len(data))]
        rho = spearman_corr(sample[x], sample[y])
        if math.isfinite(rho):
            values.append(rho)
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    p_value = min(1.0, 2.0 * min(float(np.mean(arr <= 0.0)), float(np.mean(arr >= 0.0))))
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), p_value


def auroc_binary(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        rank_sum_pos += avg_rank * sum(label for _, label in pairs[i:j])
        i = j
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auc(
    labels: Sequence[int],
    scores: Sequence[float],
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    if len(labels) == 0 or len(labels) != len(scores) or n_bootstrap <= 0:
        return float("nan"), float("nan")
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    values: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(labels_arr), size=len(labels_arr))
        auc = auroc_binary(labels_arr[idx].tolist(), scores_arr[idx].tolist())
        if math.isfinite(auc):
            values.append(auc)
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def mean_topk_overlap(df: pd.DataFrame, predictor: str, target: str, k: int) -> float:
    overlaps = topk_overlaps_by_sample(df, predictor, target, k)
    return float(sum(overlaps) / len(overlaps)) if overlaps else float("nan")


def topk_overlaps_by_sample(df: pd.DataFrame, predictor: str, target: str, k: int) -> List[float]:
    overlaps: List[float] = []
    for _, group in df.groupby(["prompt_id", "seed"], sort=True):
        g = group[[predictor, target, "heavy_idx"]].replace([math.inf, -math.inf], math.nan).dropna()
        if len(g) < k:
            continue
        top_pred = set(g.nlargest(k, predictor)["heavy_idx"])
        top_target = set(g.nlargest(k, target)["heavy_idx"])
        overlaps.append(len(top_pred & top_target) / k)
    return overlaps


def bootstrap_mean(values: Sequence[float], n_bootstrap: int, seed: int) -> tuple[float, float]:
    clean = np.asarray([x for x in values if math.isfinite(float(x))], dtype=float)
    if len(clean) == 0 or n_bootstrap <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(np.mean(clean[rng.integers(0, len(clean), size=len(clean))])) for _ in range(n_bootstrap)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def within_group_z(df: pd.DataFrame, value_col: str, group_cols: Sequence[str]) -> pd.Series:
    def zscore(series: pd.Series) -> pd.Series:
        std = series.std(ddof=0)
        if not std or not math.isfinite(float(std)):
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std

    return df.groupby(list(group_cols))[value_col].transform(zscore)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=Path("outputs/ar_teacher_long_keyframe_budget/compression_transition/compression_vbench_table.parquet"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/ar_teacher_long_keyframe_budget/compression_transition/analysis"))
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.table)
    metric_targets = ["delta_temp", "delta_qual"] + [f"{m}_delta" for m in TEMPORAL_METRICS + QUALITY_METRICS]

    corr_rows = []
    for target in metric_targets:
        if target not in df:
            continue
        for predictor in PREDICTORS:
            data = finite_pair(df, predictor, target)
            if len(data) < 3:
                rho = ci_low = ci_high = p = float("nan")
            else:
                rho = spearman_corr(data[predictor], data[target])
                ci_low, ci_high, p = bootstrap_spearman(
                    data,
                    predictor,
                    target,
                    n_bootstrap=args.bootstrap,
                    seed=stable_seed("spearman", target, predictor),
                )
            corr_rows.append(
                {
                    "target": target,
                    "predictor": predictor,
                    "n": len(data),
                    "spearman": rho,
                    "spearman_ci_low": ci_low,
                    "spearman_ci_high": ci_high,
                    "p_value": p,
                }
            )
    write_csv(args.out_dir / "spearman_correlations.csv", corr_rows)

    topk_rows = []
    for target in ["delta_temp", "delta_qual"]:
        for predictor in PREDICTORS:
            for k in (4, 5, 10):
                overlaps = topk_overlaps_by_sample(df, predictor, target, k)
                ci_low, ci_high = bootstrap_mean(
                    overlaps,
                    n_bootstrap=args.bootstrap,
                    seed=stable_seed("topk", target, predictor, k),
                )
                topk_rows.append(
                    {
                        "target": target,
                        "predictor": predictor,
                        "k": k,
                        "n_samples": len(overlaps),
                        "mean_overlap": mean_topk_overlap(df, predictor, target, k),
                        "mean_overlap_ci_low": ci_low,
                        "mean_overlap_ci_high": ci_high,
                    }
                )
    write_csv(args.out_dir / "topk_overlap.csv", topk_rows)

    bad_rows = []
    bad_df = df[["prompt_id", "seed", "D_max", "Q", "delta_temp"]].replace([math.inf, -math.inf], math.nan).dropna().copy()
    if not bad_df.empty and len(set(bad_df["delta_temp"] < 0)) == 2:
        bad_df["lowD_highQ"] = within_group_z(bad_df, "Q", ["prompt_id", "seed"]) - within_group_z(
            bad_df, "D_max", ["prompt_id", "seed"]
        )
        labels = list((bad_df["delta_temp"] < 0).astype(int))
        auc = auroc_binary(labels, list(bad_df["lowD_highQ"]))
        auc_low, auc_high = bootstrap_auc(labels, list(bad_df["lowD_highQ"]), args.bootstrap, seed=17)
    else:
        auc = auc_low = auc_high = float("nan")
    bad_rows.append(
        {
            "label": "delta_temp_negative",
            "predictor": "within_sample_z(Q)-within_sample_z(D_max)",
            "n": len(bad_df),
            "auroc": auc,
            "auroc_ci_low": auc_low,
            "auroc_ci_high": auc_high,
        }
    )
    write_csv(args.out_dir / "auroc_bad_chunks.csv", bad_rows)

    energy_rows = []
    for (prompt_id, seed), group in df.groupby(["prompt_id", "seed"], sort=True):
        first = group.iloc[0]
        row = {"prompt_id": prompt_id, "seed": int(seed), "C_F": first["C_F"], "C_H": first["C_H"], "C_H_minus_C_F": first["C_H_minus_C_F"]}
        for metric in TEMPORAL_METRICS + QUALITY_METRICS:
            col = f"{metric}_all_heavy_delta"
            if col in first:
                row[col] = first[col]
        energy_rows.append(row)
    write_csv(args.out_dir / "allfast_vs_allheavy_delta_energy.csv", energy_rows)

    plt.figure(figsize=(6, 4))
    plt.scatter(df["R_max"], df["delta_temp"], s=8, alpha=0.45)
    plt.xlabel("R_max compression residual")
    plt.ylabel("Temporal VBench gain (mean z)")
    plt.tight_layout()
    plt.savefig(args.out_dir / "scatter_Rmax_vs_delta_temp.png", dpi=180)
    plt.close()

    heat_rows = pd.DataFrame(corr_rows)
    pivot = heat_rows.pivot(index="predictor", columns="target", values="spearman")
    plt.figure(figsize=(10, 5))
    plt.imshow(pivot.fillna(0.0), aspect="auto", vmin=-0.5, vmax=0.5, cmap="coolwarm")
    plt.colorbar(label="Spearman rho")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.tight_layout()
    plt.savefig(args.out_dir / "heatmap_transition_scores_vs_vbench.png", dpi=180)
    plt.close()

    energy = pd.DataFrame(energy_rows)
    temporal_heavy_cols = [f"{m}_all_heavy_delta" for m in TEMPORAL_METRICS if f"{m}_all_heavy_delta" in energy]
    if temporal_heavy_cols:
        energy["temporal_all_heavy_delta_mean"] = energy[temporal_heavy_cols].mean(axis=1)
        plt.figure(figsize=(6, 4))
        plt.scatter(energy["C_H_minus_C_F"], energy["temporal_all_heavy_delta_mean"], s=18, alpha=0.7)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xlabel("C_H - C_F")
        plt.ylabel("Mean temporal all-heavy VBench delta")
        plt.tight_layout()
        plt.savefig(args.out_dir / "allfast_vs_allheavy_delta_energy.png", dpi=180)
        plt.close()

    print(f"[compression-analysis] wrote={args.out_dir}")


if __name__ == "__main__":
    main()
