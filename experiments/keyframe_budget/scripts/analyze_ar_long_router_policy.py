#!/usr/bin/env python3
"""Analyze AR-long A-router policy rollouts against existing baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TEMPORAL_METRICS = (
    "background_consistency",
    "subject_consistency",
    "motion_smoothness",
    "overall_consistency",
)
QUALITY_METRICS = ("imaging_quality", "aesthetic_quality")
ALL_METRICS = (
    "dynamic_degree",
    "motion_smoothness",
    "overall_consistency",
    "imaging_quality",
    "aesthetic_quality",
    "subject_consistency",
    "background_consistency",
)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def bootstrap_mean_ci(values: pd.Series, n_bootstrap: int = 1000, seed: int = 0) -> Tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) == 0 or n_bootstrap <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(np.mean(clean[rng.integers(0, len(clean), size=len(clean))])) for _ in range(n_bootstrap)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_policy_manifests(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("ar_teacher_long_router_p*_s*/router_policy_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("policies", []):
            rows.append(
                {
                    **row,
                    "selected_chunks_json": json.dumps(row.get("selected_chunks", [])),
                    "steps_per_chunk_json": json.dumps(row.get("steps_per_chunk", [])),
                }
            )
    if not rows:
        raise RuntimeError(f"No router policy manifests found under {root}")
    df = pd.DataFrame(rows)
    return df.drop(columns=["selected_chunks", "steps_per_chunk"], errors="ignore")


def load_video_records(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"No VBench video records in {path}")
    df["prompt_id"] = df["prompt_id"].astype(str)
    df["seed"] = df["seed"].astype(int)
    df["schedule"] = df["schedule"].astype(str)
    return df


def metric_pivot(records: pd.DataFrame, schedule_col: str = "schedule") -> pd.DataFrame:
    grouped = (
        records.groupby(["prompt_id", "seed", schedule_col, "metric"], as_index=False)["score"]
        .mean()
        .rename(columns={schedule_col: "policy_name"})
    )
    return grouped.pivot_table(
        index=["prompt_id", "seed", "policy_name"],
        columns="metric",
        values="score",
        aggfunc="mean",
    ).reset_index()


def parse_policy(policy_name: str) -> Tuple[str, int, str]:
    if policy_name == "all_fast" or policy_name == "all_heavy":
        return policy_name, -1, "baseline"
    if "_k05" in policy_name:
        k = 5
    elif "_k10" in policy_name:
        k = 10
    else:
        k = -1
    if policy_name.startswith("random"):
        family = "random"
    elif policy_name.startswith("periodic"):
        family = "periodic"
    elif policy_name.startswith("Amax") and "no_chunk0" in policy_name:
        family = "Amax_no_chunk0"
    elif policy_name.startswith("Amax"):
        family = "Amax"
    elif policy_name.startswith("single_heavy_oracle") or policy_name.startswith("oracle"):
        family = "single_heavy_oracle"
    else:
        family = policy_name
    return family, k, policy_name


def add_deltas(policy_scores: pd.DataFrame, baseline_records: pd.DataFrame) -> pd.DataFrame:
    same_batch_baselines = policy_scores[policy_scores["policy_name"].isin(["all_fast", "all_heavy"])]
    if set(same_batch_baselines["policy_name"]) == {"all_fast", "all_heavy"}:
        baselines = same_batch_baselines
        baseline_source = "same_batch"
    else:
        baselines = metric_pivot(
            baseline_records[baseline_records["schedule"].isin(["all_fast", "all_heavy"])]
        )
        baseline_source = "external"
    fast = baselines[baselines["policy_name"] == "all_fast"].drop(columns=["policy_name"])
    heavy = baselines[baselines["policy_name"] == "all_heavy"].drop(columns=["policy_name"])
    if fast.empty or heavy.empty:
        raise RuntimeError("Need all_fast and all_heavy baselines, either in router VBench or baseline VBench.")
    merged = policy_scores.merge(fast, on=["prompt_id", "seed"], suffixes=("", "_all_fast"))
    merged = merged.merge(heavy, on=["prompt_id", "seed"], suffixes=("", "_all_heavy"))
    merged["baseline_source"] = baseline_source

    for metric in ALL_METRICS:
        if metric not in merged or f"{metric}_all_fast" not in merged:
            continue
        merged[f"{metric}_delta"] = merged[metric] - merged[f"{metric}_all_fast"]
        denom = merged[f"{metric}_all_heavy"] - merged[f"{metric}_all_fast"]
        merged[f"{metric}_recovery"] = merged[f"{metric}_delta"] / denom.replace(0, np.nan)

    delta_cols = [f"{m}_delta" for m in TEMPORAL_METRICS if f"{m}_delta" in merged]
    qual_cols = [f"{m}_delta" for m in QUALITY_METRICS if f"{m}_delta" in merged]
    metric_cols = [m for m in ALL_METRICS if m in merged]
    merged["delta_temp_raw"] = merged[delta_cols].mean(axis=1)
    merged["delta_qual_raw"] = merged[qual_cols].mean(axis=1)
    merged["vbench_mean"] = merged[metric_cols].mean(axis=1)
    merged["vbench_mean_delta"] = merged[[f"{m}_delta" for m in metric_cols if f"{m}_delta" in merged]].mean(axis=1)
    return merged


def add_within_sample_z_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for metric in TEMPORAL_METRICS + QUALITY_METRICS:
        col = f"{metric}_delta"
        if col not in out:
            continue
        z_col = f"{metric}_delta_z"
        out[z_col] = out.groupby(["prompt_id", "seed"])[col].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else np.nan)
        )
    temp_z = [f"{m}_delta_z" for m in TEMPORAL_METRICS if f"{m}_delta_z" in out]
    qual_z = [f"{m}_delta_z" for m in QUALITY_METRICS if f"{m}_delta_z" in out]
    out["delta_temp"] = out[temp_z].mean(axis=1)
    out["delta_qual"] = out[qual_z].mean(axis=1)
    return out


def paired_comparisons(df: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    targets = ["delta_temp_raw", "delta_temp", "vbench_mean_delta"] + [
        f"{metric}_delta" for metric in ALL_METRICS if f"{metric}_delta" in df
    ]
    for k in sorted(x for x in df["k"].dropna().unique() if x > 0):
        subset = df[df["k"] == k]
        random_mean = (
            subset[subset["policy_family"] == "random"]
            .groupby(["prompt_id", "seed"])[targets]
            .mean()
            .reset_index()
        )
        comparators = {
            "random_mean": random_mean,
            "periodic": subset[subset["policy_family"] == "periodic"],
            "Amax_no_chunk0": subset[subset["policy_family"] == "Amax_no_chunk0"],
            "single_heavy_oracle": subset[subset["policy_family"] == "single_heavy_oracle"],
            "all_fast": df[df["policy_family"] == "all_fast"],
            "all_heavy": df[df["policy_family"] == "all_heavy"],
        }
        amax = subset[subset["policy_family"] == "Amax"]
        for comp_name, comp_df in comparators.items():
            if comp_df.empty or amax.empty:
                continue
            paired = amax[["prompt_id", "seed", *targets]].merge(
                comp_df[["prompt_id", "seed", *targets]],
                on=["prompt_id", "seed"],
                suffixes=("_Amax", f"_{comp_name}"),
            )
            for target in targets:
                diff = paired[f"{target}_Amax"] - paired[f"{target}_{comp_name}"]
                ci_low, ci_high = bootstrap_mean_ci(
                    diff,
                    n_bootstrap=n_bootstrap,
                    seed=stable_seed("router_pairwise", k, comp_name, target),
                )
                rows.append(
                    {
                        "k": int(k),
                        "comparison": f"Amax_vs_{comp_name}",
                        "target": target,
                        "n": int(diff.notna().sum()),
                        "mean_diff": float(diff.mean()),
                        "mean_diff_ci_low": ci_low,
                        "mean_diff_ci_high": ci_high,
                        "median_diff": float(diff.median()),
                        "win_rate": float((diff > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def oracle_recovery(table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    targets = ["delta_temp_raw", "delta_temp", "vbench_mean_delta"] + [
        f"{metric}_delta" for metric in ALL_METRICS if f"{metric}_delta" in table
    ]
    for k in sorted(x for x in table["k"].dropna().unique() if x > 0):
        subset = table[table["k"] == k]
        amax = subset[subset["policy_family"] == "Amax"]
        oracle = subset[subset["policy_family"] == "single_heavy_oracle"]
        if amax.empty or oracle.empty:
            continue
        paired = amax[["prompt_id", "seed", *targets]].merge(
            oracle[["prompt_id", "seed", *targets]],
            on=["prompt_id", "seed"],
            suffixes=("_Amax", "_single_heavy_oracle"),
        )
        for target in targets:
            denom = paired[f"{target}_single_heavy_oracle"].replace(0, np.nan)
            recovery = paired[f"{target}_Amax"] / denom
            rows.append(
                {
                    "k": int(k),
                    "target": target,
                    "n": int(recovery.notna().sum()),
                    "mean_recovery_to_single_heavy_oracle": float(recovery.mean()),
                    "median_recovery_to_single_heavy_oracle": float(recovery.median()),
                }
            )
    return pd.DataFrame(rows)


def plot_policy_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for target in ("delta_temp_raw", "delta_temp", "vbench_mean_delta"):
        if target not in summary:
            continue
        for k in sorted(summary["k"].dropna().unique()):
            if k < 0:
                continue
            sub = summary[summary["k"] == k].sort_values("policy_family")
            plt.figure(figsize=(9, 4))
            plt.bar(sub["policy_family"], sub[target])
            plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
            plt.xticks(rotation=35, ha="right")
            plt.ylabel(target)
            plt.title(f"Router policy {target}, k={int(k)}")
            plt.tight_layout()
            plt.savefig(out_dir / f"{target}_k{int(k):02d}.png", dpi=180)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router_vbench", type=Path, default=Path("outputs/ar_teacher_long_router_policy/vbench_analysis/vbench_video_records.csv"))
    parser.add_argument("--baseline_vbench", type=Path, default=Path("outputs/ar_teacher_long_keyframe_budget/vbench_analysis/vbench_video_records.csv"))
    parser.add_argument("--router_root", type=Path, default=Path("outputs/ar_teacher_long_router_policy"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/ar_teacher_long_router_policy/analysis"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    router_records = load_video_records(args.router_vbench)
    baseline_records = load_video_records(args.baseline_vbench)
    policy_scores = metric_pivot(router_records)
    families = policy_scores["policy_name"].map(parse_policy)
    policy_scores["policy_family"] = [x[0] for x in families]
    policy_scores["k"] = [x[1] for x in families]
    policy_scores["policy_label"] = [x[2] for x in families]

    table = add_within_sample_z_targets(add_deltas(policy_scores, baseline_records))
    manifests = read_policy_manifests(args.router_root)
    chunks = manifests[["prompt_id", "seed", "policy_name", "selected_chunks_json", "total_nfe_requested"]].copy()
    chunks["seed"] = chunks["seed"].astype(int)
    table = table.merge(chunks, on=["prompt_id", "seed", "policy_name"], how="left")

    summary = (
        table.groupby(["policy_family", "k"], as_index=False)
        .agg(
            n=("delta_temp_raw", "count"),
            delta_temp_raw=("delta_temp_raw", "mean"),
            delta_temp_z=("delta_temp", "mean"),
            delta_qual=("delta_qual", "mean"),
            vbench_mean_delta=("vbench_mean_delta", "mean"),
            vbench_mean=("vbench_mean", "mean"),
        )
        .sort_values(["k", "policy_family"])
    )
    pairwise = paired_comparisons(table, n_bootstrap=args.bootstrap)
    recovery_to_oracle = oracle_recovery(table)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "router_policy_by_sample.csv", table)
    write_csv(args.out_dir / "router_policy_summary.csv", summary)
    write_csv(args.out_dir / "router_policy_pairwise_tests.csv", pairwise)
    write_csv(args.out_dir / "router_policy_recovery_to_single_heavy_oracle.csv", recovery_to_oracle)
    write_csv(args.out_dir / "router_policy_chosen_chunks.csv", manifests)
    plot_policy_summary(summary, args.out_dir / "router_policy_plots")

    print(f"[router-analysis] rows={len(table)} wrote={args.out_dir}")


if __name__ == "__main__":
    main()
