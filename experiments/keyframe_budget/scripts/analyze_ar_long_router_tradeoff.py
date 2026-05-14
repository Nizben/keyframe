#!/usr/bin/env python3
"""Analyze AR-long router tradeoff confirmation VBench results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_manifests(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("ar_teacher_long_router_tradeoff_p*_s*/router_tradeoff_manifest.json")):
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
        raise RuntimeError(f"No tradeoff manifests found under {root}")
    return pd.DataFrame(rows).drop(columns=["selected_chunks", "steps_per_chunk"], errors="ignore")


def load_video_records(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"No VBench video records in {path}")
    df["prompt_id"] = df["prompt_id"].astype(str)
    df["seed"] = df["seed"].astype(int)
    df["schedule"] = df["schedule"].astype(str)
    return df


def metric_pivot(records: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        records.groupby(["prompt_id", "seed", "schedule", "metric"], as_index=False)["score"]
        .mean()
        .rename(columns={"schedule": "policy_name"})
    )
    return grouped.pivot_table(
        index=["prompt_id", "seed", "policy_name"],
        columns="metric",
        values="score",
        aggfunc="mean",
    ).reset_index()


def parse_policy(policy_name: str) -> Tuple[str, int]:
    if policy_name == "all_fast":
        return "all_fast", 8
    if policy_name == "all_heavy_h24":
        return "all_heavy", 24
    if policy_name.startswith("random_k05_h"):
        return "random", int(policy_name.split("_h", 1)[1].split("_", 1)[0])
    if policy_name.startswith("Amax_k05_h"):
        return "Amax", int(policy_name.rsplit("_h", 1)[1])
    if policy_name.startswith("AplusS_k05_h"):
        return "AplusS", int(policy_name.rsplit("_h", 1)[1])
    if policy_name.startswith("temporal_oracle_k05_h"):
        return "temporal_oracle", int(policy_name.rsplit("_h", 1)[1])
    if policy_name.startswith("imaging_oracle_k05_h"):
        return "imaging_oracle", int(policy_name.rsplit("_h", 1)[1])
    return policy_name, -1


def add_deltas(scores: pd.DataFrame) -> pd.DataFrame:
    fast = scores[scores["policy_name"] == "all_fast"].drop(columns=["policy_name"])
    heavy = scores[scores["policy_name"] == "all_heavy_h24"].drop(columns=["policy_name"])
    if fast.empty or heavy.empty:
        raise RuntimeError("Tradeoff analysis requires same-batch all_fast and all_heavy_h24.")
    merged = scores.merge(fast, on=["prompt_id", "seed"], suffixes=("", "_all_fast"))
    merged = merged.merge(heavy, on=["prompt_id", "seed"], suffixes=("", "_all_heavy"))
    for metric in ALL_METRICS:
        if metric in merged and f"{metric}_all_fast" in merged:
            merged[f"{metric}_delta"] = merged[metric] - merged[f"{metric}_all_fast"]
            denom = merged[f"{metric}_all_heavy"] - merged[f"{metric}_all_fast"]
            merged[f"{metric}_recovery"] = merged[f"{metric}_delta"] / denom.replace(0, np.nan)
    temp_cols = [f"{m}_delta" for m in TEMPORAL_METRICS if f"{m}_delta" in merged]
    qual_cols = [f"{m}_delta" for m in QUALITY_METRICS if f"{m}_delta" in merged]
    metric_cols = [m for m in ALL_METRICS if m in merged]
    merged["delta_temp_raw"] = merged[temp_cols].mean(axis=1)
    merged["delta_qual_raw"] = merged[qual_cols].mean(axis=1)
    merged["vbench_mean"] = merged[metric_cols].mean(axis=1)
    merged["vbench_mean_delta"] = merged[[f"{m}_delta" for m in metric_cols if f"{m}_delta" in merged]].mean(axis=1)
    return merged


def paired_against(table: pd.DataFrame, target_policy: str, comp: pd.DataFrame, comp_name: str) -> pd.DataFrame:
    targets = ["delta_temp_raw", "delta_qual_raw", "vbench_mean_delta"] + [
        f"{metric}_delta" for metric in ALL_METRICS if f"{metric}_delta" in table
    ]
    target = table[table["policy_name"] == target_policy]
    paired = target[["prompt_id", "seed", *targets]].merge(
        comp[["prompt_id", "seed", *targets]],
        on=["prompt_id", "seed"],
        suffixes=("_target", "_comp"),
    )
    rows = []
    for metric in targets:
        diff = paired[f"{metric}_target"] - paired[f"{metric}_comp"]
        rows.append(
            {
                "target_policy": target_policy,
                "comparison": comp_name,
                "metric": metric,
                "n": int(diff.notna().sum()),
                "mean_diff": float(diff.mean()),
                "median_diff": float(diff.median()),
                "win_rate": float((diff > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def random_mean_for_step(table: pd.DataFrame, heavy_steps: int) -> pd.DataFrame:
    return table[(table["policy_family"] == "random") & (table["heavy_steps"] == heavy_steps)].groupby(
        ["prompt_id", "seed"], as_index=False
    ).mean(numeric_only=True)


def build_pairwise(table: pd.DataFrame) -> pd.DataFrame:
    comparisons = []
    matched_random_targets = [
        ("Amax_k05_h12", 12),
        ("Amax_k05_h16", 16),
        ("AplusS_k05_h12", 12),
        ("AplusS_k05_h16", 16),
        ("AplusS_k05_h24", 24),
        ("temporal_oracle_k05_h16", 16),
        ("imaging_oracle_k05_h12", 12),
    ]
    for target, heavy_steps in matched_random_targets:
        comparisons.append(paired_against(table, target, random_mean_for_step(table, heavy_steps), f"random_h{heavy_steps:02d}_mean"))

    direct = [
        ("AplusS_k05_h12", "Amax_k05_h12"),
        ("AplusS_k05_h16", "Amax_k05_h16"),
        ("AplusS_k05_h24", "Amax_k05_h16"),
        ("AplusS_k05_h24", "Amax_k05_h12"),
        ("temporal_oracle_k05_h16", "AplusS_k05_h16"),
        ("imaging_oracle_k05_h12", "Amax_k05_h12"),
    ]
    for target, comp_name in direct:
        comparisons.append(paired_against(table, target, table[table["policy_name"] == comp_name], comp_name))

    return pd.concat(comparisons, ignore_index=True).drop_duplicates(
        ["target_policy", "comparison", "metric"],
        keep="first",
    )


def plot_summaries(summary: pd.DataFrame, pairwise: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = summary[summary["policy_family"].isin(["random", "Amax", "AplusS", "temporal_oracle", "imaging_oracle", "all_fast", "all_heavy"])].copy()
    sub["label"] = sub["policy_family"] + "_h" + sub["heavy_steps"].astype(str)
    sub.loc[sub["policy_family"].isin(["all_fast", "all_heavy"]), "label"] = sub["policy_family"]
    order = [
        "all_fast",
        "random_h12",
        "Amax_h12",
        "AplusS_h12",
        "imaging_oracle_h12",
        "random_h16",
        "Amax_h16",
        "AplusS_h16",
        "temporal_oracle_h16",
        "random_h24",
        "AplusS_h24",
        "all_heavy",
    ]
    sub["order"] = sub["label"].map({name: i for i, name in enumerate(order)})
    sub = sub.dropna(subset=["order"]).sort_values("order")
    for metric in ("delta_temp_raw", "vbench_mean_delta"):
        plt.figure(figsize=(12, 4))
        plt.bar(sub["label"], sub[metric])
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(out_dir / f"policy_{metric}.png", dpi=180)
        plt.close()

    plt.figure(figsize=(6, 5))
    for _, row in sub.iterrows():
        plt.scatter(row["vbench_mean_delta"], row["delta_temp_raw"])
        plt.text(row["vbench_mean_delta"], row["delta_temp_raw"], row["label"], fontsize=8)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("vbench_mean_delta")
    plt.ylabel("delta_temp_raw")
    plt.tight_layout()
    plt.savefig(out_dir / "temporal_vs_global_tradeoff.png", dpi=180)
    plt.close()

    wins = pairwise[
        (pairwise["comparison"].str.startswith("random_h")) & (pairwise["metric"].isin(["delta_temp_raw", "vbench_mean_delta"]))
    ].copy()
    wins["label"] = wins["target_policy"] + " / " + wins["metric"]
    plt.figure(figsize=(12, 4))
    plt.bar(wins["label"], wins["win_rate"])
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("paired win rate vs matched random")
    plt.tight_layout()
    plt.savefig(out_dir / "matched_random_winrates.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vbench",
        type=Path,
        default=Path("outputs/ar_teacher_long_router_tradeoff/vbench_analysis/vbench_video_records.csv"),
    )
    parser.add_argument("--root", type=Path, default=Path("outputs/ar_teacher_long_router_tradeoff"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/ar_teacher_long_router_tradeoff/analysis"))
    args = parser.parse_args()

    records = load_video_records(args.vbench)
    scores = metric_pivot(records)
    parsed = scores["policy_name"].map(parse_policy)
    scores["policy_family"] = [x[0] for x in parsed]
    scores["heavy_steps"] = [x[1] for x in parsed]
    table = add_deltas(scores)

    manifests = read_manifests(args.root)
    manifests["seed"] = manifests["seed"].astype(int)
    chunks = manifests[["prompt_id", "seed", "policy_name", "selected_chunks_json", "total_nfe_requested"]].copy()
    table = table.merge(chunks, on=["prompt_id", "seed", "policy_name"], how="left")

    summary = (
        table.groupby(["policy_family", "heavy_steps"], as_index=False)
        .agg(
            n=("delta_temp_raw", "count"),
            delta_temp_raw=("delta_temp_raw", "mean"),
            delta_qual_raw=("delta_qual_raw", "mean"),
            vbench_mean_delta=("vbench_mean_delta", "mean"),
            vbench_mean=("vbench_mean", "mean"),
            total_nfe_requested=("total_nfe_requested", "mean"),
        )
        .sort_values(["heavy_steps", "policy_family"])
    )
    pairwise = build_pairwise(table)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "tradeoff_by_sample.csv", table)
    write_csv(args.out_dir / "tradeoff_summary.csv", summary)
    write_csv(args.out_dir / "tradeoff_pairwise.csv", pairwise)
    plot_summaries(summary, pairwise, args.out_dir / "plots")
    print(f"[router-tradeoff-analysis] rows={len(table)} wrote={args.out_dir}")


if __name__ == "__main__":
    main()
