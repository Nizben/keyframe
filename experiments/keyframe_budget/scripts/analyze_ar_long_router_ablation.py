#!/usr/bin/env python3
"""Analyze AR-long Amax/S router ablation VBench results."""

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
    for path in sorted(root.glob("ar_teacher_long_router_ablation_p*_s*/router_ablation_manifest.json")):
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
        raise RuntimeError(f"No ablation manifests found under {root}")
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
    if policy_name in {"all_fast", "all_heavy"}:
        return policy_name, 24 if policy_name == "all_heavy" else 8
    if policy_name.startswith("random"):
        return "random", 24
    if policy_name.startswith("Amax_k05_h"):
        return "Amax", int(policy_name.rsplit("_h", 1)[1])
    if policy_name.startswith("S_instability"):
        return "S_instability", 24
    if policy_name.startswith("AplusS"):
        return "AplusS", 24
    if policy_name.startswith("AmulS"):
        return "AmulS", 24
    if policy_name.startswith("global_mean_oracle"):
        return "global_mean_oracle", 24
    if policy_name.startswith("imaging_oracle"):
        return "imaging_oracle", 24
    if policy_name.startswith("temporal_oracle"):
        return "temporal_oracle", 24
    return policy_name, -1


def add_deltas(scores: pd.DataFrame) -> pd.DataFrame:
    fast = scores[scores["policy_name"] == "all_fast"].drop(columns=["policy_name"])
    heavy = scores[scores["policy_name"] == "all_heavy"].drop(columns=["policy_name"])
    if fast.empty or heavy.empty:
        raise RuntimeError("Ablation analysis requires same-batch all_fast and all_heavy.")
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


def build_pairwise(table: pd.DataFrame) -> pd.DataFrame:
    random_mean = table[table["policy_family"] == "random"].groupby(["prompt_id", "seed"], as_index=False).mean(
        numeric_only=True
    )
    comparisons = []
    for policy in ("Amax_k05_h24", "S_instability_k05_h24", "AplusS_k05_h24", "AmulS_k05_h24"):
        comparisons.append(paired_against(table, policy, random_mean, "random_mean"))
        if policy != "Amax_k05_h24":
            comparisons.append(paired_against(table, policy, table[table["policy_name"] == "Amax_k05_h24"], "Amax_k05_h24"))
    for policy in ("Amax_k05_h12", "Amax_k05_h16", "Amax_k05_h20", "Amax_k05_h24"):
        comparisons.append(paired_against(table, policy, random_mean, "random_mean"))
    for policy in ("global_mean_oracle_k05_h24", "imaging_oracle_k05_h24", "temporal_oracle_k05_h24"):
        comparisons.append(paired_against(table, "Amax_k05_h24", table[table["policy_name"] == policy], policy))
    return pd.concat(comparisons, ignore_index=True)


def visual_candidates(table: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    random_mean = table[table["policy_family"] == "random"].groupby(["prompt_id", "seed"], as_index=False).mean(
        numeric_only=True
    )
    base = table[table["policy_name"] == "Amax_k05_h24"][
        ["prompt_id", "seed", "delta_temp_raw", "vbench_mean_delta"]
    ].merge(random_mean[["prompt_id", "seed", "delta_temp_raw", "vbench_mean_delta"]], on=["prompt_id", "seed"], suffixes=("_Amax", "_random"))
    temporal_oracle = table[table["policy_name"] == "temporal_oracle_k05_h24"][
        ["prompt_id", "seed", "delta_temp_raw", "vbench_mean_delta"]
    ]
    base = base.merge(temporal_oracle, on=["prompt_id", "seed"], suffixes=("", "_temporal_oracle"))
    base["Amax_minus_random_temp"] = base["delta_temp_raw_Amax"] - base["delta_temp_raw_random"]
    base["Amax_minus_random_mean"] = base["vbench_mean_delta_Amax"] - base["vbench_mean_delta_random"]
    base["Amax_minus_oracle_temp"] = base["delta_temp_raw_Amax"] - base["delta_temp_raw"]

    groups = [
        ("Amax_gt_random", base.nlargest(4, "Amax_minus_random_temp")),
        ("Amax_lt_random", base.nsmallest(3, "Amax_minus_random_temp")),
        ("Amax_gt_oracle", base.nlargest(3, "Amax_minus_oracle_temp")),
    ]
    rows = []
    for label, subset in groups:
        for _, row in subset.iterrows():
            rows.append({"audit_group": label, **row.to_dict()})
    candidates = pd.DataFrame(rows).drop_duplicates(["audit_group", "prompt_id", "seed"])
    write_csv(out_dir / "visual_audit_candidates.csv", candidates)
    return candidates


def plot_summaries(summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("delta_temp_raw", "vbench_mean_delta"):
        policy_order = [
            "random",
            "Amax",
            "S_instability",
            "AplusS",
            "AmulS",
            "global_mean_oracle",
            "imaging_oracle",
            "temporal_oracle",
        ]
        sub = summary[(summary["heavy_steps"] == 24) & (summary["policy_family"].isin(policy_order))].copy()
        sub["order"] = sub["policy_family"].map({name: i for i, name in enumerate(policy_order)})
        sub = sub.sort_values("order")
        plt.figure(figsize=(10, 4))
        plt.bar(sub["policy_family"], sub[metric])
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(out_dir / f"policy_{metric}.png", dpi=180)
        plt.close()

    step = summary[(summary["policy_family"] == "Amax") & (summary["heavy_steps"].isin([12, 16, 20, 24]))].sort_values("heavy_steps")
    plt.figure(figsize=(6, 4))
    plt.plot(step["heavy_steps"], step["delta_temp_raw"], marker="o", label="delta_temp_raw")
    plt.plot(step["heavy_steps"], step["vbench_mean_delta"], marker="o", label="vbench_mean_delta")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Amax heavy steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "Amax_step_size_ablation.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vbench",
        type=Path,
        default=Path("outputs/ar_teacher_long_router_ablation/vbench_analysis/vbench_video_records.csv"),
    )
    parser.add_argument("--root", type=Path, default=Path("outputs/ar_teacher_long_router_ablation"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/ar_teacher_long_router_ablation/analysis"))
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
        )
        .sort_values(["heavy_steps", "policy_family"])
    )
    pairwise = build_pairwise(table)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "ablation_by_sample.csv", table)
    write_csv(args.out_dir / "ablation_summary.csv", summary)
    write_csv(args.out_dir / "ablation_pairwise.csv", pairwise)
    write_csv(args.out_dir / "ablation_chosen_chunks.csv", manifests)
    visual_candidates(table, args.out_dir)
    plot_summaries(summary, args.out_dir / "plots")
    print(f"[router-ablation-analysis] rows={len(table)} wrote={args.out_dir}")


if __name__ == "__main__":
    main()
