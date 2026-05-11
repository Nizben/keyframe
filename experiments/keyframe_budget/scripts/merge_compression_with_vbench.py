#!/usr/bin/env python3
"""Merge compression-transition scores with per-video VBench deltas."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


TEMPORAL_METRICS = [
    "background_consistency",
    "subject_consistency",
    "motion_smoothness",
    "overall_consistency",
]
QUALITY_METRICS = ["imaging_quality", "aesthetic_quality"]


def zscore_by_sample(rows: pd.DataFrame, metric: str) -> pd.Series:
    vals = rows[metric].astype(float)
    std = vals.std(ddof=0)
    if std == 0 or pd.isna(std):
        return vals * 0.0
    return (vals - vals.mean()) / std


def zscore_series(vals: pd.Series) -> pd.Series:
    vals = vals.astype(float)
    std = vals.std(ddof=0)
    if std == 0 or pd.isna(std):
        return vals * 0.0
    return (vals - vals.mean()) / std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=Path("outputs/ar_teacher_long_keyframe_budget/compression_transition/transition_scores.parquet"))
    parser.add_argument("--vbench", type=Path, default=Path("outputs/ar_teacher_long_keyframe_budget/vbench_analysis/vbench_video_records.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/ar_teacher_long_keyframe_budget/compression_transition/compression_vbench_table.parquet"))
    args = parser.parse_args()

    scores = pd.read_parquet(args.scores)
    vbench = pd.read_csv(args.vbench)
    vbench = vbench[vbench["split"] == "full"].copy()
    idx: Dict[Tuple[str, int, str, str], float] = {}
    for row in vbench.itertuples(index=False):
        idx[(str(row.prompt_id), int(row.seed), str(row.metric), str(row.schedule))] = float(row.score)

    rows: List[dict] = []
    metrics = sorted(vbench["metric"].unique())
    for srow in scores.itertuples(index=False):
        prompt_id = str(srow.prompt_id)
        seed = int(srow.seed)
        heavy_idx = int(srow.heavy_idx)
        schedule = f"single_heavy_{heavy_idx:02d}"
        row = srow._asdict()
        for metric in metrics:
            fast = idx.get((prompt_id, seed, metric, "all_fast"))
            heavy = idx.get((prompt_id, seed, metric, "all_heavy"))
            single = idx.get((prompt_id, seed, metric, schedule))
            row[f"{metric}_all_fast"] = fast
            row[f"{metric}_all_heavy"] = heavy
            row[f"{metric}_single_heavy"] = single
            row[f"{metric}_delta"] = None if fast is None or single is None else single - fast
            row[f"{metric}_all_heavy_delta"] = None if fast is None or heavy is None else heavy - fast
        rows.append(row)

    table = pd.DataFrame(rows)
    for metric in metrics:
        delta_col = f"{metric}_delta"
        z_col = f"{metric}_delta_z"
        table[z_col] = table.groupby(["prompt_id", "seed"], group_keys=False)[delta_col].transform(zscore_series)

    table["delta_temp"] = table[[f"{m}_delta_z" for m in TEMPORAL_METRICS if f"{m}_delta_z" in table]].mean(axis=1)
    table["delta_qual"] = table[[f"{m}_delta_z" for m in QUALITY_METRICS if f"{m}_delta_z" in table]].mean(axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    print(f"[compression-merge] rows={len(table)} wrote={args.out}")


if __name__ == "__main__":
    main()
