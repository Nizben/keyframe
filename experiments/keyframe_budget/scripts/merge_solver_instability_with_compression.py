#!/usr/bin/env python3
"""Merge 8-step all-fast solver-instability scores into compression/VBench scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def zscore_by_sample(df: pd.DataFrame, column: str) -> pd.Series:
    def zscore(series: pd.Series) -> pd.Series:
        std = series.std(ddof=0)
        if not std:
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std

    return df.groupby(["prompt_id", "seed"])[column].transform(zscore)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compression",
        type=Path,
        default=Path("outputs/ar_teacher_long_keyframe_budget/compression_transition/compression_vbench_table.parquet"),
    )
    parser.add_argument(
        "--instability",
        type=Path,
        default=Path("outputs/ar_teacher_long_router_ablation/solver_instability_scores.parquet"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/ar_teacher_long_router_ablation/compression_vbench_with_solver_instability.parquet"),
    )
    args = parser.parse_args()

    comp = pd.read_parquet(args.compression).copy()
    instability = pd.read_parquet(args.instability)[
        ["prompt_id", "seed", "heavy_idx", "preview_steps", "S_instability"]
    ].copy()
    comp["prompt_id"] = comp["prompt_id"].astype(str)
    comp["seed"] = comp["seed"].astype(int)
    comp["heavy_idx"] = comp["heavy_idx"].astype(int)
    instability["prompt_id"] = instability["prompt_id"].astype(str)
    instability["seed"] = instability["seed"].astype(int)
    instability["heavy_idx"] = instability["heavy_idx"].astype(int)

    merged = comp.merge(instability, on=["prompt_id", "seed", "heavy_idx"], how="inner")
    expected = len(comp)
    if len(merged) != expected:
        raise RuntimeError(f"S merge changed row count: expected={expected}, got={len(merged)}")

    merged["Amax_z"] = zscore_by_sample(merged, "A_max")
    merged["S_z"] = zscore_by_sample(merged, "S_instability")
    merged["AplusS"] = merged["Amax_z"] + merged["S_z"]
    merged["AtimesS"] = merged["Amax_z"] * merged["S_z"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out, index=False)
    print(f"[merge-S-compression] rows={len(merged)} wrote={args.out}")


if __name__ == "__main__":
    main()
