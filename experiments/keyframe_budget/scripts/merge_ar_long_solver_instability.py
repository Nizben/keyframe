#!/usr/bin/env python3
"""Merge per-task solver-instability shards into one parquet table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glob",
        type=str,
        default="outputs/ar_teacher_long_router_ablation/solver_instability/*/solver_instability.parquet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/ar_teacher_long_router_ablation/solver_instability_scores.parquet"),
    )
    args = parser.parse_args()

    paths = sorted(Path().glob(args.glob))
    if not paths:
        raise RuntimeError(f"No solver-instability parquet shards matched: {args.glob}")
    df = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    df = df.sort_values(["prompt_id", "seed", "heavy_idx"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[merge-solver-instability] shards={len(paths)} rows={len(df)} wrote={args.out}")


if __name__ == "__main__":
    main()
