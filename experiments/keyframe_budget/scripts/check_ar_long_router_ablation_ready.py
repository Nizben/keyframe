#!/usr/bin/env python3
"""Check whether the AR-long Amax/S ablation pipeline is ready to launch."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


EXPECTED_SAMPLES = 72
EXPECTED_CHUNKS = 40
EXPECTED_S_ROWS = EXPECTED_SAMPLES * EXPECTED_CHUNKS
EXPECTED_COHORTS = 32
EXPECTED_VBENCH_TASKS = 7 * EXPECTED_COHORTS
REQUIRED_SCORE_COLUMNS = {"A_max", "S_instability", "AplusS", "AtimesS"}
REQUIRED_POLICIES = {
    "all_fast",
    "all_heavy",
    "Amax_k05_h12",
    "Amax_k05_h16",
    "Amax_k05_h20",
    "Amax_k05_h24",
    "S_top_k05_h24",
    "AplusS_top_k05_h24",
    "AtimesS_top_k05_h24",
    "global_mean_oracle_k05_h24",
    "imaging_oracle_k05_h24",
    "temporal_oracle_k05_h24",
}


def check(condition: bool, message: str, failures: List[str]) -> None:
    prefix = "PASS" if condition else "FAIL"
    print(f"[{prefix}] {message}")
    if not condition:
        failures.append(message)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_s_shards(root: Path, failures: List[str]) -> None:
    shards = sorted((root / "solver_instability").glob("ar_teacher_long_router_ablation_p*_s*/solver_instability.parquet"))
    check(len(shards) == EXPECTED_SAMPLES, f"S preview shards: {len(shards)}/{EXPECTED_SAMPLES}", failures)
    if not shards:
        return
    bad = []
    for path in shards:
        try:
            if len(pd.read_parquet(path)) != EXPECTED_CHUNKS:
                bad.append(str(path))
        except Exception:
            bad.append(str(path))
    check(not bad, f"S preview shard row counts are {EXPECTED_CHUNKS}", failures)


def check_tables(root: Path, failures: List[str]) -> None:
    s_path = root / "solver_instability_scores.parquet"
    check(s_path.exists(), f"merged S table exists: {s_path}", failures)
    if s_path.exists():
        s_df = pd.read_parquet(s_path)
        check(len(s_df) == EXPECTED_S_ROWS, f"merged S rows: {len(s_df)}/{EXPECTED_S_ROWS}", failures)
        check({"prompt_id", "seed", "heavy_idx", "S_instability"}.issubset(s_df.columns), "merged S columns are present", failures)

    merged_path = root / "compression_vbench_with_solver_instability.parquet"
    check(merged_path.exists(), f"merged compression+S table exists: {merged_path}", failures)
    if merged_path.exists():
        merged = pd.read_parquet(merged_path)
        check(len(merged) == EXPECTED_S_ROWS, f"compression+S rows: {len(merged)}/{EXPECTED_S_ROWS}", failures)
        missing = sorted(REQUIRED_SCORE_COLUMNS - set(merged.columns))
        check(not missing, f"router score columns present: {sorted(REQUIRED_SCORE_COLUMNS)}", failures)


def check_manifests(root: Path, failures: List[str]) -> None:
    manifests = sorted(root.glob("ar_teacher_long_router_ablation_p*_s*/router_ablation_manifest.json"))
    check(len(manifests) in {0, EXPECTED_SAMPLES}, f"ablation manifests: {len(manifests)}/{EXPECTED_SAMPLES} or none before generation", failures)
    if not manifests:
        return

    bad_count = []
    bad_policy = []
    bad_k = []
    for path in manifests:
        policies = load_json(path).get("policies", [])
        names = {str(row.get("policy_name")) for row in policies}
        if len(policies) != EXPECTED_COHORTS:
            bad_count.append(str(path))
        if not REQUIRED_POLICIES.issubset(names):
            bad_policy.append(str(path))
        for row in policies:
            name = str(row.get("policy_name"))
            if name in {"all_fast", "all_heavy"}:
                continue
            if len(row.get("selected_chunks", [])) != 5:
                bad_k.append(f"{path}:{name}")
    check(not bad_count, f"each manifest has {EXPECTED_COHORTS} cohorts", failures)
    check(not bad_policy, "required ablation policy names are present", failures)
    check(not bad_k, "all non-baseline policies select exactly k=5 chunks", failures)


def check_vbench_slurm(path: Path, failures: List[str]) -> None:
    check(path.exists(), f"VBench slurm exists: {path}", failures)
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    array_ok = bool(re.search(r"#SBATCH\s+--array=0-223(?:%|\s|$)", text))
    shards_ok = 'METRIC_SHARDS="${METRIC_SHARDS:-32}"' in text
    check(array_ok, f"VBench array covers {EXPECTED_VBENCH_TASKS} tasks: 0-223", failures)
    check(shards_ok, "VBench METRIC_SHARDS defaults to 32", failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/ar_teacher_long_router_ablation"))
    parser.add_argument(
        "--vbench_slurm",
        type=Path,
        default=Path("experiments/keyframe_budget/scripts/vbench_ar_teacher_long_router_ablation.slurm"),
    )
    args = parser.parse_args()

    failures: List[str] = []
    check_vbench_slurm(args.vbench_slurm, failures)
    check_s_shards(args.root, failures)
    check_tables(args.root, failures)
    check_manifests(args.root, failures)

    if failures:
        print(f"[ablation-ready] NOT READY: {len(failures)} failed checks")
        raise SystemExit(1)
    print("[ablation-ready] READY")


if __name__ == "__main__":
    main()
