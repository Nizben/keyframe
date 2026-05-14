#!/usr/bin/env python3
"""Check whether the AR-long router tradeoff batch is ready for VBench."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence


EXPECTED_SAMPLES = 72
EXPECTED_COHORTS = 69
EXPECTED_VBENCH_TASKS = 7 * EXPECTED_COHORTS
FAST_STEPS = 8
REQUIRED_POLICIES = {
    "all_fast",
    "all_heavy_h24",
    "Amax_k05_h12",
    "Amax_k05_h16",
    "AplusS_k05_h12",
    "AplusS_k05_h16",
    "AplusS_k05_h24",
    "temporal_oracle_k05_h16",
    "imaging_oracle_k05_h12",
}


def check(condition: bool, message: str, failures: List[str]) -> None:
    prefix = "PASS" if condition else "FAIL"
    print(f"[{prefix}] {message}")
    if not condition:
        failures.append(message)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_from_steps(steps: Sequence[int]) -> List[int]:
    return [idx for idx, step in enumerate(steps) if int(step) != FAST_STEPS]


def check_manifests(root: Path, failures: List[str]) -> None:
    manifests = sorted(root.glob("ar_teacher_long_router_tradeoff_p*_s*/router_tradeoff_manifest.json"))
    check(len(manifests) in {0, EXPECTED_SAMPLES}, f"tradeoff manifests: {len(manifests)}/{EXPECTED_SAMPLES} or none before generation", failures)
    if not manifests:
        return

    bad_count = []
    bad_policy = []
    bad_k = []
    bad_shared = []
    for path in manifests:
        policies = load_json(path).get("policies", [])
        by_name = {str(row.get("policy_name")): row for row in policies}
        names = set(by_name)
        if len(policies) != EXPECTED_COHORTS:
            bad_count.append(str(path))
        if not REQUIRED_POLICIES.issubset(names):
            bad_policy.append(str(path))
        for heavy_steps in (12, 16, 24):
            expected = {f"random_k05_h{heavy_steps:02d}_r{rep:02d}" for rep in range(20)}
            if not expected.issubset(names):
                bad_policy.append(f"{path}:random_h{heavy_steps:02d}")
        for name, row in by_name.items():
            if name == "all_fast":
                if any(int(x) != FAST_STEPS for x in row.get("steps_per_chunk", [])):
                    bad_k.append(f"{path}:{name}")
                continue
            if name == "all_heavy_h24":
                if any(int(x) != 24 for x in row.get("steps_per_chunk", [])):
                    bad_k.append(f"{path}:{name}")
                continue
            if len(row.get("selected_chunks", [])) != 5:
                bad_k.append(f"{path}:{name}")

        try:
            if by_name["Amax_k05_h12"]["selected_chunks"] != by_name["Amax_k05_h16"]["selected_chunks"]:
                bad_shared.append(f"{path}:Amax")
            aplus = [
                by_name["AplusS_k05_h12"]["selected_chunks"],
                by_name["AplusS_k05_h16"]["selected_chunks"],
                by_name["AplusS_k05_h24"]["selected_chunks"],
            ]
            if not (aplus[0] == aplus[1] == aplus[2]):
                bad_shared.append(f"{path}:AplusS")
        except KeyError as exc:
            bad_policy.append(f"{path}:{exc}")

    check(not bad_count, f"each manifest has {EXPECTED_COHORTS} cohorts", failures)
    check(not bad_policy, "required tradeoff policy names are present", failures)
    check(not bad_k, "all non-baseline policies select exactly k=5 chunks", failures)
    check(not bad_shared, "matched-step router policies share selected chunks", failures)


def check_vbench_slurm(path: Path, failures: List[str]) -> None:
    check(path.exists(), f"VBench slurm exists: {path}", failures)
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    array_ok = bool(re.search(r"#SBATCH\s+--array=0-482(?:%|\s|$)", text))
    shards_ok = 'METRIC_SHARDS="${METRIC_SHARDS:-69}"' in text
    check(array_ok, f"VBench array covers {EXPECTED_VBENCH_TASKS} tasks: 0-482", failures)
    check(shards_ok, "VBench METRIC_SHARDS defaults to 69", failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/ar_teacher_long_router_tradeoff"))
    parser.add_argument(
        "--vbench_slurm",
        type=Path,
        default=Path("experiments/keyframe_budget/scripts/vbench_ar_teacher_long_router_tradeoff.slurm"),
    )
    args = parser.parse_args()

    failures: List[str] = []
    check_vbench_slurm(args.vbench_slurm, failures)
    check_manifests(args.root, failures)
    if failures:
        print(f"[tradeoff-ready] NOT READY: {len(failures)} failed checks")
        raise SystemExit(1)
    print("[tradeoff-ready] READY")


if __name__ == "__main__":
    main()
