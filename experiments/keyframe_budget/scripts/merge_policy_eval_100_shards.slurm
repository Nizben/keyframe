#!/bin/bash
#SBATCH --job-name=kfb_merge_pol100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=logs/keyframe_budget/%x-%j.stdout.log
#SBATCH --error=logs/keyframe_budget/%x-%j.stderr.log
#SBATCH -C h100
#SBATCH -A hdy@h100

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$WORK/Causal-Forcing}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/keyframe_budget}"

export ROOT_DIR OUTPUT_ROOT
python3 - <<'PY'
import glob
import os
import shutil
from pathlib import Path

root_dir = Path(os.environ["ROOT_DIR"]).resolve()
output_root = Path(os.environ["OUTPUT_ROOT"])
if not output_root.is_absolute():
    output_root = (root_dir / output_root).resolve()

source_pattern = str(output_root / "policy_eval_100_p*_s*" / "aggregates" / "prompt_seed_aggregates" / "*.json")
source_files = sorted(glob.glob(source_pattern))
if not source_files:
    raise FileNotFoundError(f"No policy shard aggregate files found: {source_pattern}")

dest_aggregate_root = output_root / "policy_eval_100" / "aggregates"
dest_prompt_dir = dest_aggregate_root / "prompt_seed_aggregates"
dest_prompt_dir.mkdir(parents=True, exist_ok=True)

for existing in dest_prompt_dir.glob("*.json"):
    existing.unlink()

for src in source_files:
    src_path = Path(src)
    shutil.copy2(src_path, dest_prompt_dir / src_path.name)

for jsonl_name in ("gain_maps.jsonl", "policy_results.jsonl"):
    dest_jsonl = dest_aggregate_root / jsonl_name
    dest_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with dest_jsonl.open("w", encoding="utf-8") as out_f:
        shard_pattern = str(output_root / "policy_eval_100_p*_s*" / "aggregates" / jsonl_name)
        for shard_path in sorted(glob.glob(shard_pattern)):
            with open(shard_path, "r", encoding="utf-8") as in_f:
                out_f.write(in_f.read())

print(f"[merge-policy100] copied prompt_seed aggregates: {len(source_files)}")
print(f"[merge-policy100] destination: {dest_prompt_dir}")
PY
