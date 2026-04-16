#!/bin/bash
#SBATCH --job-name=kfb_disc_arr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/keyframe_budget/%x-%A_%a.stdout.log
#SBATCH --error=logs/keyframe_budget/%x-%A_%a.stderr.log
#SBATCH -C h100
#SBATCH -A hdy@h100
#SBATCH --array=0-71%24

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$WORK/Causal-Forcing}"
PROMPT_SOURCE="${ROOT_DIR}/experiments/keyframe_budget/prompts/motion_rich_24.json"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

# 24 prompts x 3 seeds = 72 array tasks.
PROMPT_INDEX=$((TASK_ID / 3))
SEED_INDEX=$((TASK_ID % 3))
TASK_SEED="${SEED_INDEX}"
PROMPT_TAG=$(printf "%03d" "${PROMPT_INDEX}")

TASK_PROMPT_FILE="${SLURM_TMPDIR:-/tmp}/kfb_discovery_prompt_${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}.json"
export PROMPT_SOURCE PROMPT_INDEX TASK_PROMPT_FILE
python3 - <<'PY'
import json
import os

src = os.environ["PROMPT_SOURCE"]
idx = int(os.environ["PROMPT_INDEX"])
dst = os.environ["TASK_PROMPT_FILE"]

with open(src, "r", encoding="utf-8") as f:
    payload = json.load(f)

if not isinstance(payload, list):
    raise TypeError(f"Expected list in prompt source, got: {type(payload).__name__}")
if idx < 0 or idx >= len(payload):
    raise IndexError(f"Prompt index {idx} out of bounds for {len(payload)} prompts.")

item = payload[idx]
if isinstance(item, str):
    prompt = {"prompt_id": f"prompt_{idx:04d}", "prompt_text": item}
elif isinstance(item, dict):
    prompt = item
else:
    raise TypeError(f"Unsupported prompt item type: {type(item).__name__}")

with open(dst, "w", encoding="utf-8") as f:
    json.dump([prompt], f, indent=2, ensure_ascii=True)
PY

export EXP_CONFIG="experiments/keyframe_budget/configs/discovery_split.yaml"
export PROMPT_SET="${TASK_PROMPT_FILE}"
export NUM_PROMPTS=1
export SEEDS="${TASK_SEED}"
export EXPERIMENT_NAME="discovery_split_p${PROMPT_TAG}_s${TASK_SEED}"

source "${ROOT_DIR}/experiments/keyframe_budget/scripts/keyframe_budget_core.slurm"
