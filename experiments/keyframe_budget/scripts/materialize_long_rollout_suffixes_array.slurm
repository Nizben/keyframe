#!/bin/bash
#SBATCH --job-name=kfb_suffix_arr
#SBATCH -A hdy@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/keyframe_budget/%x_%A_%a.out
#SBATCH --error=logs/keyframe_budget/%x_%A_%a.err

set -euo pipefail

module purge
module load arch/h100
module load cuda/12.1.0
module load miniforge/24.9.0
source ~/.bashrc

CONDA_ENV_NAME="${CONDA_ENV_NAME:-vbench_env}"
conda activate "${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"

ROOT_DIR="${ROOT_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/Causal-Forcing}"
INPUT_VIDEO_DIR="${INPUT_VIDEO_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/Causal-Forcing/outputs/long_keyframe_budget}"
SUFFIX_OUTPUT_ROOT="${SUFFIX_OUTPUT_ROOT:-/lustre/fswork/projects/rech/hdy/ujc37rw/Causal-Forcing/outputs/long_keyframe_budget_suffixes}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
SUFFIX_OVERWRITE="${SUFFIX_OVERWRITE:-0}"
MANIFEST_OFFSET="${MANIFEST_OFFSET:-0}"

if [ -z "$MANIFEST_PATH" ]; then
  echo "[suffix-array] MANIFEST_PATH is required."
  exit 1
fi
if [ ! -f "$MANIFEST_PATH" ]; then
  echo "[suffix-array] Manifest not found: $MANIFEST_PATH"
  exit 1
fi

CMD=(
  python "${ROOT_DIR}/experiments/keyframe_budget/scripts/materialize_long_rollout_suffixes.py"
  --input_root "$INPUT_VIDEO_DIR"
  --output_root "$SUFFIX_OUTPUT_ROOT"
  --manifest_path "$MANIFEST_PATH"
  --task_index "$((MANIFEST_OFFSET + SLURM_ARRAY_TASK_ID))"
)
if [ "$SUFFIX_OVERWRITE" = "1" ]; then
  CMD+=(--overwrite)
fi

echo "[suffix-array] task=${SLURM_ARRAY_TASK_ID} offset=${MANIFEST_OFFSET} manifest=$MANIFEST_PATH"
"${CMD[@]}"
