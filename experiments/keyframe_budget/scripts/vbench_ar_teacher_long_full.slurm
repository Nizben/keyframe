#!/bin/bash
#SBATCH --job-name=kfb_arl_vb_a100
#SBATCH -A hdy@a100
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --array=0-293%12
#SBATCH --output=logs/keyframe_budget/%x_%A_%a.out
#SBATCH --error=logs/keyframe_budget/%x_%A_%a.err

set -euo pipefail

# Full-video VBench for AR-teacher long rollouts generated with ar_diffusion.pt.
# Uses 7 metrics x 42 cohort shards, 4 A100 per task, capped at 48 GPUs.

ROOT_DIR="${ROOT_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/Causal-Forcing}"
VBENCH_DIR="${VBENCH_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/VBench}"

export ROOT_DIR
export VBENCH_DIR
export INPUT_VIDEO_DIR="${INPUT_VIDEO_DIR:-${ROOT_DIR}/outputs/ar_teacher_long_keyframe_budget}"
export RUN_TAG="${RUN_TAG:-ar_teacher_long_full}"
export TMP_BASE="${TMP_BASE:-${SLURM_TMPDIR:-/tmp}/keyframe_budget_ar_teacher_long_full}"
export VBENCH_OUTPUT_FAMILY="${VBENCH_OUTPUT_FAMILY:-keyframe_budget_ar_teacher_long_full}"
export SHARD_NAME_REGEX="${SHARD_NAME_REGEX:-^ar_teacher_long_p[0-9][0-9][0-9]_s[0-9]+$}"
export METRIC_SHARDS="${METRIC_SHARDS:-42}"

source "${ROOT_DIR}/experiments/keyframe_budget/scripts/vbench_overnight_long_rollout_full_36gpus.slurm"
