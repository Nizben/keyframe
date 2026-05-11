#!/bin/bash
#SBATCH --job-name=kfb_ar5_vb_a100
#SBATCH -A hdy@a100
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --array=0-62%9
#SBATCH --output=logs/keyframe_budget/%x_%A_%a.out
#SBATCH --error=logs/keyframe_budget/%x_%A_%a.err

set -euo pipefail

# Full-video VBench for the 5s AR-teacher chunkwise keyframe-budget run.
#
# Parallelization model inherited from the long-rollout full-video launcher:
# - 7 VBench metrics x 9 cohort shards = 63 Slurm array tasks.
# - each task gets 4 A100 GPUs.
# - array concurrency is capped at 9 tasks, so the job can use up to 36 A100 GPUs.

ROOT_DIR="${ROOT_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/Causal-Forcing}"
VBENCH_DIR="${VBENCH_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/VBench}"

export ROOT_DIR
export VBENCH_DIR
export INPUT_VIDEO_DIR="${INPUT_VIDEO_DIR:-${ROOT_DIR}/outputs/ar_teacher_keyframe_5s}"
export RUN_TAG="${RUN_TAG:-ar_teacher_5s_full}"
export TMP_BASE="${TMP_BASE:-${SLURM_TMPDIR:-/tmp}/keyframe_budget_ar_teacher_5s_full}"
export VBENCH_OUTPUT_FAMILY="${VBENCH_OUTPUT_FAMILY:-keyframe_budget_ar_teacher_5s_full}"
export SHARD_NAME_REGEX="${SHARD_NAME_REGEX:-^ar_teacher_5s_p[0-9][0-9][0-9]_s[0-9]+$}"

source "${ROOT_DIR}/experiments/keyframe_budget/scripts/vbench_overnight_long_rollout_full_36gpus.slurm"
