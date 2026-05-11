#!/bin/bash
#SBATCH --job-name=kfb_arl_vb_part
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

# Partial full-video VBench for AR-teacher long rollouts.
#
# This intentionally writes to a separate RUN_TAG/output family from the final
# full-cohort run. The inherited VBench launcher skips an entire chunk once an
# eval JSON exists, so mixing partial and final evaluations in the same output
# directory would make later completion unsafe.
#
# Uses 7 metrics x 42 cohort shards, 4 A100 per task, capped at 48 GPUs.

ROOT_DIR="${ROOT_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/Causal-Forcing}"
VBENCH_DIR="${VBENCH_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/VBench}"

export ROOT_DIR
export VBENCH_DIR
export INPUT_VIDEO_DIR="${INPUT_VIDEO_DIR:-${ROOT_DIR}/outputs/ar_teacher_long_keyframe_budget}"
export RUN_TAG="${RUN_TAG:-ar_teacher_long_full_partial_intersect}"
export TMP_BASE="${TMP_BASE:-${SLURM_TMPDIR:-/tmp}/keyframe_budget_ar_teacher_long_partial_full}"
export VBENCH_OUTPUT_FAMILY="${VBENCH_OUTPUT_FAMILY:-keyframe_budget_ar_teacher_long_partial_full}"
export SHARD_NAME_REGEX="${SHARD_NAME_REGEX:-^ar_teacher_long_p[0-9][0-9][0-9]_s[0-9]+$}"
export METRIC_SHARDS="${METRIC_SHARDS:-42}"

# Critical for running before every prompt/seed sample has finished.
export FAIR_MATCH_MODE="${FAIR_MATCH_MODE:-intersect}"
export RESUME_SKIP_COMPLETED="${RESUME_SKIP_COMPLETED:-1}"

source "${ROOT_DIR}/experiments/keyframe_budget/scripts/vbench_overnight_long_rollout_full_36gpus.slurm"
