#!/bin/bash
#SBATCH --job-name=kfb_vbench_long_smoke
#SBATCH -A hdy@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/keyframe_budget/%x_%A_%a.out
#SBATCH --error=logs/keyframe_budget/%x_%A_%a.err

set -euo pipefail

# This smoke preflight intentionally reuses the overnight script logic
# and only caps sample/cohort counts so we exercise the same codepaths.
ROOT_DIR="${ROOT_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/Causal-Forcing}"

export RUN_TAG="${RUN_TAG:-long_rollout_eval_smoke}"
export FAIR_MATCH_MODE="${FAIR_MATCH_MODE:-strict}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-4}"
export STAGGER_SECONDS="${STAGGER_SECONDS:-20}"

# Smoke caps:
# - keep all 7 metrics and full cohort/parsing/fairness logic
# - cap common samples/cohorts for quick preflight runtime
export SMOKE_MAX_COMMON_SAMPLES="${SMOKE_MAX_COMMON_SAMPLES:-3}"
export SMOKE_MAX_COHORTS="${SMOKE_MAX_COHORTS:-8}"

bash "${ROOT_DIR}/experiments/keyframe_budget/scripts/vbench_overnight_long_rollout_32gpus.slurm"
