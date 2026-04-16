#!/bin/bash
#SBATCH --job-name=kfb_policy100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/keyframe_budget/%x-%j.stdout.log
#SBATCH --error=logs/keyframe_budget/%x-%j.stderr.log
#SBATCH -C h100
#SBATCH -A hdy@h100

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$WORK/Causal-Forcing}"
export EXP_CONFIG="experiments/keyframe_budget/configs/policy_eval_100.yaml"
source "${ROOT_DIR}/experiments/keyframe_budget/scripts/keyframe_budget_core.slurm"
