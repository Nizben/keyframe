#!/bin/bash
#SBATCH --job-name=kfb_comp_score
#SBATCH -A hdy@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-72%24
#SBATCH --output=logs/keyframe_budget/%x-%A_%a.stdout.log
#SBATCH --error=logs/keyframe_budget/%x-%A_%a.stderr.log

set -euo pipefail

module purge
module load arch/h100
module load miniforge/24.9.0
source ~/.bashrc
conda activate "${CONDA_ENV_NAME:-deepforcing}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

ROOT_DIR="${ROOT_DIR:-$WORK/Causal-Forcing}"
cd "${ROOT_DIR}"
mkdir -p logs/keyframe_budget

MANIFEST_GLOB="${MANIFEST_GLOB:-outputs/ar_teacher_long_keyframe_budget/compression_transition/manifests/manifest_task_*.parquet}"
OUT="${OUT:-outputs/ar_teacher_long_keyframe_budget/compression_transition/transition_scores.parquet}"
SHARD_DIR="${SHARD_DIR:-outputs/ar_teacher_long_keyframe_budget/compression_transition/transition_score_shards}"
NUM_SCORE_TASKS="${NUM_SCORE_TASKS:-72}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

if [ "${TASK_ID}" -lt "${NUM_SCORE_TASKS}" ]; then
  python experiments/keyframe_budget/scripts/compute_compression_transition_scores.py \
    --manifest_glob "${MANIFEST_GLOB}" \
    --out "${OUT}" \
    --shard_dir "${SHARD_DIR}" \
    --task_id "${TASK_ID}" \
    --num_tasks "${NUM_SCORE_TASKS}" \
    --progress_every 1
else
  python experiments/keyframe_budget/scripts/compute_compression_transition_scores.py \
    --manifest_glob "${MANIFEST_GLOB}" \
    --out "${OUT}" \
    --shard_dir "${SHARD_DIR}" \
    --num_tasks "${NUM_SCORE_TASKS}" \
    --merge_only \
    --wait \
    --wait_timeout_sec "${WAIT_TIMEOUT_SEC:-7200}"
fi
