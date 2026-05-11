#!/bin/bash
#SBATCH --job-name=kfb_cheap_proxy
#SBATCH -A hdy@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-72%24
#SBATCH --output=logs/keyframe_budget/%x_%A_%a.out
#SBATCH --error=logs/keyframe_budget/%x_%A_%a.err

set -euo pipefail

# Offline cheap-metric proxy analysis for long-rollout discovery.
#
# Tasks 0-71 compute one prompt/seed shard each.
# Task 72 waits for all shards and writes merged correlations.
# No video generation is performed.

module purge
module load arch/h100
module load cuda/12.1.0
module load miniforge/24.9.0
source ~/.bashrc

CONDA_ENV_NAME="${CONDA_ENV_NAME:-vbench_env}"
conda activate "${CONDA_ENV_NAME}"

ROOT_DIR="${ROOT_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/Causal-Forcing}"
cd "${ROOT_DIR}"

INPUT_ROOT="${INPUT_ROOT:-outputs/long_keyframe_budget}"
AGGREGATE_ROOT="${AGGREGATE_ROOT:-outputs/long_keyframe_budget/long_rollout_discovery/aggregates}"
RUN_TAG="${RUN_TAG:-suffix_boundary_accel_rank_combo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/long_keyframe_budget/cheap_chunk_metrics/${RUN_TAG}}"
FLOW_MAX_SIDE="${FLOW_MAX_SIDE:-256}"
TOPK="${TOPK:-10}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
mkdir -p logs/keyframe_budget "${OUTPUT_ROOT}/shards"

echo "=========================================================="
echo " Cheap proxy long-rollout analysis"
echo " Task id:        ${TASK_ID}"
echo " Input root:     ${INPUT_ROOT}"
echo " Aggregate root: ${AGGREGATE_ROOT}"
echo " Output root:    ${OUTPUT_ROOT}"
echo " Flow max side:  ${FLOW_MAX_SIDE}"
echo " Top-k:          ${TOPK}"
echo "=========================================================="

if [ "${TASK_ID}" -lt 72 ]; then
  python experiments/keyframe_budget/scripts/cheap_chunk_metric_proxy.py \
    --input_root "${INPUT_ROOT}" \
    --aggregate_root "${AGGREGATE_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --flow_max_side "${FLOW_MAX_SIDE}" \
    --topk "${TOPK}" \
    compute \
    --resume \
    --task_index "${TASK_ID}"
else
  python experiments/keyframe_budget/scripts/cheap_chunk_metric_proxy.py \
    --input_root "${INPUT_ROOT}" \
    --aggregate_root "${AGGREGATE_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --flow_max_side "${FLOW_MAX_SIDE}" \
    --topk "${TOPK}" \
    merge \
    --wait
fi
