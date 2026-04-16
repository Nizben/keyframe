#!/bin/bash
set -euo pipefail

# Usage:
#   bash experiments/keyframe_budget/scripts/submit_overnight_32gpus.sh [optional_smoke_job_id]
#
# Optional environment overrides:
#   ROOT_DIR=/path/to/Causal-Forcing
#   MAX_GPUS=32
#   SMOKE_JOB=2129592

if [ -z "${ROOT_DIR:-}" ]; then
  if [ -n "${WORK:-}" ] && [ -d "${WORK}/Causal-Forcing" ]; then
    ROOT_DIR="${WORK}/Causal-Forcing"
  else
    ROOT_DIR="$(pwd)"
  fi
fi

MAX_GPUS="${MAX_GPUS:-32}"
SMOKE_JOB="${1:-${SMOKE_JOB:-}}"

cd "${ROOT_DIR}"

DISC_SLURM="experiments/keyframe_budget/scripts/discovery_split_array.slurm"
MERGE_DISC_SLURM="experiments/keyframe_budget/scripts/merge_discovery_shards.slurm"
POLICY_SLURM="experiments/keyframe_budget/scripts/policy_eval_100_array.slurm"
MERGE_POLICY_SLURM="experiments/keyframe_budget/scripts/merge_policy_eval_100_shards.slurm"

for f in "$DISC_SLURM" "$MERGE_DISC_SLURM" "$POLICY_SLURM" "$MERGE_POLICY_SLURM"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing required file: ${ROOT_DIR}/$f"
    exit 1
  fi
done

if [ -n "$SMOKE_JOB" ] && [ -n "$(squeue -h -j "$SMOKE_JOB")" ]; then
  echo "[submit] smoke job ${SMOKE_JOB} still running; discovery will wait for it."
  DISC=$(sbatch --parsable --dependency=afterok:"${SMOKE_JOB}" --array=0-71%"${MAX_GPUS}" "$DISC_SLURM")
else
  if [ -n "$SMOKE_JOB" ]; then
    echo "[submit] smoke job ${SMOKE_JOB} not found in queue; submitting discovery now."
  fi
  DISC=$(sbatch --parsable --array=0-71%"${MAX_GPUS}" "$DISC_SLURM")
fi

MERGE_DISC=$(sbatch --parsable --dependency=afterok:"${DISC}" "$MERGE_DISC_SLURM")
POLICY=$(sbatch --parsable --dependency=afterok:"${MERGE_DISC}" --array=0-299%"${MAX_GPUS}" "$POLICY_SLURM")
MERGE_POLICY=$(sbatch --parsable --dependency=afterok:"${POLICY}" "$MERGE_POLICY_SLURM")

echo "[submit] ROOT_DIR=${ROOT_DIR}"
echo "[submit] MAX_GPUS=${MAX_GPUS}"
echo "[submit] DISC=${DISC}"
echo "[submit] MERGE_DISC=${MERGE_DISC}"
echo "[submit] POLICY=${POLICY}"
echo "[submit] MERGE_POLICY=${MERGE_POLICY}"
echo "[submit] Done. Current queue:"
squeue -u "$USER"
