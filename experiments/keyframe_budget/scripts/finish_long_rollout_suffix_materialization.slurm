#!/bin/bash
#SBATCH --job-name=kfb_suffix_finish
#SBATCH -A hdy@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --array=0-3023%64
#SBATCH --output=logs/keyframe_budget/%x_%A_%a.out
#SBATCH --error=logs/keyframe_budget/%x_%A_%a.err

set -euo pipefail

# Finish fixed-window suffix materialization for all long-rollout discovery outputs.
#
# This array is intentionally safe to rerun:
# - each task owns one rollout schedule directory,
# - complete outputs are skipped,
# - existing suffix clips are not overwritten unless SUFFIX_OVERWRITE=1.
#
# Submit from the repo root on Jean Zay, for example:
#   sbatch experiments/keyframe_budget/scripts/finish_long_rollout_suffix_materialization.slurm
#
# If the CPU account differs on your allocation, change the #SBATCH -A line above
# or submit with an explicit account override.

module purge
module load miniforge/24.9.0
source ~/.bashrc

CONDA_ENV_NAME="${CONDA_ENV_NAME:-vbench_env}"
conda activate "${CONDA_ENV_NAME}"

ROOT_DIR="${ROOT_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/Causal-Forcing}"
INPUT_VIDEO_DIR="${INPUT_VIDEO_DIR:-${ROOT_DIR}/outputs/long_keyframe_budget}"
SUFFIX_OUTPUT_ROOT="${SUFFIX_OUTPUT_ROOT:-${ROOT_DIR}/outputs/long_keyframe_budget_suffixes}"
RUN_TAG="${RUN_TAG:-long_rollout_suffix_finish}"
SUFFIX_OVERWRITE="${SUFFIX_OVERWRITE:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"

cd "${ROOT_DIR}"
mkdir -p logs/keyframe_budget

if [ ! -d "${INPUT_VIDEO_DIR}" ]; then
  echo "[suffix-finish] INPUT_VIDEO_DIR not found: ${INPUT_VIDEO_DIR}"
  exit 1
fi

if [ ! -d "${SUFFIX_OUTPUT_ROOT}" ]; then
  mkdir -p "${SUFFIX_OUTPUT_ROOT}"
fi

TASK_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
TASK_DIR="${SLURM_TMPDIR:-/tmp}/kfb_suffix_finish_${SLURM_ARRAY_JOB_ID:-manual}_${TASK_INDEX}"
mkdir -p "${TASK_DIR}"
TASK_MANIFEST="${TASK_DIR}/rollout_manifest.txt"

export INPUT_VIDEO_DIR SUFFIX_OUTPUT_ROOT TASK_INDEX TASK_MANIFEST

set +e
python - <<'PY'
import os
import sys
import json
from pathlib import Path

input_root = Path(os.environ["INPUT_VIDEO_DIR"]).resolve()
output_root = Path(os.environ["SUFFIX_OUTPUT_ROOT"]).resolve()
task_index = int(os.environ["TASK_INDEX"])
task_manifest = Path(os.environ["TASK_MANIFEST"])

rollouts = []
for shard in sorted(input_root.glob("long_rollout_discovery_p???_s?")):
    rollout_root = shard / "rollouts"
    if not rollout_root.exists():
        continue
    for full_video in sorted(rollout_root.glob("*/*/*/full.mp4")):
        rollout_dir = full_video.parent
        if (rollout_dir / "chunk_boundaries.json").exists():
            rollouts.append(rollout_dir)

if task_index >= len(rollouts):
    print(
        f"[suffix-finish] task_index={task_index} out of range for {len(rollouts)} rollouts; exiting."
    )
    raise SystemExit(20)

rollout_dir = rollouts[task_index]
rel = rollout_dir.relative_to(input_root)
suffix_dir = output_root / rel / "suffixes"
manifest = suffix_dir / "suffix_manifest.json"
clips = sorted(suffix_dir.glob("suffix_from_*.mp4")) if suffix_dir.exists() else []

manifest_ok = False
if manifest.exists():
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        manifest_ok = len(payload.get("clips", [])) == 40
    except Exception:
        manifest_ok = False

complete = manifest_ok and len(clips) == 40 and all(path.stat().st_size > 0 for path in clips)
if complete:
    print(f"[suffix-finish] complete skip: {rel}")
    raise SystemExit(10)

task_manifest.parent.mkdir(parents=True, exist_ok=True)
task_manifest.write_text(
    "".join(f"{path}\n" for path in rollouts),
    encoding="utf-8",
)
print(
    f"[suffix-finish] selected task_index={task_index} rel={rel} "
    f"existing_clips={len(clips)} manifest_ok={manifest_ok}"
)
PY
PY_STATUS="$?"
set -e
case "${PY_STATUS}" in
  0)
    ;;
  10|20)
    exit 0
    ;;
  *)
    exit 1
    ;;
esac

CMD=(
  python "${ROOT_DIR}/experiments/keyframe_budget/scripts/materialize_long_rollout_suffixes.py"
  --input_root "${INPUT_VIDEO_DIR}"
  --output_root "${SUFFIX_OUTPUT_ROOT}"
  --manifest_path "${TASK_MANIFEST}"
  --task_index "${TASK_INDEX}"
)
if [ "${SUFFIX_OVERWRITE}" = "1" ]; then
  CMD+=(--overwrite)
fi

echo "[suffix-finish] running: ${CMD[*]}"
"${CMD[@]}"

TASK_SUMMARY_NAME=$(printf "task_%06d.json" "${SLURM_ARRAY_TASK_ID}")
SUMMARY_ROOT="${SUFFIX_OUTPUT_ROOT}/_suffix_materialization/${RUN_TAG}"
mkdir -p "${SUMMARY_ROOT}/tasks"
cp "${SUFFIX_OUTPUT_ROOT}/_suffix_materialization/tasks/${TASK_SUMMARY_NAME}" \
  "${SUMMARY_ROOT}/tasks/${TASK_SUMMARY_NAME}" 2>/dev/null || true
echo "[suffix-finish] done task=${SLURM_ARRAY_TASK_ID}"
