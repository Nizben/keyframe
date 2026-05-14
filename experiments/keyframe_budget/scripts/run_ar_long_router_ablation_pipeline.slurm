#!/bin/bash
#SBATCH --job-name=kfb_arl_ablate_pipe
#SBATCH -A hdy@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/keyframe_budget/%x_%j.out
#SBATCH --error=logs/keyframe_budget/%x_%j.err

set -euo pipefail

# Submit the complete AR-long Amax/S ablation pipeline as a dependency chain.
# This orchestration job exits after submitting the chain; Slurm dependencies
# enforce the actual sequential execution.

ROOT_DIR="${ROOT_DIR:-${WORK:-/lustre/fswork/projects/rech/hdy/ujc37rw}/Causal-Forcing}"
VBENCH_DIR="${VBENCH_DIR:-/lustre/fswork/projects/rech/hdy/ujc37rw/VBench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ar_teacher_long_router_ablation}"

S_ARRAY_CONCURRENCY="${S_ARRAY_CONCURRENCY:-12}"
GEN_ARRAY_CONCURRENCY="${GEN_ARRAY_CONCURRENCY:-12}"
VBENCH_ARRAY_CONCURRENCY="${VBENCH_ARRAY_CONCURRENCY:-12}"
CPU_CONDA_ENV_NAME="${CPU_CONDA_ENV_NAME:-deepforcing}"
GEN_CONDA_ENV_NAME="${GEN_CONDA_ENV_NAME:-deepforcing}"
VBENCH_CONDA_ENV_NAME="${VBENCH_CONDA_ENV_NAME:-vbench_env}"

cd "${ROOT_DIR}"
mkdir -p logs/keyframe_budget

S_SLURM="experiments/keyframe_budget/scripts/compute_ar_long_solver_instability_array.slurm"
GEN_SLURM="experiments/keyframe_budget/scripts/ar_teacher_long_router_ablation_array.slurm"
VBENCH_SLURM="experiments/keyframe_budget/scripts/vbench_ar_teacher_long_router_ablation.slurm"

for path in "${S_SLURM}" "${GEN_SLURM}" "${VBENCH_SLURM}"; do
  if [ ! -f "${path}" ]; then
    echo "[ablation-pipeline] Missing required slurm file: ${path}"
    exit 1
  fi
done

submit_cpu_python() {
  local job_name="$1"
  local dependency="$2"
  local command="$3"
  local dep_args=()
  if [ -n "${dependency}" ]; then
    dep_args=(--dependency="afterok:${dependency}")
  fi
  sbatch --parsable \
    -A hdy@cpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --time=02:00:00 \
    --job-name="${job_name}" \
    --output="logs/keyframe_budget/%x_%j.out" \
    --error="logs/keyframe_budget/%x_%j.err" \
    "${dep_args[@]}" \
    --wrap="set -euo pipefail; module purge; module load arch/h100; module load cuda/12.1.0; module load miniforge/24.9.0; source ~/.bashrc; conda activate ${CPU_CONDA_ENV_NAME}; cd ${ROOT_DIR}; ${command}"
}

echo "[ablation-pipeline] ROOT_DIR=${ROOT_DIR}"
echo "[ablation-pipeline] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[ablation-pipeline] S_ARRAY_CONCURRENCY=${S_ARRAY_CONCURRENCY}"
echo "[ablation-pipeline] GEN_ARRAY_CONCURRENCY=${GEN_ARRAY_CONCURRENCY}"
echo "[ablation-pipeline] VBENCH_ARRAY_CONCURRENCY=${VBENCH_ARRAY_CONCURRENCY}"

S_JOB=$(sbatch --parsable \
  --array="0-71%${S_ARRAY_CONCURRENCY}" \
  --export=ALL,ROOT_DIR="${ROOT_DIR}",OUTPUT_ROOT="${OUTPUT_ROOT}",CONDA_ENV_NAME="${GEN_CONDA_ENV_NAME}" \
  "${S_SLURM}")

MERGE_S_JOB=$(submit_cpu_python \
  "kfb_merge_sinst" \
  "${S_JOB}" \
  "python experiments/keyframe_budget/scripts/merge_ar_long_solver_instability.py --glob '${OUTPUT_ROOT}/solver_instability/*/solver_instability.parquet' --out '${OUTPUT_ROOT}/solver_instability_scores.parquet'")

MERGE_SCORE_JOB=$(submit_cpu_python \
  "kfb_merge_s_score" \
  "${MERGE_S_JOB}" \
  "python experiments/keyframe_budget/scripts/merge_solver_instability_with_compression.py --out '${OUTPUT_ROOT}/compression_vbench_with_solver_instability.parquet'")

CHECK_PRE_GEN_JOB=$(submit_cpu_python \
  "kfb_check_ab_pre" \
  "${MERGE_SCORE_JOB}" \
  "python experiments/keyframe_budget/scripts/check_ar_long_router_ablation_ready.py --root '${OUTPUT_ROOT}'")

GEN_JOB=$(sbatch --parsable \
  --dependency="afterok:${CHECK_PRE_GEN_JOB}" \
  --array="0-71%${GEN_ARRAY_CONCURRENCY}" \
  --export=ALL,ROOT_DIR="${ROOT_DIR}",OUTPUT_ROOT="${OUTPUT_ROOT}",CONDA_ENV_NAME="${GEN_CONDA_ENV_NAME}" \
  "${GEN_SLURM}")

CHECK_POST_GEN_JOB=$(submit_cpu_python \
  "kfb_check_ab_post" \
  "${GEN_JOB}" \
  "python experiments/keyframe_budget/scripts/check_ar_long_router_ablation_ready.py --root '${OUTPUT_ROOT}'")

VBENCH_JOB=$(sbatch --parsable \
  --dependency="afterok:${CHECK_POST_GEN_JOB}" \
  --array="0-223%${VBENCH_ARRAY_CONCURRENCY}" \
  --export=ALL,ROOT_DIR="${ROOT_DIR}",VBENCH_DIR="${VBENCH_DIR}",INPUT_VIDEO_DIR="${ROOT_DIR}/${OUTPUT_ROOT}",CONDA_ENV_NAME="${VBENCH_CONDA_ENV_NAME}" \
  "${VBENCH_SLURM}")

AGG_JOB=$(submit_cpu_python \
  "kfb_agg_ab_vb" \
  "${VBENCH_JOB}" \
  "python experiments/keyframe_budget/scripts/aggregate_long_rollout_vbench.py --full_root '${VBENCH_DIR}/evaluation_results/keyframe_budget_ar_teacher_long_router_ablation/ar_teacher_long_router_ablation_full' --skip_suffix --output_root '${OUTPUT_ROOT}/vbench_analysis'")

ANALYZE_JOB=$(submit_cpu_python \
  "kfb_an_ablate" \
  "${AGG_JOB}" \
  "python experiments/keyframe_budget/scripts/analyze_ar_long_router_ablation.py --vbench '${OUTPUT_ROOT}/vbench_analysis/vbench_video_records.csv' --root '${OUTPUT_ROOT}' --out_dir '${OUTPUT_ROOT}/analysis'")

echo "[ablation-pipeline] Submitted dependency chain:"
echo "  S preview array:        ${S_JOB}"
echo "  Merge S:                ${MERGE_S_JOB}"
echo "  Merge compression+S:    ${MERGE_SCORE_JOB}"
echo "  Check pre-generation:   ${CHECK_PRE_GEN_JOB}"
echo "  Ablation generation:    ${GEN_JOB}"
echo "  Check post-generation:  ${CHECK_POST_GEN_JOB}"
echo "  VBench:                 ${VBENCH_JOB}"
echo "  Aggregate VBench:       ${AGG_JOB}"
echo "  Analyze:                ${ANALYZE_JOB}"
echo "[ablation-pipeline] Current queue:"
squeue -u "${USER}"
