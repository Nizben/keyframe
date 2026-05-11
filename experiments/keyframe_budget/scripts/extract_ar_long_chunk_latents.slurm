#!/bin/bash
#SBATCH --job-name=kfb_comp_lat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=logs/keyframe_budget/%x-%A_%a.stdout.log
#SBATCH --error=logs/keyframe_budget/%x-%A_%a.stderr.log
#SBATCH -C h100
#SBATCH -A hdy@h100
#SBATCH --array=0-71%12

set -euo pipefail

module purge
module load arch/${JEANZAY_ARCH:-h100}
module load cuda/12.1.0
module load miniforge/24.9.0
source ~/.bashrc
conda activate deepforcing

export http_proxy=${http_proxy:-http://10.10.2.1:8123}
export https_proxy=${https_proxy:-http://10.10.2.1:8123}
export HTTP_PROXY=${HTTP_PROXY:-$http_proxy}
export HTTPS_PROXY=${HTTPS_PROXY:-$https_proxy}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-32}}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

ROOT_DIR="${ROOT_DIR:-$WORK/Causal-Forcing}"
cd "${ROOT_DIR}"
mkdir -p logs/keyframe_budget
DEFAULT_SCHEDULES="all_fast,all_heavy"
for IDX in $(seq -w 0 39); do
  DEFAULT_SCHEDULES="${DEFAULT_SCHEDULES},single_heavy_${IDX}"
done

python -m experiments.keyframe_budget.scripts.extract_ar_long_chunk_latents \
  --runs_dir "${RUNS_DIR:-outputs/ar_teacher_long_keyframe_budget}" \
  --out_dir "${OUT_DIR:-outputs/ar_teacher_long_keyframe_budget/compression_transition}" \
  --task_id "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" \
  --device "${DEVICE:-cuda:0}" \
  --dtype "${DTYPE:-bfloat16}" \
  --schedules "${SCHEDULES:-${DEFAULT_SCHEDULES}}" \
  --save_mode "${SAVE_MODE:-lean}"
