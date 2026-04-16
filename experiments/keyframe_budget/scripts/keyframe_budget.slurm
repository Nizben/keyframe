#!/bin/bash
#SBATCH --job-name=kfb_v35
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

ROOT_DIR="${ROOT_DIR:-$WORK/Causal-Forcing}"
source "${ROOT_DIR}/experiments/keyframe_budget/scripts/keyframe_budget_core.slurm"
exit 0

#SBATCH --job-name=kfb_v35
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

module purge
module load arch/h100
module load cuda/12.1.0
module load miniforge/24.9.0
source ~/.bashrc
conda activate deepforcing

export http_proxy=${http_proxy:-http://10.10.2.1:8123}
export https_proxy=${https_proxy:-http://10.10.2.1:8123}
export HTTP_PROXY=${HTTP_PROXY:-$http_proxy}
export HTTPS_PROXY=${HTTPS_PROXY:-$https_proxy}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-64}}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

resolve_snapshot_dir() {
  local repo_dir="$1"
  local snapshots_dir="$repo_dir/snapshots"

  if [ ! -d "$snapshots_dir" ]; then
    return 1
  fi

  local latest
  latest=$(find "$snapshots_dir" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
  if [ -n "$latest" ]; then
    printf "%s\n" "$latest"
    return 0
  fi

  return 1
}

ROOT_DIR=${ROOT_DIR:-$WORK/Causal-Forcing}
CONFIG_PATH=${EXP_CONFIG:-experiments/keyframe_budget/configs/smoke_test.yaml}
TMP_CONFIG=${SLURM_TMPDIR:-/tmp}/kfb_${SLURM_JOB_ID:-manual}.yaml
DEVICE=${DEVICE:-cuda:0}

cd "$ROOT_DIR"

mkdir -p logs/keyframe_budget
mkdir -p outputs/keyframe_budget
mkdir -p checkpoints/chunkwise
mkdir -p wan_models

export HF_HOME=${HF_HOME:-$WORK/.cache/huggingface}
HF_HUB_DIR="$HF_HOME/hub"
WAN_REPO_DIR=${WAN_REPO_DIR:-$HF_HUB_DIR/models--Wan-AI--Wan2.1-T2V-1.3B}
CF_REPO_DIR=${CF_REPO_DIR:-$HF_HUB_DIR/models--zhuhz22--Causal-Forcing}

WAN_SNAPSHOT_DIR=${WAN_SNAPSHOT_DIR:-$(resolve_snapshot_dir "$WAN_REPO_DIR" || true)}
CF_SNAPSHOT_DIR=${CF_SNAPSHOT_DIR:-$(resolve_snapshot_dir "$CF_REPO_DIR" || true)}

if [ -n "${WAN_SNAPSHOT_DIR:-}" ] && [ ! -e "wan_models/Wan2.1-T2V-1.3B" ]; then
  ln -s "$WAN_SNAPSHOT_DIR" "wan_models/Wan2.1-T2V-1.3B"
fi

if [ -n "${CF_SNAPSHOT_DIR:-}" ] && [ ! -e "checkpoints/chunkwise/ar_diffusion.pt" ] && [ -f "${CF_SNAPSHOT_DIR}/chunkwise/ar_diffusion.pt" ]; then
  ln -s "${CF_SNAPSHOT_DIR}/chunkwise/ar_diffusion.pt" "checkpoints/chunkwise/ar_diffusion.pt"
fi

CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
if [ -z "$CHECKPOINT_PATH" ] && [ -n "${CF_SNAPSHOT_DIR:-}" ] && [ -f "${CF_SNAPSHOT_DIR}/chunkwise/ar_diffusion.pt" ]; then
  CHECKPOINT_PATH="${CF_SNAPSHOT_DIR}/chunkwise/ar_diffusion.pt"
fi

export ROOT_DIR CONFIG_PATH TMP_CONFIG CHECKPOINT_PATH
python - <<'PY'
import os
from omegaconf import OmegaConf

def parse_bool(value: str):
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool from '{value}'.")

cfg = OmegaConf.load(os.environ["CONFIG_PATH"])
root_dir = os.environ["ROOT_DIR"]

if os.environ.get("CHECKPOINT_PATH"):
    cfg["checkpoint"] = os.environ["CHECKPOINT_PATH"]
if os.environ.get("OUTPUT_ROOT"):
    cfg["output_root"] = os.environ["OUTPUT_ROOT"]
if os.environ.get("PROMPT_SET"):
    cfg["prompt_set"] = os.environ["PROMPT_SET"]
if os.environ.get("MODEL_CONFIG_PATH"):
    cfg["model_config_path"] = os.environ["MODEL_CONFIG_PATH"]
if os.environ.get("ORACLE_SOURCE"):
    cfg["oracle_source"] = os.environ["ORACLE_SOURCE"]
if os.environ.get("DTYPE"):
    cfg["dtype"] = os.environ["DTYPE"]

for env_key, cfg_key in {
    "NUM_PROMPTS": "num_prompts",
    "FAST_STEPS": "fast_steps",
    "HEAVY_STEPS": "heavy_steps",
    "TARGET_AVG_STEPS": "target_avg_steps",
    "NUM_OUTPUT_FRAMES": "num_output_frames",
}.items():
    if os.getenv(env_key):
        cfg[cfg_key] = int(os.getenv(env_key))

if os.getenv("SEEDS"):
    cfg["seeds"] = [int(x.strip()) for x in os.getenv("SEEDS").split(",") if x.strip()]

if os.getenv("RUN_EXHAUSTIVE") is not None:
    cfg["run_exhaustive_single_heavy"] = parse_bool(os.getenv("RUN_EXHAUSTIVE"))
if os.getenv("RUN_POLICY") is not None:
    cfg["run_policy_comparison"] = parse_bool(os.getenv("RUN_POLICY"))
if os.getenv("USE_EMA") is not None:
    cfg["use_ema"] = parse_bool(os.getenv("USE_EMA"))
if os.getenv("DEBUG_CACHE_LOGS") is not None:
    cfg["debug_cache_logs"] = parse_bool(os.getenv("DEBUG_CACHE_LOGS"))

for key in ("checkpoint", "model_config_path", "prompt_set", "output_root"):
    value = cfg.get(key, "")
    if not value:
        raise ValueError(f"Missing required config key: {key}")
    if not os.path.isabs(value):
        cfg[key] = os.path.join(root_dir, value)

for key in ("checkpoint", "model_config_path", "prompt_set"):
    if not os.path.exists(cfg[key]):
        raise FileNotFoundError(f"{key} does not exist: {cfg[key]}")

OmegaConf.save(cfg, os.environ["TMP_CONFIG"])
print("[keyframe-budget] Effective config:")
print(OmegaConf.to_yaml(cfg))
PY

if [ "${FORCE:-0}" = "1" ]; then
  srun --ntasks=1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" \
    python -m experiments.keyframe_budget.run_experiment \
    --config_path "$TMP_CONFIG" \
    --device "$DEVICE" \
    --force
else
  srun --ntasks=1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" \
    python -m experiments.keyframe_budget.run_experiment \
    --config_path "$TMP_CONFIG" \
    --device "$DEVICE"
fi

# Safety stop: prevent executing any stale duplicated tail blocks.
exit 0
#!/bin/bash
#SBATCH --job-name=kfb_v35
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

resolve_snapshot_dir() {
  local repo_dir="$1"
  local snapshots_dir="$repo_dir/snapshots"

  if [ ! -d "$snapshots_dir" ]; then
    return 1
  fi

  local latest
  latest=$(find "$snapshots_dir" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
  if [ -n "$latest" ]; then
    printf "%s\n" "$latest"
    return 0
  fi

  return 1
}

module purge
module load arch/h100
module load cuda/12.1.0
module load miniforge/24.9.0
source ~/.bashrc
conda activate deepforcing

export http_proxy=${http_proxy:-http://10.10.2.1:8123}
export https_proxy=${https_proxy:-http://10.10.2.1:8123}
export HTTP_PROXY=${HTTP_PROXY:-$http_proxy}
export HTTPS_PROXY=${HTTPS_PROXY:-$https_proxy}

ROOT_DIR="${ROOT_DIR:-$WORK/Causal-Forcing}"
cd "$ROOT_DIR"

mkdir -p logs/keyframe_budget
mkdir -p outputs/keyframe_budget
mkdir -p checkpoints/chunkwise
mkdir -p wan_models

export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
HF_HUB_DIR="$HF_HOME/hub"
WAN_REPO_DIR=${WAN_REPO_DIR:-$HF_HUB_DIR/models--Wan-AI--Wan2.1-T2V-1.3B}
CF_REPO_DIR=${CF_REPO_DIR:-$HF_HUB_DIR/models--zhuhz22--Causal-Forcing}

WAN_SNAPSHOT_DIR=${WAN_SNAPSHOT_DIR:-$(resolve_snapshot_dir "$WAN_REPO_DIR" || true)}
CF_SNAPSHOT_DIR=${CF_SNAPSHOT_DIR:-$(resolve_snapshot_dir "$CF_REPO_DIR" || true)}

if [ -n "${WAN_SNAPSHOT_DIR:-}" ] && [ ! -e "wan_models/Wan2.1-T2V-1.3B" ]; then
  ln -s "$WAN_SNAPSHOT_DIR" "wan_models/Wan2.1-T2V-1.3B"
fi

if [ -n "${CF_SNAPSHOT_DIR:-}" ] && [ ! -e "checkpoints/chunkwise/ar_diffusion.pt" ] && [ -f "${CF_SNAPSHOT_DIR}/chunkwise/ar_diffusion.pt" ]; then
  ln -s "${CF_SNAPSHOT_DIR}/chunkwise/ar_diffusion.pt" "checkpoints/chunkwise/ar_diffusion.pt"
fi

CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
if [ -z "$CHECKPOINT_PATH" ] && [ -n "${CF_SNAPSHOT_DIR:-}" ] && [ -f "${CF_SNAPSHOT_DIR}/chunkwise/ar_diffusion.pt" ]; then
  CHECKPOINT_PATH="${CF_SNAPSHOT_DIR}/chunkwise/ar_diffusion.pt"
fi

CONFIG_PATH="${EXP_CONFIG:-experiments/keyframe_budget/configs/smoke_test.yaml}"
TMP_CONFIG="${SLURM_TMPDIR:-/tmp}/kfb_${SLURM_JOB_ID:-manual}.yaml"

export ROOT_DIR CONFIG_PATH TMP_CONFIG CHECKPOINT_PATH
python - <<'PY'
import os
from omegaconf import OmegaConf

def parse_bool(value: str):
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool from '{value}'.")

root_dir = os.environ["ROOT_DIR"]
cfg = OmegaConf.load(os.environ["CONFIG_PATH"])

if os.environ.get("CHECKPOINT_PATH"):
    cfg["checkpoint"] = os.environ["CHECKPOINT_PATH"]

for env_key, cfg_key in {
    "MODEL_CONFIG_PATH": "model_config_path",
    "PROMPT_SET": "prompt_set",
    "OUTPUT_ROOT": "output_root",
    "ORACLE_SOURCE": "oracle_source",
    "DTYPE": "dtype",
}.items():
    if os.getenv(env_key):
        cfg[cfg_key] = os.getenv(env_key)

for env_key, cfg_key in {
    "NUM_PROMPTS": "num_prompts",
    "FAST_STEPS": "fast_steps",
    "HEAVY_STEPS": "heavy_steps",
    "TARGET_AVG_STEPS": "target_avg_steps",
    "NUM_OUTPUT_FRAMES": "num_output_frames",
}.items():
    if os.getenv(env_key):
        cfg[cfg_key] = int(os.getenv(env_key))

if os.getenv("SEEDS"):
    cfg["seeds"] = [int(x.strip()) for x in os.getenv("SEEDS").split(",") if x.strip()]

if os.getenv("RUN_EXHAUSTIVE") is not None:
    cfg["run_exhaustive_single_heavy"] = parse_bool(os.getenv("RUN_EXHAUSTIVE"))
if os.getenv("RUN_POLICY") is not None:
    cfg["run_policy_comparison"] = parse_bool(os.getenv("RUN_POLICY"))
if os.getenv("USE_EMA") is not None:
    cfg["use_ema"] = parse_bool(os.getenv("USE_EMA"))
if os.getenv("DEBUG_CACHE_LOGS") is not None:
    cfg["debug_cache_logs"] = parse_bool(os.getenv("DEBUG_CACHE_LOGS"))

for key in ("checkpoint", "model_config_path", "prompt_set", "output_root"):
    value = cfg.get(key, "")
    if not value:
        raise ValueError(f"Missing required config key: {key}")
    if not os.path.isabs(value):
        cfg[key] = os.path.join(root_dir, value)

for key in ("checkpoint", "model_config_path", "prompt_set"):
    if not os.path.exists(cfg[key]):
        raise FileNotFoundError(f"{key} does not exist: {cfg[key]}")

OmegaConf.save(cfg, os.environ["TMP_CONFIG"])
print("[keyframe-budget] Effective config:")
print(OmegaConf.to_yaml(cfg))
PY

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-64}}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
DEVICE="${DEVICE:-cuda:0}"

if [ "${FORCE:-0}" = "1" ]; then
  srun --ntasks=1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" \
    python -m experiments.keyframe_budget.run_experiment \
    --config_path "$TMP_CONFIG" \
    --device "$DEVICE" \
    --force
else
  srun --ntasks=1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" \
    python -m experiments.keyframe_budget.run_experiment \
    --config_path "$TMP_CONFIG" \
    --device "$DEVICE"
fi
#!/bin/bash
#SBATCH --job-name=kfb_v35
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

resolve_snapshot_dir() {
  local repo_dir="$1"
  local snapshots_dir="$repo_dir/snapshots"
  if [ ! -d "$snapshots_dir" ]; then
    return 1
  fi

  local latest=""
  for d in "$snapshots_dir"/*; do
    if [ -d "$d" ]; then
      latest="$d"
    fi
  done

  if [ -n "$latest" ]; then
    printf "%s\n" "$latest"
    return 0
  fi
  return 1
}

# -----------------------------------------------------------------------------
# Cluster + environment setup
# -----------------------------------------------------------------------------
if [ -z "${ROOT_DIR:-}" ]; then
  if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}/experiments/keyframe_budget/scripts" ]; then
    ROOT_DIR="${SLURM_SUBMIT_DIR}"
  elif [ -n "${WORK:-}" ] && [ -d "${WORK}/Causal-Forcing/experiments/keyframe_budget/scripts" ]; then
    ROOT_DIR="${WORK}/Causal-Forcing"
  else
    echo "[keyframe-budget] ERROR: Unable to infer ROOT_DIR. Submit from repo root or set ROOT_DIR."
    exit 1
  fi
fi

# Optional override. If unset, the script auto-selects an env that has omegaconf.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
# Jean Zay cache layout shared by the user:
#   $WORK/.cache/huggingface/hub/models--zhuhz22--Causal-Forcing/snapshots/<sha>/chunkwise/ar_diffusion.pt
#   $WORK/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/<sha>/
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"

cd "$ROOT_DIR"
mkdir -p logs/keyframe_budget outputs/keyframe_budget checkpoints/chunkwise wan_models

# Activate conda env (same env convention as your other repo script).
if ! command -v conda >/dev/null 2>&1; then
  for conda_bin in \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "$WORK/conda_envs/miniconda3/bin/conda" \
    "$WORK/conda_envs/miniforge3/bin/conda"; do
    if [ -x "$conda_bin" ]; then
      export PATH="$(dirname "$conda_bin"):$PATH"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[keyframe-budget] ERROR: conda not found on PATH."
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

chosen_env=""
candidate_envs=()
if [ -n "${CONDA_ENV_NAME}" ]; then
  candidate_envs+=("${CONDA_ENV_NAME}")
fi
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" != "base" ]; then
  candidate_envs+=("${CONDA_DEFAULT_ENV}")
fi
candidate_envs+=("causal_forcing" "deepforcing" "base")

for env_name in "${candidate_envs[@]}"; do
  if [ -z "${env_name}" ]; then
    continue
  fi

  if conda activate "${env_name}" >/dev/null 2>&1; then
    if python - <<'PY' >/dev/null 2>&1
import omegaconf  # noqa: F401
PY
    then
      chosen_env="${env_name}"
      break
    fi
  fi
done

if [ -z "${chosen_env}" ]; then
  echo "[keyframe-budget] ERROR: Could not find a usable conda env with 'omegaconf'."
  echo "[keyframe-budget] Tried: ${candidate_envs[*]}"
  echo "[keyframe-budget] Install deps in one env or set CONDA_ENV_NAME explicitly in the script."
  exit 1
fi
echo "[keyframe-budget] Using conda env: ${chosen_env}"

# Optional convenience: auto-link Wan and checkpoint from HF cache snapshots.
HF_HUB_DIR="${HF_HUB_DIR:-$HF_HOME/hub}"
WAN_REPO_DIR="${WAN_REPO_DIR:-$HF_HUB_DIR/models--Wan-AI--Wan2.1-T2V-1.3B}"
CF_REPO_DIR="${CF_REPO_DIR:-$HF_HUB_DIR/models--zhuhz22--Causal-Forcing}"

WAN_SNAPSHOT_DIR="${WAN_SNAPSHOT_DIR:-$(resolve_snapshot_dir "$WAN_REPO_DIR" || true)}"
CF_SNAPSHOT_DIR="${CF_SNAPSHOT_DIR:-$(resolve_snapshot_dir "$CF_REPO_DIR" || true)}"

if [ -n "${WAN_SNAPSHOT_DIR:-}" ] && [ ! -e "wan_models/Wan2.1-T2V-1.3B" ]; then
  ln -s "$WAN_SNAPSHOT_DIR" "wan_models/Wan2.1-T2V-1.3B"
fi

if [ -n "${CF_SNAPSHOT_DIR:-}" ] && [ ! -e "checkpoints/chunkwise/ar_diffusion.pt" ] && [ -f "${CF_SNAPSHOT_DIR}/chunkwise/ar_diffusion.pt" ]; then
  ln -s "${CF_SNAPSHOT_DIR}/chunkwise/ar_diffusion.pt" "checkpoints/chunkwise/ar_diffusion.pt"
fi

# Default checkpoint path directly from HF cache snapshot if available.
if [ -z "${CHECKPOINT_PATH:-}" ] && [ -n "${CF_SNAPSHOT_DIR:-}" ] && [ -f "${CF_SNAPSHOT_DIR}/chunkwise/ar_diffusion.pt" ]; then
  export CHECKPOINT_PATH="${CF_SNAPSHOT_DIR}/chunkwise/ar_diffusion.pt"
fi

# -----------------------------------------------------------------------------
# Experiment config patching via env vars (same spirit as your smoke script)
# -----------------------------------------------------------------------------
BASE_CONFIG="${EXP_CONFIG:-experiments/keyframe_budget/configs/smoke_test.yaml}"
TMP_CONFIG="${SLURM_TMPDIR:-/tmp}/kfb_${SLURM_JOB_ID:-manual}.yaml"

export BASE_CONFIG TMP_CONFIG
python - <<'PY'
import os
from omegaconf import OmegaConf

def parse_bool(value: str):
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool from '{value}'.")

root_dir = os.environ.get("ROOT_DIR", "")
cfg = OmegaConf.load(os.environ["BASE_CONFIG"])

str_overrides = {
    "checkpoint": os.getenv("CHECKPOINT_PATH"),
    "prompt_set": os.getenv("PROMPT_SET"),
    "output_root": os.getenv("OUTPUT_ROOT"),
    "model_config_path": os.getenv("MODEL_CONFIG_PATH"),
    "oracle_source": os.getenv("ORACLE_SOURCE"),
    "dtype": os.getenv("DTYPE"),
}
for key, value in str_overrides.items():
    if value:
        cfg[key] = value

int_overrides = {
    "NUM_PROMPTS": "num_prompts",
    "FAST_STEPS": "fast_steps",
    "HEAVY_STEPS": "heavy_steps",
    "TARGET_AVG_STEPS": "target_avg_steps",
    "NUM_OUTPUT_FRAMES": "num_output_frames",
}
for env_key, cfg_key in int_overrides.items():
    env_val = os.getenv(env_key)
    if env_val:
        cfg[cfg_key] = int(env_val)

if os.getenv("SEEDS"):
    cfg["seeds"] = [int(x.strip()) for x in os.getenv("SEEDS").split(",") if x.strip()]

if os.getenv("RUN_EXHAUSTIVE") is not None:
    cfg["run_exhaustive_single_heavy"] = parse_bool(os.getenv("RUN_EXHAUSTIVE"))
if os.getenv("RUN_POLICY") is not None:
    cfg["run_policy_comparison"] = parse_bool(os.getenv("RUN_POLICY"))
if os.getenv("USE_EMA") is not None:
    cfg["use_ema"] = parse_bool(os.getenv("USE_EMA"))
if os.getenv("DEBUG_CACHE_LOGS") is not None:
    cfg["debug_cache_logs"] = parse_bool(os.getenv("DEBUG_CACHE_LOGS"))

# Resolve and validate critical paths before launch.
if root_dir:
    if not os.path.isabs(cfg["checkpoint"]):
        cfg["checkpoint"] = os.path.join(root_dir, cfg["checkpoint"])
    if not os.path.isabs(cfg["model_config_path"]):
        cfg["model_config_path"] = os.path.join(root_dir, cfg["model_config_path"])
    if not os.path.isabs(cfg["prompt_set"]):
        cfg["prompt_set"] = os.path.join(root_dir, cfg["prompt_set"])
    if not os.path.isabs(cfg["output_root"]):
        cfg["output_root"] = os.path.join(root_dir, cfg["output_root"])

for key in ("checkpoint", "model_config_path", "prompt_set"):
    path = cfg.get(key, "")
    if not path:
        raise ValueError(f"Missing required config key: {key}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{key} does not exist: {path}")

OmegaConf.save(cfg, os.environ["TMP_CONFIG"])
print("[keyframe-budget] Effective config:")
print(OmegaConf.to_yaml(cfg))
PY

# -----------------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-64}}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
DEVICE="${DEVICE:-cuda:0}"

CMD=(python -m experiments.keyframe_budget.run_experiment --config_path "$TMP_CONFIG" --device "$DEVICE")
if [ "${FORCE:-0}" = "1" ]; then
  CMD+=(--force)
fi

printf "[keyframe-budget] Running command: %s\n" "${CMD[*]}"
srun --ntasks=1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" "${CMD[@]}"
