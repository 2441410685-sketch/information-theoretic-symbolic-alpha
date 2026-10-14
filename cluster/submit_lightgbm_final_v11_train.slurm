#!/usr/bin/env bash
#SBATCH --job-name=infoalpha-v11-train
#SBATCH --partition=cpu_2020
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-lgbm-v11-train-%j.out
#SBATCH --error=slurm-lgbm-v11-train-%j.err

set -euo pipefail

PROJECT_DIR="${INFO_ALPHA_PROJECT_DIR:-$HOME/Information_alpha}"
cd "$PROJECT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base

LIGHTGBM_VENDOR="$PROJECT_DIR/.vendor/lightgbm335"
export PYTHONPATH="$LIGHTGBM_VENDOR:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if ! python -c "import lightgbm" >/dev/null 2>&1; then
    echo "ERROR: 找不到 V1 已验证的 LightGBM 3.3.5: $LIGHTGBM_VENDOR" >&2
    exit 2
fi

python -c "import lightgbm; print('lightgbm', lightgbm.__version__, lightgbm.__file__)"
python scripts/run_lightgbm_final_v11.py \
  --config configs/lightgbm_final_v11.json \
  --data-dir experiments/lightgbm_final_v11/data \
  --output-dir experiments/lightgbm_final_v11/results
