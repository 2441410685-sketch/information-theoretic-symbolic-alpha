#!/usr/bin/env bash
#SBATCH --job-name=infoalpha-v11-prep
#SBATCH --partition=cpu_2020
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-lgbm-v11-prep-%j.out
#SBATCH --error=slurm-lgbm-v11-prep-%j.err

set -euo pipefail

PROJECT_DIR="${INFO_ALPHA_PROJECT_DIR:-$HOME/Information_alpha}"
cd "$PROJECT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python scripts/prepare_lightgbm_final_v1.py \
  --config configs/lightgbm_final_v11.json \
  --output-dir experiments/lightgbm_final_v11/data
