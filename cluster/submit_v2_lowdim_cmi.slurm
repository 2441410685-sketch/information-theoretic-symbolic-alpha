#!/usr/bin/env bash
#SBATCH --job-name=infoalpha-cmi
#SBATCH --partition=cpu_2020
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-cmi-%j.out
#SBATCH --error=slurm-cmi-%j.err

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$project_root"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p experiments/large_scale_gp_v2/cmi_selection

python scripts/run_v2_lowdim_cmi_selection.py \
  --mode run \
  --allow-long-run \
  --workers 16 \
  --max-factors 4 \
  --permutations 99 \
  --block-length 20 \
  --output experiments/large_scale_gp_v2/cmi_selection
