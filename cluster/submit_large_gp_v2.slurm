#!/usr/bin/env bash
#SBATCH --job-name=infoalpha-v2
#SBATCH --partition=cpu_2020
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-v2-%j.out
#SBATCH --error=slurm-v2-%j.err

set -euo pipefail

# Slurm copies this file into a temporary spool directory. Use the original
# sbatch working directory instead of deriving the project root from BASH_SOURCE.
project_root="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
cd "$project_root"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

bash "$project_root/cluster/run_large_gp_v2.sh"
