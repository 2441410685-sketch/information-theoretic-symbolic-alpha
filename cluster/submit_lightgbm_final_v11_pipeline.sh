#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${INFO_ALPHA_PROJECT_DIR:-$HOME/Information_alpha}"
cd "$PROJECT_DIR"
mkdir -p experiments/lightgbm_final_v11

prepare_job="$(sbatch --parsable cluster/submit_lightgbm_final_v11_prepare.slurm)"
train_job="$(sbatch --parsable --dependency="afterok:${prepare_job}" cluster/submit_lightgbm_final_v11_train.slurm)"

printf 'prepare_job=%s\ntrain_job=%s\n' "$prepare_job" "$train_job" \
  | tee experiments/lightgbm_final_v11/submission_jobs.txt
