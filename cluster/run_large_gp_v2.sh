#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

config="configs/large_scale_gp_v2.json"
root="experiments/large_scale_gp_v2"
data_dir="$root/data"
validation_dir="$root/validation"
mkdir -p "$root/logs" "$data_dir" "$validation_dir"

# Reuse prepared panels so a restarted cluster job does not rescan the raw CSV.
if [[ ! -f "$data_dir/selection_panel.parquet" || ! -f "$data_dir/validation_panel.parquet" ]]; then
  python scripts/prepare_large_gp_panel.py \
    --mode prepare \
    --config "$config" \
    --output "$data_dir" \
    > "$root/logs/prepare.log" 2>&1
fi

# --resume starts from generation zero when no checkpoint exists and otherwise
# continues from the latest valid checkpoint.
python scripts/run_large_gp_cluster.py \
  --mode run --channel rankic --workers 8 \
  --config "$config" --panel "$data_dir/selection_panel.parquet" --output "$root" \
  --resume \
  > "$root/logs/rankic.log" 2>&1 &
rankic_pid=$!

python scripts/run_large_gp_cluster.py \
  --mode run --channel mi --workers 8 \
  --config "$config" --panel "$data_dir/selection_panel.parquet" --output "$root" \
  --resume \
  > "$root/logs/mi.log" 2>&1 &
mi_pid=$!

wait "$rankic_pid"
wait "$mi_pid"

python scripts/build_large_gp_union.py \
  --mode build \
  --rankic "$root/rankic/shortlist_top100.csv" \
  --mi "$root/mi/shortlist_top100.csv" \
  --output "$root/candidate_union_max200.csv" \
  > "$root/logs/union.log" 2>&1

python scripts/validate_large_gp_candidates.py \
  --mode run --workers 16 \
  --config "$config" \
  --candidates "$root/candidate_union_max200.csv" \
  --panel-directory "$data_dir" \
  --output "$validation_dir" \
  > "$root/logs/validation.log" 2>&1

# The 2023 validation is a development screen; this does not open the locked
# 2024--2025 final holdout.
python scripts/run_large_gp_validation_null.py \
  --mode run --workers 16 --permutations 999 --allow-long-run \
  --validation-directory "$validation_dir" \
  --candidates "$root/candidate_union_max200.csv" \
  --panel "$data_dir/validation_panel.parquet" \
  > "$root/logs/null.log" 2>&1

echo "V2 completed: $root"
