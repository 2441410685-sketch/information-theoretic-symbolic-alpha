"""用 2022 selection 样本执行 V2 的低维 CMI forward selection。

流程固定为：selection-only 绝对相关阈值 0.98 近重复压缩，然后依次最大化
``I(f;Y)`` 与 ``I(f;Y|S)``。条件集最多含 3 个因子，因此联合 kNN 空间最高
为 5 维。脚本不读取 2023 validation 或 2024--2025 Final Holdout。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Iterable

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.information.conditional_mi import estimate_cmi_frenzel_pompe_details  # noqa: E402
from src.information.mutual_information import estimate_mi_ksg_details  # noqa: E402
from src.information.significance import (  # noqa: E402
    block_permutation_indices,
    monte_carlo_p_value,
)
from src.validation.large_gp_candidates import (  # noqa: E402
    load_candidate_union,
    select_semantic_representatives,
)


DEFAULT_ROOT = PROJECT_ROOT / "experiments" / "large_scale_gp_v2"
DEFAULT_OUTPUT = DEFAULT_ROOT / "cmi_selection"
BASE_SEED = 20260924

_WORKER: dict[str, object] = {}


def _stable_seed(text: str, base_seed: int) -> int:
    """由候选与步骤生成跨进程稳定的随机种子。"""

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int((base_seed + int(digest[:8], 16)) % (2**32 - 1))


def build_corr098_representatives(
    candidate_path: Path,
    correlation_path: Path,
    *,
    threshold: float = 0.98,
) -> pd.DataFrame:
    """只使用 selection 相关矩阵压缩近重复候选。"""

    candidates = load_candidate_union(candidate_path)
    correlation = pd.read_csv(correlation_path, index_col=0)
    representatives = select_semantic_representatives(
        candidates,
        correlation,
        threshold=threshold,
    )
    return representatives


def estimate_workload(
    representative_count: int,
    max_factors: int,
    permutations: int,
    workers: int,
) -> dict[str, float | int | str]:
    """给出估计器调用量和保守集群时间范围。"""

    candidates_per_step = [representative_count - step for step in range(max_factors)]
    observed_calls = int(sum(candidates_per_step))
    half_calls = int(2 * max_factors)
    null_calls = int(permutations * observed_calls)
    total_calls = observed_calls + half_calls + null_calls
    # 120000 行、2--5 维 cKDTree 的耗时强依赖节点负载，因此保留宽区间。
    low_core_hours = total_calls * 3.0 / 3600.0
    high_core_hours = total_calls * 30.0 / 3600.0
    return {
        "representative_count": representative_count,
        "max_selected_factors": max_factors,
        "maximum_conditioning_dimension": max_factors - 1,
        "maximum_joint_dimension": max_factors + 1,
        "permutations": permutations,
        "observed_estimator_calls": observed_calls,
        "half_window_estimator_calls": half_calls,
        "null_estimator_calls": null_calls,
        "total_estimator_calls": total_calls,
        "workers": workers,
        "rough_core_hours_low": round(low_core_hours, 2),
        "rough_core_hours_high": round(high_core_hours, 2),
        "execution_policy": "cluster_only",
    }


def _load_selection_arrays(
    factor_path: Path,
    panel_path: Path,
    candidate_ids: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """读取冻结候选与 label，并严格核对 2022 的 date-stock 键。"""

    factors = pd.read_parquet(factor_path, columns=["date", "stock", *candidate_ids])
    factors["date"] = pd.to_datetime(factors["date"])
    factors["stock"] = factors["stock"].astype("string")
    factors = factors.sort_values(["date", "stock"]).reset_index(drop=True)

    panel = pd.read_parquet(
        panel_path,
        columns=["date", "stock", "label_rankgauss", "is_fitness_sample"],
    )
    panel["date"] = pd.to_datetime(panel["date"])
    panel["stock"] = panel["stock"].astype("string")
    sample = (
        panel.loc[
            panel["is_fitness_sample"].astype(bool),
            ["date", "stock", "label_rankgauss"],
        ]
        .sort_values(["date", "stock"])
        .reset_index(drop=True)
    )
    if not factors[["date", "stock"]].equals(sample[["date", "stock"]]):
        raise ValueError("selection 因子缓存与 panel 的冻结 date-stock 键不一致")
    years = sample["date"].dt.year.unique().tolist()
    if years != [2022]:
        raise ValueError(f"CMI selection 只能读取 2022，实际年份为 {years}")
    counts = sample.groupby("date", sort=True).size()
    if len(counts) != 240 or not counts.eq(500).all():
        raise ValueError("CMI selection 要求严格的 240 日×500 股冻结样本")

    factor_matrix = factors[list(candidate_ids)].to_numpy(dtype=np.float64)
    label = sample["label_rankgauss"].to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(label)):
        raise ValueError("selection label_rankgauss 不得包含 NaN/inf")
    return factor_matrix, label.reshape(len(counts), int(counts.iloc[0])), len(counts), int(counts.iloc[0])


def _initialize_worker(
    factor_path: str,
    panel_path: str,
    candidate_ids: tuple[str, ...],
    permutations: int,
    block_length: int,
    base_seed: int,
) -> None:
    """每个进程只读取一次 2022 冻结矩阵，避免逐任务重复 IO。"""

    matrix, label, date_count, stock_count = _load_selection_arrays(
        Path(factor_path),
        Path(panel_path),
        candidate_ids,
    )
    rng = np.random.default_rng(base_seed)
    date_permutations = np.vstack(
        [
            block_permutation_indices(date_count, block_length, rng)
            for _ in range(permutations)
        ]
    )
    _WORKER.update(
        {
            "factor_matrix": matrix,
            "label_matrix": label,
            "candidate_ids": candidate_ids,
            "id_to_column": {name: index for index, name in enumerate(candidate_ids)},
            "date_permutations": date_permutations,
            "date_count": date_count,
            "stock_count": stock_count,
        }
    )


def _score_task(task: tuple[object, ...]) -> dict[str, object]:
    """在一个进程中估计一个候选的 MI 或低维 CMI。"""

    (
        candidate,
        selected,
        repetition,
        first_date,
        last_date,
        k,
        min_observations,
        random_state,
    ) = task
    candidate = str(candidate)
    selected = tuple(str(name) for name in selected)
    repetition = int(repetition)
    first_date = int(first_date)
    last_date = int(last_date)
    id_to_column = _WORKER["id_to_column"]
    matrix = _WORKER["factor_matrix"]
    label_matrix = _WORKER["label_matrix"]
    stock_count = int(_WORKER["stock_count"])

    row_start = first_date * stock_count
    row_end = last_date * stock_count
    x = matrix[row_start:row_end, id_to_column[candidate]]
    if repetition >= 0:
        date_order = _WORKER["date_permutations"][repetition]
        y_full = label_matrix[date_order].reshape(-1)
    else:
        y_full = label_matrix.reshape(-1)
    y = y_full[row_start:row_end]
    z = None
    if selected:
        selected_columns = [id_to_column[name] for name in selected]
        z = matrix[row_start:row_end, selected_columns]

    arrays = [x, y] if z is None else [x, y, *[z[:, i] for i in range(z.shape[1])]]
    finite = np.logical_and.reduce([np.isfinite(array) for array in arrays])
    if int(finite.sum()) < int(min_observations):
        return {
            "candidate_id": candidate,
            "repetition": repetition,
            "score": np.nan,
            "n_observations": int(finite.sum()),
            "status": "insufficient_observations",
        }
    x_valid = x[finite]
    y_valid = y[finite]
    if z is None:
        score = estimate_mi_ksg_details(
            x_valid,
            y_valid,
            k=int(k),
            tie_strategy="jitter",
            random_state=int(random_state),
        ).estimate
        score_type = "mi"
    else:
        score = estimate_cmi_frenzel_pompe_details(
            x_valid,
            y_valid,
            z[finite],
            k=int(k),
            tie_strategy="jitter",
            random_state=int(random_state),
        ).estimate
        score_type = "cmi"
    return {
        "candidate_id": candidate,
        "repetition": repetition,
        "score": float(score),
        "score_type": score_type,
        "n_observations": int(finite.sum()),
        "status": "estimated",
    }


def _make_tasks(
    candidates: Iterable[str],
    selected: tuple[str, ...],
    *,
    repetitions: Iterable[int],
    first_date: int,
    last_date: int,
    k: int,
    min_observations: int,
    base_seed: int,
    step: int,
) -> list[tuple[object, ...]]:
    """建立可复现的候选×置换任务列表。"""

    tasks: list[tuple[object, ...]] = []
    for repetition in repetitions:
        for candidate in candidates:
            seed = _stable_seed(
                f"{candidate}|{','.join(selected)}|{step}|{repetition}|{first_date}|{last_date}",
                base_seed,
            )
            tasks.append(
                (
                    candidate,
                    selected,
                    repetition,
                    first_date,
                    last_date,
                    k,
                    min_observations,
                    seed,
                )
            )
    return tasks


def run_forward_selection(
    representatives: pd.DataFrame,
    factor_path: Path,
    panel_path: Path,
    *,
    workers: int,
    max_factors: int,
    k: int,
    min_observations: int,
    permutations: int,
    block_length: int,
    base_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """并行运行最多四步的低维 CMI 前向选择。"""

    if not 1 <= max_factors <= 4:
        raise ValueError("本项目的低维 CMI 最多选择4个因子")
    if permutations < 19:
        raise ValueError("至少需要19次置换，才能用加一修正得到0.05级 p-value")
    candidate_ids = tuple(representatives["candidate_id"].astype(str))
    selected: list[str] = []
    audit_rows: list[dict[str, object]] = []
    score_parts: list[pd.DataFrame] = []
    null_rows: list[dict[str, object]] = []
    stop_reason = "maximum_factor_count_reached"

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(
            str(factor_path.resolve()),
            str(panel_path.resolve()),
            candidate_ids,
            permutations,
            block_length,
            base_seed,
        ),
    ) as executor:
        for step in range(1, max_factors + 1):
            remaining = [name for name in candidate_ids if name not in selected]
            observed_tasks = _make_tasks(
                remaining,
                tuple(selected),
                repetitions=[-1],
                first_date=0,
                last_date=240,
                k=k,
                min_observations=min_observations,
                base_seed=base_seed,
                step=step,
            )
            observed = pd.DataFrame(executor.map(_score_task, observed_tasks, chunksize=1))
            observed = observed.sort_values(
                ["score", "candidate_id"],
                ascending=[False, True],
                na_position="last",
            ).reset_index(drop=True)
            observed.insert(0, "step", step)
            observed.insert(1, "selected_before", ",".join(selected))
            observed["rank"] = np.arange(1, len(observed) + 1)
            score_parts.append(observed)
            valid = observed.loc[observed["status"].eq("estimated") & observed["score"].notna()]
            if valid.empty:
                stop_reason = "no_estimable_candidate"
                break
            winner = str(valid.iloc[0]["candidate_id"])
            winner_score = float(valid.iloc[0]["score"])

            half_tasks = []
            for first_date, last_date in ((0, 120), (120, 240)):
                half_tasks.extend(
                    _make_tasks(
                        [winner],
                        tuple(selected),
                        repetitions=[-1],
                        first_date=first_date,
                        last_date=last_date,
                        k=k,
                        min_observations=max(k + 1, min_observations // 3),
                        base_seed=base_seed,
                        step=step,
                    )
                )
            half_scores = [float(row["score"]) for row in executor.map(_score_task, half_tasks)]

            null_tasks = _make_tasks(
                remaining,
                tuple(selected),
                repetitions=range(permutations),
                first_date=0,
                last_date=240,
                k=k,
                min_observations=min_observations,
                base_seed=base_seed,
                step=step,
            )
            null = pd.DataFrame(executor.map(_score_task, null_tasks, chunksize=1))
            maximum_null = (
                null.groupby("repetition", sort=True)["score"].max().to_numpy(dtype=float)
            )
            for repetition, maximum in enumerate(maximum_null, start=1):
                null_rows.append(
                    {
                        "step": step,
                        "repetition": repetition,
                        "maximum_null_score": maximum,
                    }
                )
            null_q95 = float(np.quantile(maximum_null, 0.95, method="higher"))
            p_value, exceedances = monte_carlo_p_value(
                winner_score,
                maximum_null,
                alternative="greater",
            )
            passes_null = bool(winner_score > null_q95 and p_value <= 0.05)
            passes_stability = bool(min(half_scores) > 0.0)
            accepted = bool(passes_null and passes_stability)
            if not passes_null:
                decision = "stop_max_null"
                stop_reason = "winner_did_not_exceed_block_max_null"
            elif not passes_stability:
                decision = "stop_unstable_halves"
                stop_reason = "winner_not_positive_in_both_selection_halves"
            else:
                decision = "accept"
            audit_rows.append(
                {
                    "step": step,
                    "conditioning_dimension": len(selected),
                    "selected_before": ",".join(selected),
                    "winner": winner,
                    "winner_score": winner_score,
                    "score_type": "mi" if not selected else "cmi",
                    "first_half_score": half_scores[0],
                    "second_half_score": half_scores[1],
                    "null_mean": float(np.mean(maximum_null)),
                    "null_q95": null_q95,
                    "max_null_p_value": p_value,
                    "null_exceedances": exceedances,
                    "candidate_count": len(remaining),
                    "passes_max_null": passes_null,
                    "passes_selection_stability": passes_stability,
                    "accepted": accepted,
                    "decision": decision,
                }
            )
            if not accepted:
                break
            selected.append(winner)

    scores = pd.concat(score_parts, ignore_index=True) if score_parts else pd.DataFrame()
    return pd.DataFrame(audit_rows), scores, pd.DataFrame(null_rows), stop_reason


def parse_args() -> argparse.Namespace:
    """解析 selection-only CMI 运行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "run"), default="plan")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_ROOT / "candidate_union_max200.csv")
    parser.add_argument(
        "--selection-correlation",
        type=Path,
        default=DEFAULT_ROOT / "validation" / "semantic_correlation_selection.csv",
    )
    parser.add_argument(
        "--factor-values",
        type=Path,
        default=DEFAULT_ROOT / "validation" / "selection_factor_values.parquet",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=DEFAULT_ROOT / "data" / "selection_panel.parquet",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--max-factors", type=int, default=4)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-observations", type=int, default=96_000)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--allow-long-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """先冻结 0.98 代表，再按模式输出计划或运行低维 CMI。"""

    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    representatives = build_corr098_representatives(
        args.candidates,
        args.selection_correlation,
        threshold=args.threshold,
    )
    representative_path = args.output / "v2_semantic_representatives_corr098.csv"
    representatives.to_csv(representative_path, index=False, encoding="utf-8-sig")
    workload = estimate_workload(
        len(representatives),
        args.max_factors,
        args.permutations,
        args.workers,
    )
    (args.output / "cmi_workload_plan.json").write_text(
        json.dumps(workload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.mode == "plan":
        print(json.dumps(workload, ensure_ascii=False, indent=2))
        print(f"representatives: {representative_path}")
        return
    if not args.allow_long_run:
        raise RuntimeError("正式 CMI 运行必须显式传入 --allow-long-run，并提交到集群")
    for required in (args.factor_values, args.panel):
        if not required.exists():
            raise FileNotFoundError(f"缺少集群缓存：{required}")

    started = perf_counter()
    audit, scores, null, stop_reason = run_forward_selection(
        representatives,
        args.factor_values,
        args.panel,
        workers=args.workers,
        max_factors=args.max_factors,
        k=args.k,
        min_observations=args.min_observations,
        permutations=args.permutations,
        block_length=args.block_length,
        base_seed=args.seed,
    )
    audit.to_csv(args.output / "cmi_forward_audit.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(args.output / "cmi_candidate_scores.csv", index=False, encoding="utf-8-sig")
    null.to_csv(args.output / "cmi_block_max_null.csv", index=False, encoding="utf-8-sig")
    manifest = {
        **workload,
        "selection_year": 2022,
        "semantic_threshold": args.threshold,
        "representative_count": len(representatives),
        "selected_factors": audit.loc[audit["accepted"].astype(bool), "winner"].tolist()
        if not audit.empty
        else [],
        "stop_reason": stop_reason,
        "k": args.k,
        "block_length_dates": args.block_length,
        "null": "shared_20_session_label_block_max_screen",
        "null_is_exact_conditional_randomization": False,
        "null_interpretation": "selection-only multiplicity screen; not final confirmation",
        "validation_year_opened_by_this_script": False,
        "locked_final_holdout_opened": False,
        "elapsed_seconds": perf_counter() - started,
    }
    (args.output / "cmi_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
