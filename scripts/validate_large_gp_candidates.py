"""对冻结的 100 个大规模 GP 候选做 2022/2023 双指标交叉验证。

本脚本只读取 2022 selection 与 2023 development validation；不会读取
2024–2025 Final Holdout。默认 ``--mode plan`` 只给出资源估计。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.symbolic.expression import (  # noqa: E402
    EvaluationContext,
    ExpressionFactory,
    evaluate_expression,
    make_default_registry,
    parse_canonical_expression,
    structural_hash,
)
from src.symbolic.large_scale import (  # noqa: E402
    load_search_panel,
    preprocess_candidate_by_date,
)
from src.symbolic.operators import OutputPolicy, make_operator_executors  # noqa: E402
from src.validation.large_gp_candidates import (  # noqa: E402
    evaluate_processed_candidate,
    load_candidate_union,
    mean_daily_factor_correlation,
    validate_development_panel,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "large_scale_gp.json"
DEFAULT_CANDIDATES = PROJECT_ROOT / "experiments" / "day26" / "candidate_union_max100.csv"
DEFAULT_PANEL_DIRECTORY = PROJECT_ROOT / "experiments" / "large_scale_gp" / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "day26" / "validation"

_WORKER: dict[str, object] = {}


def _sha256(path: Path) -> str:
    """计算输入文件哈希，确保候选集合和配置没有在验证时被偷换。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate_runtime(candidate_path: Path, day26_directory: Path, workers: int) -> dict[str, float]:
    """用正式搜索已记录的单候选耗时估计交叉验证成本。"""

    candidates = load_candidate_union(candidate_path)
    recorded: list[float] = []
    for channel in ("rankic", "mi"):
        paths = sorted(day26_directory.glob(f"{channel}_shortlist_top*.csv"))
        if paths:
            path = paths[-1]
            table = pd.read_csv(path)
            recorded.extend(pd.to_numeric(table["evaluation_seconds"], errors="coerce").dropna())
    mean_seconds = float(np.mean(recorded)) if recorded else 30.0
    # 每个 period 除原搜索指标外还要补算另一指标、四块稳定性及分组收益。
    multiplier = 2.2
    total_core_hours = len(candidates) * 2 * mean_seconds * multiplier / 3600.0
    wall_hours = total_core_hours / max(1, workers) * 1.25
    return {
        "candidate_count": float(len(candidates)),
        "recorded_mean_seconds": mean_seconds,
        "estimated_core_hours": total_core_hours,
        "estimated_wall_hours": wall_hours,
        "workers": float(workers),
    }


def _initialize_worker(panel_path: str, period: str, config_path: str) -> None:
    """每个进程只加载一次年度面板，避免逐候选重复读取 parquet。"""

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    frame = load_search_panel(panel_path)
    year = int(config["research_split"][f"{period}_year"])
    sessions = int(config["research_split"][f"{period}_sessions"])
    validate_development_panel(
        frame,
        expected_year=year,
        expected_score_dates=sessions,
        expected_stocks_per_date=int(config["sampling"]["stocks_per_date"]),
    )
    _WORKER.update(
        {
            "frame": frame,
            "context": EvaluationContext(frame),
            "registry": make_default_registry(),
            "executors": make_operator_executors(),
            "sample_mask": frame["is_fitness_sample"].to_numpy(dtype=bool),
            "config": config,
            "period": period,
        }
    )


def _evaluate_worker(candidate: dict[str, object]) -> dict[str, object]:
    """进程任务：表达式只执行和预处理一次，再同时计算所有验证指标。"""

    frame = _WORKER["frame"]
    registry = _WORKER["registry"]
    expression = parse_canonical_expression(
        str(candidate["canonical"]), ExpressionFactory(registry)
    )
    if structural_hash(expression, registry) != str(candidate["structural_hash"]):
        raise ValueError(f"{candidate['candidate_id']} 的结构哈希与表达式不一致")
    raw = evaluate_expression(
        expression,
        _WORKER["context"],
        registry,
        _WORKER["executors"],
    )
    processed = preprocess_candidate_by_date(
        raw,
        frame,
        mad_multiple=float(_WORKER["config"]["preprocessing"]["mad_multiple"]),
        minimum_cross_section=int(
            _WORKER["config"]["preprocessing"]["minimum_cross_section"]
        ),
    )
    mask = _WORKER["sample_mask"]
    sample = frame.loc[mask].copy()
    factor = processed.loc[mask]
    hard = _WORKER["config"]["hard_constraints"]
    policy = OutputPolicy(
        minimum_valid_fraction=float(hard["minimum_valid_fraction"]),
        minimum_valid_observations=int(hard["minimum_valid_observations"]),
        minimum_unique_values=int(hard["minimum_unique_values"]),
    )
    metrics, daily, blocks = evaluate_processed_candidate(
        expression,
        str(candidate["candidate_id"]),
        factor,
        sample,
        k=int(_WORKER["config"]["search"]["ksg_k"]),
        minimum_dates=int(hard["minimum_valid_dates"]),
        minimum_cross_section=int(
            _WORKER["config"]["preprocessing"]["minimum_cross_section"]
        ),
        block_count=4,
        output_policy=policy,
        base_random_seed=int(_WORKER["config"]["search"]["base_random_seed"]),
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "metrics": metrics,
        "daily": daily,
        "blocks": blocks,
        "factor": factor.to_numpy(dtype=np.float64),
    }


def _sample_keys(panel_path: Path) -> pd.DataFrame:
    """只读取结果缓存需要的冻结样本键，避免主进程常驻完整面板。"""

    keys = pd.read_parquet(
        panel_path,
        columns=["date", "stock", "is_fitness_sample"],
    )
    keys["date"] = pd.to_datetime(keys["date"])
    keys["stock"] = keys["stock"].astype("string")
    return (
        keys.loc[keys["is_fitness_sample"].astype(bool), ["date", "stock"]]
        .sort_values(["date", "stock"])
        .reset_index(drop=True)
    )


def run_period(
    candidates: pd.DataFrame,
    panel_path: Path,
    config_path: Path,
    *,
    period: str,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """并行评价一个 period，并返回摘要、日度、时间块和因子值。"""

    candidate_records = candidates.to_dict(orient="records")
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(str(panel_path.resolve()), period, str(config_path.resolve())),
    ) as executor:
        results = list(executor.map(_evaluate_worker, candidate_records, chunksize=1))

    keys = _sample_keys(panel_path)
    factor_columns: dict[str, np.ndarray] = {}
    metrics, daily, blocks = [], [], []
    for result in results:
        candidate_id = str(result["candidate_id"])
        values = np.asarray(result["factor"], dtype=np.float64)
        if len(values) != len(keys):
            raise RuntimeError(f"{candidate_id} 的因子行数与冻结样本键不一致")
        factor_columns[candidate_id] = values
        metrics.append(result["metrics"])
        daily.append(result["daily"])
        blocks.append(result["blocks"])
    factor_values = pd.concat(
        [keys, pd.DataFrame(factor_columns, index=keys.index)], axis=1
    )
    metric_frame = pd.DataFrame(metrics).sort_values("candidate_id").reset_index(drop=True)
    daily_frame = pd.concat(daily, ignore_index=True)
    block_frame = pd.concat(blocks, ignore_index=True)
    metric_frame.insert(1, "period", period)
    daily_frame.insert(1, "period", period)
    block_frame.insert(1, "period", period)
    return metric_frame, daily_frame, block_frame, factor_values


def _prefix_period(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    """给年度指标加前缀，避免 selection 与 validation 列名混淆。"""

    result = frame.drop(columns="period").copy()
    return result.rename(
        columns={
            column: f"{period}_{column}"
            for column in result.columns
            if column != "candidate_id"
        }
    )


def save_semantic_correlation(
    factor_values: pd.DataFrame,
    candidate_ids: list[str],
    path: Path,
) -> None:
    """保存按日横截面相关的时间平均，用于发现结构不同但输出近重复的因子。"""

    correlation = mean_daily_factor_correlation(
        factor_values[candidate_ids].to_numpy(dtype=float),
        factor_values["date"],
        minimum_pair_observations=50,
    )
    output = pd.DataFrame(correlation, index=candidate_ids, columns=candidate_ids)
    output.index.name = "candidate_id"
    output.to_csv(path)


def run_metrics(
    config_path: Path,
    candidate_path: Path,
    panel_directory: Path,
    output: Path,
    *,
    workers: int,
) -> None:
    """运行冻结候选的两期交叉验证，并保存后续 null/CMI 所需缓存。"""

    started = perf_counter()
    candidates = load_candidate_union(candidate_path)
    panel_paths = {
        "selection": panel_directory / "selection_panel.parquet",
        "validation": panel_directory / "validation_panel.parquet",
    }
    missing = [str(path) for path in panel_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少正式 development panel：{missing}")
    output.mkdir(parents=True, exist_ok=True)

    period_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for period, panel_path in panel_paths.items():
        period_outputs[period] = run_period(
            candidates,
            panel_path,
            config_path,
            period=period,
            workers=workers,
        )

    selection_metrics, selection_daily, selection_blocks, selection_values = period_outputs[
        "selection"
    ]
    validation_metrics, validation_daily, validation_blocks, validation_values = period_outputs[
        "validation"
    ]
    combined = candidates.merge(
        _prefix_period(selection_metrics, "selection"), on="candidate_id", validate="one_to_one"
    ).merge(
        _prefix_period(validation_metrics, "validation"), on="candidate_id", validate="one_to_one"
    )
    selection_rankic = combined["selection_rankic_mean_rank_ic"].to_numpy(dtype=float)
    direction = np.where(np.isfinite(selection_rankic), np.sign(selection_rankic), 0.0)
    combined["selection_direction"] = direction.astype(int)
    combined["validation_directional_rank_ic"] = (
        direction * combined["validation_rankic_mean_rank_ic"].to_numpy(dtype=float)
    )
    combined["rankic_sign_preserved"] = combined["validation_directional_rank_ic"] > 0.0

    blocks = pd.concat([selection_blocks, validation_blocks], ignore_index=True)
    validation_block = blocks.loc[blocks["period"].eq("validation")].merge(
        combined[["candidate_id", "selection_direction"]], on="candidate_id", how="left"
    )
    block_summary = validation_block.groupby("candidate_id", sort=False).agg(
        validation_directional_rankic_block_ratio=(
            "mean_rank_ic",
            lambda x: float(
                np.mean(
                    x.to_numpy(dtype=float)
                    * validation_block.loc[x.index, "selection_direction"].to_numpy(dtype=float)
                    > 0.0
                )
            ),
        ),
        validation_positive_mi_block_ratio=("mi_estimate", lambda x: float(np.mean(x > 0.0))),
        validation_min_block_mi=("mi_estimate", "min"),
    ).reset_index()
    combined = combined.merge(block_summary, on="candidate_id", validate="one_to_one")

    combined.to_csv(output / "candidate_cross_validation_metrics.csv", index=False)
    pd.concat([selection_daily, validation_daily], ignore_index=True).to_csv(
        output / "candidate_daily_metrics.csv", index=False
    )
    blocks.to_csv(output / "candidate_block_stability.csv", index=False)
    selection_values.to_parquet(
        output / "selection_factor_values.parquet", index=False, compression="zstd"
    )
    validation_values.to_parquet(
        output / "validation_factor_values.parquet", index=False, compression="zstd"
    )
    candidate_ids = candidates["candidate_id"].astype(str).tolist()
    save_semantic_correlation(
        selection_values,
        candidate_ids,
        output / "semantic_correlation_selection.csv",
    )
    save_semantic_correlation(
        validation_values,
        candidate_ids,
        output / "semantic_correlation_validation.csv",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split = config["research_split"]
    manifest = {
        "candidate_count": len(candidates),
        "selection_year": int(split["selection_year"]),
        "validation_year": int(split["validation_year"]),
        "score_dates_per_period": int(split["selection_sessions"]),
        "stocks_per_date": int(config["sampling"]["stocks_per_date"]),
        "ksg_k": int(config["search"]["ksg_k"]),
        "block_count": 4,
        "workers": workers,
        "candidate_union_sha256": _sha256(candidate_path),
        "config_sha256": _sha256(config_path),
        "elapsed_seconds": perf_counter() - started,
        "final_holdout_opened": False,
        "null_distribution_completed": False,
    }
    (output / "validation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    """先估算资源；只有显式 ``--mode run`` 才执行候选验证。"""

    parser = argparse.ArgumentParser(description="Validate frozen large-scale GP candidates")
    parser.add_argument("--mode", choices=("plan", "run"), default="plan")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--panel-directory", type=Path, default=DEFAULT_PANEL_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    arguments = parser.parse_args()
    if arguments.workers < 1:
        raise ValueError("workers 必须为正整数")
    estimate = estimate_runtime(arguments.candidates, arguments.candidates.parent, arguments.workers)
    print(json.dumps(estimate, ensure_ascii=False, indent=2))
    if arguments.mode == "plan":
        print("PLAN ONLY: 未读取 development panel，未运行候选验证。")
        return
    if estimate["estimated_wall_hours"] > 8.0:
        raise RuntimeError("预计运行超过 8 小时，请转到集群执行")
    run_metrics(
        arguments.config,
        arguments.candidates,
        arguments.panel_directory,
        arguments.output,
        workers=arguments.workers,
    )
    print(f"validation completed: {arguments.output}")


if __name__ == "__main__":
    main()
