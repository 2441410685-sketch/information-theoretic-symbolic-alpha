"""对 selection-only 语义去重后的候选执行 2023 日期块置换检验。"""

from __future__ import annotations

import argparse
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
from scipy.stats import rankdata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.information.multiple_testing import benjamini_hochberg, bonferroni_min_p  # noqa: E402
from src.information.mutual_information import estimate_mi_ksg_details  # noqa: E402
from src.information.significance import block_permutation_indices, monte_carlo_p_value  # noqa: E402
from src.validation.large_gp_candidates import (  # noqa: E402
    load_candidate_union,
    select_semantic_representatives,
)


DEFAULT_VALIDATION_DIRECTORY = PROJECT_ROOT / "experiments" / "day26" / "validation"
DEFAULT_CANDIDATES = PROJECT_ROOT / "experiments" / "day26" / "candidate_union_max100.csv"
DEFAULT_PANEL = (
    PROJECT_ROOT / "experiments" / "large_scale_gp" / "data" / "validation_panel.parquet"
)
DEFAULT_PERMUTATIONS = 499
BLOCK_LENGTH = 20
SEMANTIC_THRESHOLD = 0.99
BASE_SEED = 20260922

_NULL: dict[str, object] = {}


def build_representatives(
    candidate_path: Path,
    validation_directory: Path,
) -> pd.DataFrame:
    """使用 selection 相关矩阵冻结近重复簇和代表，不查看 validation 排名。"""

    candidates = load_candidate_union(candidate_path)
    correlation = pd.read_csv(
        validation_directory / "semantic_correlation_selection.csv", index_col=0
    )
    return select_semantic_representatives(
        candidates,
        correlation,
        threshold=SEMANTIC_THRESHOLD,
    )


def make_shared_permutations(
    date_count: int,
    n_permutations: int,
    *,
    block_length: int = BLOCK_LENGTH,
    random_state: int = BASE_SEED,
) -> np.ndarray:
    """生成所有候选共享的日期块排列，使跨候选 null 可以直接比较。"""

    rng = np.random.default_rng(random_state)
    return np.vstack(
        [
            block_permutation_indices(date_count, block_length, rng)
            for _ in range(n_permutations)
        ]
    )


def _load_label_matrix(panel_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """读取冻结 500 股样本的 label，并整理为 date×stock-position 矩阵。"""

    frame = pd.read_parquet(
        panel_path,
        columns=["date", "stock", "label_rankgauss", "is_fitness_sample"],
    )
    frame["date"] = pd.to_datetime(frame["date"])
    frame["stock"] = frame["stock"].astype("string")
    sample = (
        frame.loc[frame["is_fitness_sample"].astype(bool), ["date", "stock", "label_rankgauss"]]
        .sort_values(["date", "stock"])
        .reset_index(drop=True)
    )
    counts = sample.groupby("date", sort=True).size()
    if len(counts) != 240 or not counts.eq(500).all():
        raise ValueError("validation null 要求严格的 240 日×500 股冻结样本")
    label = sample["label_rankgauss"].to_numpy(dtype=float).reshape(240, 500)
    if np.any(~np.isfinite(label)):
        raise ValueError("validation label_rankgauss 不得缺失")
    return sample[["date", "stock"]], label


def _initialize_null_worker(
    factor_path: str,
    panel_path: str,
    representative_ids: tuple[str, ...],
    n_permutations: int,
) -> None:
    """每个进程一次性载入候选矩阵、label 和共享日期块排列。"""

    columns = ["date", "stock", *representative_ids]
    factors = pd.read_parquet(factor_path, columns=columns)
    factors["date"] = pd.to_datetime(factors["date"])
    factors["stock"] = factors["stock"].astype("string")
    factors = factors.sort_values(["date", "stock"]).reset_index(drop=True)
    keys, label = _load_label_matrix(Path(panel_path))
    if not factors[["date", "stock"]].equals(keys):
        raise ValueError("因子缓存与 validation label 的冻结键不一致")
    _NULL.update(
        {
            "factors": factors,
            "label": label,
            "permutations": make_shared_permutations(240, n_permutations),
        }
    )


def _mean_daily_spearman(x_matrix: np.ndarray, y_matrix: np.ndarray) -> float:
    """计算日期等权的 Spearman RankIC，允许候选在少量股票上缺失。"""

    values: list[float] = []
    for x_row, y_row in zip(x_matrix, y_matrix):
        finite = np.isfinite(x_row) & np.isfinite(y_row)
        if finite.sum() < 50:
            continue
        x_rank = rankdata(x_row[finite], method="average")
        y_rank = rankdata(y_row[finite], method="average")
        correlation = float(np.corrcoef(x_rank, y_rank)[0, 1])
        if np.isfinite(correlation):
            values.append(correlation)
    return float(np.mean(values)) if values else np.nan


def _null_worker(candidate: dict[str, object]) -> dict[str, object]:
    """为一个语义代表计算共享 block-permutation 下的双 endpoint null。"""

    candidate_id = str(candidate["candidate_id"])
    x = _NULL["factors"][candidate_id].to_numpy(dtype=float).reshape(240, 500)
    y = _NULL["label"]
    finite = np.isfinite(x) & np.isfinite(y)
    observed_rankic = abs(_mean_daily_spearman(x, y))
    fingerprint = str(candidate["structural_hash"])
    seed = int((BASE_SEED + int(fingerprint[:8], 16)) % (2**32 - 1))
    observed_mi = estimate_mi_ksg_details(
        x[finite],
        y[finite],
        k=5,
        tie_strategy="jitter",
        random_state=seed,
    ).estimate
    permutations = _NULL["permutations"]
    rankic_null = np.empty(len(permutations), dtype=float)
    mi_null = np.empty(len(permutations), dtype=float)
    for repetition, date_indices in enumerate(permutations):
        permuted_y = y[date_indices]
        permuted_finite = np.isfinite(x) & np.isfinite(permuted_y)
        rankic_null[repetition] = abs(_mean_daily_spearman(x, permuted_y))
        mi_null[repetition] = estimate_mi_ksg_details(
            x[permuted_finite],
            permuted_y[permuted_finite],
            k=5,
            tie_strategy="jitter",
            random_state=seed + repetition + 1,
        ).estimate
    p_rankic, _ = monte_carlo_p_value(observed_rankic, rankic_null, "greater")
    p_mi, _ = monte_carlo_p_value(observed_mi, mi_null, "greater")
    return {
        "candidate_id": candidate_id,
        "observed_abs_mean_rankic": observed_rankic,
        "observed_mi": float(observed_mi),
        "p_rankic": p_rankic,
        "p_mi": p_mi,
        "rankic_null": rankic_null,
        "mi_null": mi_null,
    }


def run_null(
    representatives: pd.DataFrame,
    validation_directory: Path,
    panel_path: Path,
    *,
    n_permutations: int,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """并行运行语义代表的双 endpoint block null。"""

    ids = tuple(representatives["candidate_id"].astype(str))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_null_worker,
        initargs=(
            str((validation_directory / "validation_factor_values.parquet").resolve()),
            str(panel_path.resolve()),
            ids,
            n_permutations,
        ),
    ) as executor:
        results = list(executor.map(_null_worker, representatives.to_dict("records"), chunksize=1))
    summary_rows, null_rows = [], []
    for result in results:
        summary_rows.append({key: value for key, value in result.items() if not key.endswith("_null")})
        for repetition, (rankic_value, mi_value) in enumerate(
            zip(result["rankic_null"], result["mi_null"]), start=1
        ):
            null_rows.append(
                {
                    "candidate_id": result["candidate_id"],
                    "repetition": repetition,
                    "null_abs_mean_rankic": rankic_value,
                    "null_mi": mi_value,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(null_rows)


def apply_multiple_testing(
    representatives: pd.DataFrame,
    summary: pd.DataFrame,
    null_distribution: pd.DataFrame,
) -> pd.DataFrame:
    """分别对 RankIC、MI 和并集代表执行 BH，并报告 shared max-null FWER。"""

    base = representatives.merge(summary, on="candidate_id", validate="one_to_one")
    base["p_omnibus"] = [
        bonferroni_min_p(rankic, mi)
        for rankic, mi in zip(base["p_rankic"], base["p_mi"])
    ]
    rows: list[pd.DataFrame] = []
    family_masks = {
        "rankic_only": base["source_rankic"].astype(bool),
        "mi_only": base["source_mi"].astype(bool),
        "union": pd.Series(True, index=base.index),
    }
    for family, mask in family_masks.items():
        members = base.loc[mask].copy().reset_index(drop=True)
        bh = benjamini_hochberg(members["p_omnibus"].to_numpy(), alpha=0.10)
        members["bh_q_value_10"] = bh.adjusted_p_values
        members["bh_reject_10"] = bh.rejected
        member_ids = members["candidate_id"].astype(str).tolist()
        family_null = null_distribution.loc[
            null_distribution["candidate_id"].astype(str).isin(member_ids)
        ]
        max_rankic = family_null.pivot(
            index="repetition", columns="candidate_id", values="null_abs_mean_rankic"
        ).max(axis=1).to_numpy()
        max_mi = family_null.pivot(
            index="repetition", columns="candidate_id", values="null_mi"
        ).max(axis=1).to_numpy()
        members["fwer_p_rankic"] = [
            monte_carlo_p_value(value, max_rankic, "greater")[0]
            for value in members["observed_abs_mean_rankic"]
        ]
        members["fwer_p_mi"] = [
            monte_carlo_p_value(value, max_mi, "greater")[0]
            for value in members["observed_mi"]
        ]
        members["fwer_p_omnibus"] = [
            bonferroni_min_p(rankic, mi)
            for rankic, mi in zip(members["fwer_p_rankic"], members["fwer_p_mi"])
        ]
        members["family"] = family
        members["effective_family_size"] = len(members)
        rows.append(members)
    return pd.concat(rows, ignore_index=True)


def estimate_from_metrics(
    validation_directory: Path,
    representative_count: int,
    n_permutations: int,
    workers: int,
) -> dict[str, float]:
    """用已完成验证的五次 KSG 总耗时保守估计 block-null wall time。"""

    metrics = pd.read_csv(validation_directory / "candidate_cross_validation_metrics.csv")
    seconds_per_full_mi = float(metrics["validation_evaluation_seconds_total"].median() / 5.0)
    core_hours = representative_count * n_permutations * seconds_per_full_mi / 3600.0
    wall_hours = core_hours / max(1, workers) * 1.35
    return {
        "representative_count": float(representative_count),
        "permutations": float(n_permutations),
        "seconds_per_full_mi_estimate": seconds_per_full_mi,
        "estimated_core_hours": core_hours,
        "estimated_wall_hours": wall_hours,
        "workers": float(workers),
    }


def main() -> None:
    """先 plan/pilot；只有显式 run 才执行正式 499 次 null。"""

    parser = argparse.ArgumentParser(description="Large GP validation block null")
    parser.add_argument("--mode", choices=("plan", "pilot", "run"), default="plan")
    parser.add_argument("--validation-directory", type=Path, default=DEFAULT_VALIDATION_DIRECTORY)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--allow-long-run",
        action="store_true",
        help="仅供已评估资源并设置 walltime 的集群任务使用",
    )
    arguments = parser.parse_args()
    if arguments.permutations < 19 or arguments.workers < 1:
        raise ValueError("permutations 至少 19，workers 必须为正整数")
    representatives = build_representatives(arguments.candidates, arguments.validation_directory)
    representatives.to_csv(
        arguments.validation_directory / "semantic_representatives_corr099.csv", index=False
    )
    estimate = estimate_from_metrics(
        arguments.validation_directory,
        len(representatives),
        arguments.permutations,
        arguments.workers,
    )
    print(json.dumps(estimate, ensure_ascii=False, indent=2))
    if arguments.mode == "plan":
        print("PLAN ONLY: 未运行 block-permutation null。")
        return
    if arguments.mode == "pilot":
        pilot_representatives = representatives.head(2).copy()
        pilot_permutations = 19
        started = perf_counter()
        run_null(
            pilot_representatives,
            arguments.validation_directory,
            arguments.panel,
            n_permutations=pilot_permutations,
            workers=min(arguments.workers, 2),
        )
        elapsed = perf_counter() - started
        waves = np.ceil(len(representatives) / arguments.workers)
        pilot_waves = np.ceil(len(pilot_representatives) / min(arguments.workers, 2))
        projected = elapsed * (arguments.permutations / pilot_permutations) * (waves / pilot_waves)
        payload = {
            **estimate,
            "pilot_candidates": len(pilot_representatives),
            "pilot_permutations": pilot_permutations,
            "pilot_elapsed_seconds": elapsed,
            "pilot_projected_wall_hours": projected / 3600.0,
        }
        (arguments.validation_directory / "null_runtime_estimate.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if estimate["estimated_wall_hours"] > 8.0 and not arguments.allow_long_run:
        raise RuntimeError("预计 block null 超过 8 小时，请转集群执行")
    started = perf_counter()
    summary, null_distribution = run_null(
        representatives,
        arguments.validation_directory,
        arguments.panel,
        n_permutations=arguments.permutations,
        workers=arguments.workers,
    )
    decisions = apply_multiple_testing(representatives, summary, null_distribution)
    summary.to_csv(arguments.validation_directory / "representative_null_summary.csv", index=False)
    null_distribution.to_parquet(
        arguments.validation_directory / "representative_null_distribution.parquet",
        index=False,
        compression="zstd",
    )
    decisions.to_csv(arguments.validation_directory / "representative_family_decisions.csv", index=False)
    manifest_path = arguments.validation_directory / "validation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "semantic_dedup_threshold": SEMANTIC_THRESHOLD,
            "semantic_representative_count": len(representatives),
            "null_permutations": arguments.permutations,
            "null_block_length": BLOCK_LENGTH,
            "null_elapsed_seconds": perf_counter() - started,
            "null_distribution_completed": True,
            "inference_scope": "2023 development screen; not locked final confirmation",
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"validation null completed: {arguments.validation_directory}")


if __name__ == "__main__":
    main()
