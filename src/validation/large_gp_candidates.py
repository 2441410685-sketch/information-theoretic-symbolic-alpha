"""冻结 GP 候选在 development periods 上的统一评价工具。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.information.mutual_information import estimate_mi_ksg_details
from src.symbolic.expression import Expression
from src.symbolic.fitness import (
    daily_rank_ic,
    evaluate_pure_mi_fitness,
    evaluate_rank_ic_fitness,
)
from src.symbolic.large_scale import candidate_random_state
from src.symbolic.operators import OutputPolicy
from src.validation.factor_evaluation import (
    build_panel_index,
    evaluate_factor_matrix,
    summarize_daily_evaluation,
)


REQUIRED_UNION_COLUMNS = {
    "candidate_id",
    "structural_hash",
    "canonical",
    "source_rankic",
    "source_mi",
}


def load_candidate_union(path: str | Path) -> pd.DataFrame:
    """读取并检查冻结候选并集，不根据 validation 重新改写候选身份。"""

    frame = pd.read_csv(path)
    missing = REQUIRED_UNION_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"候选并集缺少字段：{sorted(missing)}")
    if frame.empty:
        raise ValueError("候选并集不能为空")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("candidate_id 必须唯一")
    if frame["structural_hash"].duplicated().any():
        raise ValueError("structural_hash 必须唯一")
    if frame["canonical"].isna().any() or frame["canonical"].astype(str).str.strip().eq("").any():
        raise ValueError("canonical expression 不能为空")
    for column in ("source_rankic", "source_mi"):
        frame[column] = frame[column].astype(bool)
    if (~(frame["source_rankic"] | frame["source_mi"])).any():
        raise ValueError("每个候选至少必须来自一个搜索通道")
    return frame.sort_values("candidate_id").reset_index(drop=True)


def validate_development_panel(
    frame: pd.DataFrame,
    *,
    expected_year: int,
    expected_score_dates: int,
    expected_stocks_per_date: int,
) -> None:
    """确认面板只含指定 development 年份，且冻结抽样键完整。"""

    required = {
        "date",
        "stock",
        "label_1d",
        "label_rankgauss",
        "is_score_date",
        "is_fitness_sample",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"development panel 缺少字段：{sorted(missing)}")
    if frame.duplicated(["date", "stock"]).any():
        raise ValueError("development panel 的 (date, stock) 必须唯一")
    score = frame.loc[frame["is_score_date"].astype(bool)]
    dates = pd.DatetimeIndex(score["date"].drop_duplicates().sort_values())
    if len(dates) != expected_score_dates:
        raise ValueError(f"计分日期应为 {expected_score_dates} 日，实际为 {len(dates)} 日")
    if not np.all(dates.year == expected_year):
        raise ValueError(f"计分日期必须全部属于 {expected_year} 年")
    sample = frame.loc[frame["is_fitness_sample"].astype(bool)]
    counts = sample.groupby("date", sort=True).size()
    if len(counts) != expected_score_dates or not counts.eq(expected_stocks_per_date).all():
        raise ValueError("冻结 fitness 样本的每日股票数不符合配置")
    if pd.to_datetime(frame["date"]).dt.year.max() >= 2024:
        raise ValueError("development panel 不得读取 2024–2025 Final Holdout")


def _chronological_blocks(
    factor: pd.Series,
    sample_frame: pd.DataFrame,
    *,
    block_count: int,
    k: int,
    random_state: int,
) -> pd.DataFrame:
    """在连续时间块上分别计算 RankIC 与 KSG MI，用于稳定性诊断。"""

    dates = pd.DatetimeIndex(sample_frame["date"].drop_duplicates().sort_values())
    rows: list[dict[str, object]] = []
    for number, block_dates in enumerate(np.array_split(dates, block_count), start=1):
        mask = sample_frame["date"].isin(block_dates)
        block = sample_frame.loc[mask]
        block_factor = factor.loc[mask]
        daily = daily_rank_ic(
            block_factor,
            block,
            label_column="label_rankgauss",
            minimum_cross_section=50,
        )
        finite = np.isfinite(block_factor.to_numpy(dtype=float)) & np.isfinite(
            block["label_rankgauss"].to_numpy(dtype=float)
        )
        mi = estimate_mi_ksg_details(
            block_factor.to_numpy(dtype=float)[finite],
            block["label_rankgauss"].to_numpy(dtype=float)[finite],
            k=k,
            tie_strategy="jitter",
            random_state=random_state + number * 10_007,
        ).estimate
        rows.append(
            {
                "block": number,
                "start_date": pd.Timestamp(block_dates.min()),
                "end_date": pd.Timestamp(block_dates.max()),
                "date_count": len(block_dates),
                "observation_count": int(finite.sum()),
                "mean_rank_ic": float(daily.loc[daily["valid"], "rank_ic"].mean()),
                "mi_estimate": float(mi),
            }
        )
    return pd.DataFrame(rows)


def evaluate_processed_candidate(
    expression: Expression,
    candidate_id: str,
    processed_factor: pd.Series,
    sample_frame: pd.DataFrame,
    *,
    k: int = 5,
    minimum_dates: int = 180,
    minimum_cross_section: int = 50,
    block_count: int = 4,
    output_policy: OutputPolicy | None = None,
    base_random_seed: int = 20260921,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """在同一预处理结果上同时计算 RankIC、MI、分组收益和稳定性。"""

    if len(processed_factor) != len(sample_frame) or not processed_factor.index.equals(
        sample_frame.index
    ):
        raise ValueError("processed_factor 必须与 sample_frame 严格对齐")
    policy = output_policy or OutputPolicy(
        minimum_valid_fraction=0.70,
        minimum_valid_observations=72_000,
        minimum_unique_values=2,
    )
    started = perf_counter()
    seed = candidate_random_state(expression, base_random_seed)
    rankic, _ = evaluate_rank_ic_fitness(
        processed_factor,
        sample_frame,
        label_column="label_rankgauss",
        minimum_cross_section=minimum_cross_section,
        minimum_dates=minimum_dates,
        output_policy=policy,
    )
    mi = evaluate_pure_mi_fitness(
        processed_factor,
        sample_frame,
        label_column="label_rankgauss",
        k=k,
        minimum_dates=minimum_dates,
        output_policy=policy,
        random_state=seed,
    )

    # 分组收益使用未经 Rank-Gaussian 化的真实 Open-to-Open label；因子仍是
    # 完整截面完成中性化和 Rank-Gaussianization 后的正式表示。
    panel_index = build_panel_index(
        sample_frame["date"], sample_frame["stock"], sample_frame["label_1d"]
    )
    daily, decay = evaluate_factor_matrix(
        processed_factor.to_numpy(dtype=float),
        [f"{candidate_id}__neutral"],
        panel_index,
        min_obs=minimum_cross_section,
        quantiles=10,
        decay_horizons=(1,),
    )
    summary = summarize_daily_evaluation(daily, decay, quantiles=10).iloc[0].to_dict()
    blocks = _chronological_blocks(
        processed_factor,
        sample_frame,
        block_count=block_count,
        k=k,
        random_state=seed,
    )
    rankic_record = {f"rankic_{key}": value for key, value in asdict(rankic).items()}
    mi_record = {f"mi_{key}": value for key, value in asdict(mi).items()}
    metrics: dict[str, object] = {
        "candidate_id": candidate_id,
        **rankic_record,
        **mi_record,
        "mean_pearson_ic": summary["mean_pearson_ic"],
        "mean_long_short_return": summary["mean_long_short_return"],
        "mean_top_decile_turnover": summary["mean_top_decile_turnover"],
        "quantile_monotonicity": summary["quantile_monotonicity"],
        "nw_se_rank_ic": summary["nw_se_rank_ic"],
        "nw_t_rank_ic": summary["nw_t_rank_ic"],
        "evaluation_seconds_total": perf_counter() - started,
    }
    for quantile in range(1, 11):
        metrics[f"q{quantile}_mean_return"] = summary[f"q{quantile}_mean_return"]
    daily.insert(0, "candidate_id", candidate_id)
    blocks.insert(0, "candidate_id", candidate_id)
    return metrics, daily, blocks


def mean_daily_factor_correlation(
    values: np.ndarray,
    dates: pd.Series | np.ndarray,
    *,
    minimum_pair_observations: int = 50,
) -> np.ndarray:
    """计算日度横截面因子相关矩阵的等权时间平均。"""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("values 必须是二维矩阵")
    date_values = pd.to_datetime(np.asarray(dates))
    if len(date_values) != len(matrix):
        raise ValueError("dates 与 values 行数必须一致")
    if minimum_pair_observations < 3:
        raise ValueError("minimum_pair_observations 至少为 3")

    total = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)
    count = np.zeros_like(total, dtype=np.int32)
    for date in pd.unique(date_values):
        block = matrix[date_values == date]
        correlation = pd.DataFrame(block).corr(
            method="pearson", min_periods=minimum_pair_observations
        ).to_numpy(dtype=float)
        finite = np.isfinite(correlation)
        total[finite] += correlation[finite]
        count[finite] += 1
    result = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    np.fill_diagonal(result, 1.0)
    return result


def correlation_components(
    correlation: pd.DataFrame,
    *,
    threshold: float = 0.99,
) -> list[list[str]]:
    """按绝对相关阈值构造连通分量，只用于近重复语义去重。"""

    if correlation.empty or correlation.shape[0] != correlation.shape[1]:
        raise ValueError("correlation 必须是非空方阵")
    if list(correlation.index.astype(str)) != list(correlation.columns.astype(str)):
        raise ValueError("correlation 的行列候选顺序必须一致")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold 必须位于 (0,1]")
    values = correlation.to_numpy(dtype=float)
    if not np.allclose(values, values.T, equal_nan=True, atol=1e-10):
        raise ValueError("correlation 必须对称")

    names = correlation.index.astype(str).tolist()
    parent = list(range(len(names)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            if np.isfinite(values[left, right]) and abs(values[left, right]) >= threshold:
                union(left, right)
    groups: dict[int, list[str]] = {}
    for index, name in enumerate(names):
        groups.setdefault(find(index), []).append(name)
    return sorted(groups.values(), key=lambda members: (-len(members), members[0]))


def select_semantic_representatives(
    candidates: pd.DataFrame,
    correlation: pd.DataFrame,
    *,
    threshold: float = 0.99,
) -> pd.DataFrame:
    """仅根据 selection 相关结构与搜索排名选择每个近重复簇的代表。"""

    required = REQUIRED_UNION_COLUMNS | {"rankic_search_rank", "mi_search_rank"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"候选表缺少字段：{sorted(missing)}")
    ids = candidates["candidate_id"].astype(str).tolist()
    if set(ids) != set(correlation.index.astype(str)):
        raise ValueError("候选表与相关矩阵包含的 candidate_id 不一致")
    ordered_correlation = correlation.loc[ids, ids]
    components = correlation_components(ordered_correlation, threshold=threshold)
    rows: list[dict[str, object]] = []
    for cluster_id, members in enumerate(components, start=1):
        subset = candidates.loc[candidates["candidate_id"].astype(str).isin(members)].copy()
        subset["source_best_rank"] = subset[["rankic_search_rank", "mi_search_rank"]].min(
            axis=1, skipna=True
        )
        representative = subset.sort_values(
            ["source_best_rank", "candidate_id"], ascending=[True, True]
        ).iloc[0]
        record = representative.to_dict()
        record.update(
            {
                "semantic_cluster": cluster_id,
                "semantic_cluster_size": len(members),
                "semantic_cluster_members": "|".join(sorted(members)),
                "semantic_threshold": threshold,
                "cluster_rankic_count": int(subset["source_rankic"].astype(bool).sum()),
                "cluster_mi_count": int(subset["source_mi"].astype(bool).sum()),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values("semantic_cluster").reset_index(drop=True)
