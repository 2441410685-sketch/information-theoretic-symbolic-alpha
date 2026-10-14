"""Day 8：日度横截面 IC、分组收益、换手率与信号衰减。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata


@dataclass(frozen=True)
class PanelIndex:
    """保存按日期排序 panel 的轻量索引，避免重复处理大表主键。"""

    dates: np.ndarray
    stock_ids: np.ndarray
    labels: np.ndarray
    unique_dates: np.ndarray
    boundaries: np.ndarray


def build_panel_index(
    dates: np.ndarray | pd.Series,
    stocks: np.ndarray | pd.Series,
    labels: np.ndarray | pd.Series,
) -> PanelIndex:
    """构造日期切片与整数股票代码；输入必须已按 date-stock 排序。"""

    date_array = pd.to_datetime(np.asarray(dates)).to_numpy(dtype="datetime64[ns]")
    stock_series = pd.Series(stocks, dtype="string")
    label_array = np.asarray(labels, dtype=np.float64)
    if not (len(date_array) == len(stock_series) == len(label_array)):
        raise ValueError("date、stock 与 label 长度必须一致")
    if len(date_array) == 0:
        raise ValueError("panel 不能为空")
    if np.any(date_array[1:] < date_array[:-1]):
        raise ValueError("panel 必须按日期升序排列")

    # 只保留整数代码可显著降低千万行主键的内存占用；同一股票始终映射到同一整数。
    stock_ids, _ = pd.factorize(stock_series, sort=True)
    boundaries = np.r_[
        0,
        np.flatnonzero(date_array[1:] != date_array[:-1]) + 1,
        len(date_array),
    ].astype(np.int64)
    unique_dates = date_array[boundaries[:-1]]
    return PanelIndex(
        dates=date_array,
        stock_ids=stock_ids.astype(np.int32, copy=False),
        labels=np.where(np.isfinite(label_array), label_array, np.nan),
        unique_dates=unique_dates,
        boundaries=boundaries,
    )


def columnwise_pearson(
    values: np.ndarray,
    target: np.ndarray,
    min_obs: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """逐列计算有限样本 Pearson 相关，并返回每列配对样本量。"""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    y = np.asarray(target, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) != len(y):
        raise ValueError("values 必须是二维矩阵，且与 target 行数一致")

    valid = np.isfinite(matrix) & np.isfinite(y)[:, None]
    count = valid.sum(axis=0).astype(np.int64)
    safe_count = np.maximum(count, 1)
    x0 = np.where(valid, matrix, 0.0)
    y0 = np.where(valid, y[:, None], 0.0)
    sum_x = x0.sum(axis=0)
    sum_y = y0.sum(axis=0)
    covariance = (x0 * y0).sum(axis=0) - sum_x * sum_y / safe_count
    variance_x = (x0 * x0).sum(axis=0) - sum_x * sum_x / safe_count
    variance_y = (y0 * y0).sum(axis=0) - sum_y * sum_y / safe_count
    denominator = np.sqrt(np.maximum(variance_x, 0.0) * np.maximum(variance_y, 0.0))
    correlation = np.divide(
        covariance,
        denominator,
        out=np.full(matrix.shape[1], np.nan),
        where=(count >= min_obs) & (denominator > 1e-15),
    )
    return correlation, count


def columnwise_spearman(
    values: np.ndarray,
    target: np.ndarray,
    min_obs: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """逐列取平均秩后计算 Pearson 相关，即日度 Spearman RankIC。"""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    y = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(matrix) & np.isfinite(y)[:, None]
    ranked_x = rankdata(
        np.where(valid, matrix, np.nan),
        axis=0,
        method="average",
        nan_policy="omit",
    )
    ranked_y = rankdata(
        np.where(valid, y[:, None], np.nan),
        axis=0,
        method="average",
        nan_policy="omit",
    )
    return _paired_rank_correlation(ranked_x, ranked_y, min_obs)


def _paired_rank_correlation(
    ranked_x: np.ndarray,
    ranked_y: np.ndarray,
    min_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """相关每列具有不同缺失掩码的两组秩矩阵。"""

    valid = np.isfinite(ranked_x) & np.isfinite(ranked_y)
    count = valid.sum(axis=0).astype(np.int64)
    safe_count = np.maximum(count, 1)
    x0 = np.where(valid, ranked_x, 0.0)
    y0 = np.where(valid, ranked_y, 0.0)
    sum_x = x0.sum(axis=0)
    sum_y = y0.sum(axis=0)
    covariance = (x0 * y0).sum(axis=0) - sum_x * sum_y / safe_count
    variance_x = (x0 * x0).sum(axis=0) - sum_x * sum_x / safe_count
    variance_y = (y0 * y0).sum(axis=0) - sum_y * sum_y / safe_count
    denominator = np.sqrt(np.maximum(variance_x, 0.0) * np.maximum(variance_y, 0.0))
    correlation = np.divide(
        covariance,
        denominator,
        out=np.full(ranked_x.shape[1], np.nan),
        where=(count >= min_obs) & (denominator > 1e-15),
    )
    return correlation, count


def assign_quantiles(values: np.ndarray, quantiles: int = 10) -> np.ndarray:
    """按列用平均秩分组；相同取值保持同组，不用股票代码人为打破 ties。"""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if quantiles < 2:
        raise ValueError("quantiles 至少为 2")
    finite = np.isfinite(matrix)
    count = finite.sum(axis=0)
    ranks = rankdata(matrix, axis=0, method="average", nan_policy="omit")
    groups = np.floor((ranks - 1.0) * quantiles / np.maximum(count, 1)).astype(
        np.float64
    ) + 1.0
    groups = np.clip(groups, 1.0, float(quantiles))
    groups[~finite] = np.nan
    return groups


def equal_weight_turnover(
    previous_top: np.ndarray,
    current_top: np.ndarray,
    previous_common_rows: np.ndarray,
    current_common_rows: np.ndarray,
) -> np.ndarray:
    """计算两个相邻日期等权 Top 组合的单边换手率。"""

    previous = np.asarray(previous_top, dtype=bool)
    current = np.asarray(current_top, dtype=bool)
    previous_count = previous.sum(axis=0)
    current_count = current.sum(axis=0)
    overlap = (
        previous[previous_common_rows] & current[current_common_rows]
    ).sum(axis=0)
    minimum_weight = np.minimum(
        np.divide(1.0, previous_count, out=np.zeros_like(previous_count, dtype=float), where=previous_count > 0),
        np.divide(1.0, current_count, out=np.zeros_like(current_count, dtype=float), where=current_count > 0),
    )
    turnover = 1.0 - overlap * minimum_weight
    turnover[(previous_count == 0) | (current_count == 0)] = np.nan
    return turnover


def newey_west_mean_se(values: np.ndarray | pd.Series, max_lag: int = 5) -> float:
    """估计时间序列均值的 Newey–West 标准误。"""

    series = np.asarray(values, dtype=np.float64)
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 3:
        return np.nan
    centered = series - series.mean()
    lag = min(max_lag, n - 1)
    long_run_variance = float(centered @ centered / n)
    for offset in range(1, lag + 1):
        weight = 1.0 - offset / (lag + 1.0)
        covariance = float(centered[offset:] @ centered[:-offset] / n)
        long_run_variance += 2.0 * weight * covariance
    return float(np.sqrt(max(long_run_variance, 0.0) / n))


def _matched_rows(
    panel: PanelIndex,
    source_date: int,
    target_date: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回两个日期中相同股票的日内行号。"""

    source_slice = slice(panel.boundaries[source_date], panel.boundaries[source_date + 1])
    target_slice = slice(panel.boundaries[target_date], panel.boundaries[target_date + 1])
    _, source_rows, target_rows = np.intersect1d(
        panel.stock_ids[source_slice],
        panel.stock_ids[target_slice],
        assume_unique=True,
        return_indices=True,
    )
    return source_rows, target_rows


def evaluate_factor_matrix(
    values: np.ndarray,
    column_names: list[str],
    panel: PanelIndex,
    min_obs: int = 100,
    quantiles: int = 10,
    decay_horizons: tuple[int, ...] = (1, 5),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """评价一批因子列，返回日度指标和信号衰减摘要。

    horizon=5 表示因子日后第5个交易日对应的那一段 1 日收益，
    不是把主标签改成5日累计收益。
    """

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if len(matrix) != len(panel.dates) or matrix.shape[1] != len(column_names):
        raise ValueError("因子矩阵形状与 panel 或列名不一致")
    if any(horizon < 1 for horizon in decay_horizons):
        raise ValueError("衰减 horizon 必须为正整数")

    n_dates = len(panel.unique_dates)
    n_columns = matrix.shape[1]
    pearson = np.full((n_dates, n_columns), np.nan)
    rank_ic = np.full_like(pearson, np.nan)
    observations = np.zeros((n_dates, n_columns), dtype=np.int32)
    group_returns = np.full((n_dates, n_columns, quantiles), np.nan)
    turnover = np.full_like(pearson, np.nan)
    decay = {
        horizon: np.full((n_dates, n_columns), np.nan)
        for horizon in decay_horizons
    }
    decay_n = {
        horizon: np.zeros((n_dates, n_columns), dtype=np.int32)
        for horizon in decay_horizons
    }
    previous_top: np.ndarray | None = None

    for date_number in range(n_dates):
        start, end = panel.boundaries[date_number : date_number + 2]
        daily_values = matrix[start:end]
        daily_label = panel.labels[start:end]
        pearson[date_number], observations[date_number] = columnwise_pearson(
            daily_values, daily_label, min_obs=min_obs
        )
        rank_ic[date_number], _ = columnwise_spearman(
            daily_values, daily_label, min_obs=min_obs
        )
        if 1 in decay:
            decay[1][date_number] = rank_ic[date_number]
            decay_n[1][date_number] = observations[date_number]

        groups = assign_quantiles(daily_values, quantiles=quantiles)
        finite_label = np.isfinite(daily_label)[:, None]
        for group in range(1, quantiles + 1):
            selected = (groups == group) & finite_label
            count = selected.sum(axis=0)
            total = np.where(selected, daily_label[:, None], 0.0).sum(axis=0)
            group_returns[date_number, :, group - 1] = np.divide(
                total,
                count,
                out=np.full(n_columns, np.nan),
                where=count > 0,
            )

        current_top = groups == quantiles
        if previous_top is not None:
            previous_rows, current_rows = _matched_rows(
                panel, date_number - 1, date_number
            )
            turnover[date_number] = equal_weight_turnover(
                previous_top, current_top, previous_rows, current_rows
            )
        previous_top = current_top

        for horizon in decay_horizons:
            if horizon == 1:
                continue
            target_date = date_number + horizon - 1
            if target_date >= n_dates:
                continue
            source_rows, target_rows = _matched_rows(panel, date_number, target_date)
            target_start = panel.boundaries[target_date]
            future_label = panel.labels[target_start + target_rows]
            future_ic, future_n = columnwise_spearman(
                daily_values[source_rows], future_label, min_obs=min_obs
            )
            decay[horizon][date_number] = future_ic
            decay_n[horizon][date_number] = future_n

    factor_names = [name.rsplit("__", 1)[0] for name in column_names]
    variants = [name.rsplit("__", 1)[1] for name in column_names]
    daily_data: dict[str, object] = {
        "date": np.repeat(panel.unique_dates, n_columns),
        "factor": np.tile(factor_names, n_dates),
        "variant": np.tile(variants, n_dates),
        "n_obs": observations.ravel(),
        "pearson_ic": pearson.ravel(),
        "rank_ic": rank_ic.ravel(),
        "long_short_return": (
            group_returns[:, :, -1] - group_returns[:, :, 0]
        ).ravel(),
        "top_decile_turnover": turnover.ravel(),
    }
    for group in range(1, quantiles + 1):
        daily_data[f"q{group}_return"] = group_returns[:, :, group - 1].ravel()
    daily = pd.DataFrame(daily_data)

    decay_rows: list[dict[str, object]] = []
    for column, (factor, variant) in enumerate(zip(factor_names, variants)):
        for horizon in decay_horizons:
            series = decay[horizon][:, column]
            finite = np.isfinite(series)
            standard_deviation = float(np.std(series[finite], ddof=1)) if finite.sum() > 1 else np.nan
            mean = float(np.mean(series[finite])) if finite.any() else np.nan
            decay_rows.append(
                {
                    "factor": factor,
                    "variant": variant,
                    "horizon": horizon,
                    "n_days": int(finite.sum()),
                    "median_cross_section_n": (
                        float(np.median(decay_n[horizon][finite, column]))
                        if finite.any()
                        else np.nan
                    ),
                    "mean_rank_ic": mean,
                    "rank_ic_std": standard_deviation,
                    "icir": (
                        mean / standard_deviation
                        if np.isfinite(standard_deviation) and standard_deviation > 0
                        else np.nan
                    ),
                    "nw_se": newey_west_mean_se(series),
                }
            )
    return daily, pd.DataFrame(decay_rows)


def summarize_daily_evaluation(
    daily: pd.DataFrame,
    decay: pd.DataFrame,
    quantiles: int = 10,
) -> pd.DataFrame:
    """把日度指标汇总为每个 factor-variant 的基线报告。"""

    rows: list[dict[str, object]] = []
    for (factor, variant), group in daily.groupby(["factor", "variant"], sort=False):
        rank_ic = group["rank_ic"].to_numpy(dtype=float)
        pearson_ic = group["pearson_ic"].to_numpy(dtype=float)
        rank_finite = rank_ic[np.isfinite(rank_ic)]
        pearson_finite = pearson_ic[np.isfinite(pearson_ic)]
        rank_mean = float(rank_finite.mean()) if rank_finite.size else np.nan
        rank_std = float(rank_finite.std(ddof=1)) if rank_finite.size > 1 else np.nan
        nw_se = newey_west_mean_se(rank_finite)
        row: dict[str, object] = {
            "factor": factor,
            "variant": variant,
            "n_days": int(rank_finite.size),
            "median_cross_section_n": float(group.loc[group["rank_ic"].notna(), "n_obs"].median()),
            "mean_pearson_ic": float(pearson_finite.mean()) if pearson_finite.size else np.nan,
            "mean_rank_ic": rank_mean,
            "rank_ic_std": rank_std,
            "icir": rank_mean / rank_std if np.isfinite(rank_std) and rank_std > 0 else np.nan,
            "annualized_icir": np.sqrt(252.0) * rank_mean / rank_std if np.isfinite(rank_std) and rank_std > 0 else np.nan,
            "ic_positive_ratio": float(np.mean(rank_finite > 0)) if rank_finite.size else np.nan,
            "directional_hit_ratio": float(np.mean(np.sign(rank_mean) * rank_finite > 0)) if rank_finite.size and rank_mean != 0 else np.nan,
            "nw_se_rank_ic": nw_se,
            "nw_t_rank_ic": rank_mean / nw_se if np.isfinite(nw_se) and nw_se > 0 else np.nan,
            "mean_long_short_return": float(group["long_short_return"].mean()),
            "mean_top_decile_turnover": float(group["top_decile_turnover"].mean()),
        }
        quantile_means = []
        for quantile in range(1, quantiles + 1):
            value = float(group[f"q{quantile}_return"].mean())
            row[f"q{quantile}_mean_return"] = value
            quantile_means.append(value)
        row["quantile_monotonicity"] = (
            float(pd.Series(quantile_means).corr(pd.Series(range(1, quantiles + 1)), method="spearman"))
            if np.isfinite(quantile_means).sum() >= 3
            else np.nan
        )
        for horizon in decay["horizon"].unique():
            match = decay[
                (decay["factor"] == factor)
                & (decay["variant"] == variant)
                & (decay["horizon"] == horizon)
            ]
            if not match.empty:
                row[f"rank_ic_h{int(horizon)}"] = float(match.iloc[0]["mean_rank_ic"])
        rows.append(row)
    return pd.DataFrame(rows)
