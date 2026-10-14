"""金融 panel 的每日横截面 MI 与 rolling pooled MI。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import ndtri

from .mutual_information import estimate_mi_ksg_details


PanelTransform = Literal["rankgauss_by_date", "zscore_by_date", "global_zscore"]


def _validate_panel(
    panel: pd.DataFrame,
    factors: Sequence[str],
    label: str,
    date_col: str,
    stock_col: str,
) -> pd.DataFrame:
    """检查 panel 的最小研究契约并按 date、stock 排序。"""

    factor_list = list(factors)
    if not factor_list or len(set(factor_list)) != len(factor_list):
        raise ValueError("factors 必须是非空且不重复的列名")
    required = {date_col, stock_col, label, *factor_list}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少字段：{sorted(missing)}")
    frame = panel[[date_col, stock_col, label, *factor_list]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    if frame[date_col].isna().any() or frame[stock_col].isna().any():
        raise ValueError("date 与 stock 不能缺失")
    if frame.duplicated([date_col, stock_col]).any():
        raise ValueError("(date, stock) 主键必须唯一")
    for column in [label, *factor_list]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[~np.isfinite(frame[column]), column] = np.nan
    return frame.sort_values([date_col, stock_col]).reset_index(drop=True)


def _rank_gaussian(values: np.ndarray) -> np.ndarray:
    """把一维有限截面转换为标准正态秩分数，缺失位置保持 NaN。"""

    array = np.asarray(values, dtype=np.float64)
    output = np.full(array.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(array)
    count = int(valid.sum())
    if count < 2:
        return output
    ranks = pd.Series(array[valid]).rank(method="average").to_numpy()
    probabilities = (ranks - 0.5) / count
    output[valid] = ndtri(probabilities)
    return output


def _zscore(values: np.ndarray) -> np.ndarray:
    """标准化有限样本；常数截面返回 NaN，避免制造伪距离。"""

    array = np.asarray(values, dtype=np.float64)
    output = np.full(array.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(array)
    if valid.sum() < 2:
        return output
    scale = float(np.std(array[valid], ddof=0))
    if scale <= 1e-15:
        return output
    output[valid] = (array[valid] - float(np.mean(array[valid]))) / scale
    return output


def transform_panel_by_date(
    panel: pd.DataFrame,
    columns: Sequence[str],
    method: Literal["rankgauss_by_date", "zscore_by_date"],
    date_col: str = "date",
) -> pd.DataFrame:
    """只用每个日期自己的横截面转换指定列。"""

    if method not in ("rankgauss_by_date", "zscore_by_date"):
        raise ValueError("method 必须是按日 rankgauss 或 zscore")
    # groupby.indices 返回的是位置编号；先重建连续索引，避免外部 DataFrame
    # 使用非连续索引时把某一天的变换结果写到错误行。
    output = panel.reset_index(drop=True).copy()
    for column in columns:
        # 因子宽表通常以 float32 保存；秩高斯化会产生 float64。新版 pandas
        # 禁止把后者隐式写回 float32，因此先建立明确的估计器工作精度列。
        output[column] = pd.to_numeric(output[column], errors="coerce").astype(
            np.float64
        )
    transform = _rank_gaussian if method == "rankgauss_by_date" else _zscore
    for indices in output.groupby(date_col, sort=True).indices.values():
        positions = np.asarray(indices, dtype=np.int64)
        for column in columns:
            output.loc[positions, column] = transform(
                output.loc[positions, column].to_numpy(dtype=np.float64)
            )
    return output


def _estimate_pair(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    random_state: int,
) -> tuple[float, float, bool]:
    """估计一对连续变量；显式 jitter 处理金融价格刻度造成的 ties。"""

    result = estimate_mi_ksg_details(
        x,
        y,
        k=k,
        tie_strategy="jitter",
        random_state=random_state,
    )
    return result.estimate, result.median_joint_radius, result.jittered


def daily_cross_sectional_mi(
    panel: pd.DataFrame,
    factors: Sequence[str],
    label: str,
    *,
    date_col: str = "date",
    stock_col: str = "stock",
    transform: Literal["rankgauss_by_date", "zscore_by_date"] = "rankgauss_by_date",
    k: int = 5,
    min_observations: int = 100,
    random_state: int = 20260901,
) -> pd.DataFrame:
    """逐日估计横截面 MI，并为无效日期保留明确状态。"""

    frame = _validate_panel(panel, factors, label, date_col, stock_col)
    transformed = transform_panel_by_date(
        frame,
        [label, *factors],
        transform,
        date_col,
    )
    rows: list[dict[str, object]] = []
    for date_number, (date, group) in enumerate(
        transformed.groupby(date_col, sort=True),
    ):
        for factor_number, factor in enumerate(factors):
            valid = group[[factor, label]].dropna()
            n_observations = len(valid)
            common = {
                "date": date,
                "start_date": date,
                "end_date": date,
                "factor": factor,
                "method": "daily_cross_sectional",
                "transform": transform,
                "window_days": 1,
                "n_dates": 1,
                "nominal_n": n_observations,
                "used_n": n_observations,
            }
            if n_observations < max(min_observations, k + 1):
                rows.append(
                    {
                        **common,
                        "mi": np.nan,
                        "median_joint_radius": np.nan,
                        "jittered": False,
                        "valid": False,
                        "reason": "insufficient_observations",
                    }
                )
                continue
            try:
                estimate, radius, jittered = _estimate_pair(
                    valid[factor].to_numpy(),
                    valid[label].to_numpy(),
                    k,
                    random_state + date_number * 1009 + factor_number,
                )
                rows.append(
                    {
                        **common,
                        "mi": estimate,
                        "median_joint_radius": radius,
                        "jittered": jittered,
                        "valid": True,
                        "reason": "",
                    }
                )
            except ValueError as error:
                rows.append(
                    {
                        **common,
                        "mi": np.nan,
                        "median_joint_radius": np.nan,
                        "jittered": False,
                        "valid": False,
                        "reason": str(error),
                    }
                )
    return pd.DataFrame(rows)


def _stable_date_subsample(
    panel: pd.DataFrame,
    date_col: str,
    maximum_per_date: int | None,
    random_state: int,
) -> pd.DataFrame:
    """对每个日期固定抽样，保证不同因子和窗口使用相同股票子集。"""

    if maximum_per_date is None:
        return panel
    if maximum_per_date < 10:
        raise ValueError("maximum_per_date 至少为10或设为 None")
    selected: list[np.ndarray] = []
    for date, group in panel.groupby(date_col, sort=True):
        indices = group.index.to_numpy()
        if len(indices) <= maximum_per_date:
            selected.append(indices)
            continue
        date_integer = int(pd.Timestamp(date).strftime("%Y%m%d"))
        rng = np.random.default_rng(
            np.random.SeedSequence([int(random_state), date_integer])
        )
        selected.append(
            np.sort(rng.choice(indices, size=maximum_per_date, replace=False))
        )
    return panel.loc[np.concatenate(selected)].sort_values([date_col]).copy()


def _daily_pair_diagnostics(
    frame: pd.DataFrame,
    factors: Sequence[str],
    label: str,
    date_col: str,
) -> dict[str, pd.DataFrame]:
    """预计算每日有效样本、均值和尺度，避免每个 rolling window 重扫全表。"""

    diagnostics: dict[str, pd.DataFrame] = {}
    for factor in factors:
        pair = frame[[date_col, factor, label]].dropna()
        diagnostics[factor] = (
            pair.groupby(date_col, sort=True)
            .agg(
                pair_n=(factor, "size"),
                factor_mean=(factor, "mean"),
                factor_std=(factor, "std"),
                label_mean=(label, "mean"),
                label_std=(label, "std"),
            )
            .reset_index()
        )
    return diagnostics


def _coefficient_of_variation(values: pd.Series) -> float:
    """返回正尺度序列的 CV；无有效均值时返回 NaN。"""

    finite = pd.to_numeric(values, errors="coerce").dropna()
    if len(finite) < 2 or abs(float(finite.mean())) <= 1e-15:
        return float("nan")
    return float(finite.std(ddof=1) / abs(finite.mean()))


def rolling_pooled_mi(
    panel: pd.DataFrame,
    factors: Sequence[str],
    label: str,
    window_days: int,
    *,
    evaluation_dates: Sequence[pd.Timestamp | str] | None = None,
    evaluation_stride: int = 20,
    maximum_per_date: int | None = 300,
    date_col: str = "date",
    stock_col: str = "stock",
    transform: PanelTransform = "rankgauss_by_date",
    k: int = 5,
    min_observations: int = 200,
    random_state: int = 20260901,
) -> pd.DataFrame:
    """在最近 L 个交易日的横截面样本上估计 rolling pooled MI。

    ``rankgauss_by_date`` 与 ``zscore_by_date`` 先按完整每日截面变换再抽样；
    ``global_zscore`` 保留日期间位置/尺度差异，只在每个 pooled window 内做
    一次全局仿射标准化，因此可用于诊断 regime mixing。
    """

    if not isinstance(window_days, (int, np.integer)) or window_days < 1:
        raise ValueError("window_days 必须是正整数")
    if evaluation_stride < 1:
        raise ValueError("evaluation_stride 必须是正整数")
    if transform not in ("rankgauss_by_date", "zscore_by_date", "global_zscore"):
        raise ValueError("不支持的 transform")

    frame = _validate_panel(panel, factors, label, date_col, stock_col)
    dates = pd.DatetimeIndex(frame[date_col].drop_duplicates().sort_values())
    if len(dates) < window_days:
        raise ValueError("panel 的交易日数量小于 window_days")
    diagnostics = _daily_pair_diagnostics(frame, factors, label, date_col)

    if transform in ("rankgauss_by_date", "zscore_by_date"):
        estimation_frame = transform_panel_by_date(
            frame,
            [label, *factors],
            transform,
            date_col,
        )
    else:
        estimation_frame = frame.copy()
    sampled = _stable_date_subsample(
        estimation_frame,
        date_col,
        maximum_per_date,
        random_state,
    )

    if evaluation_dates is None:
        endpoint_positions = list(
            range(window_days - 1, len(dates), evaluation_stride)
        )
        if endpoint_positions[-1] != len(dates) - 1:
            endpoint_positions.append(len(dates) - 1)
    else:
        requested = pd.DatetimeIndex(pd.to_datetime(list(evaluation_dates)))
        positions = dates.get_indexer(requested)
        endpoint_positions = sorted(
            {int(position) for position in positions if position >= window_days - 1}
        )
        if not endpoint_positions:
            raise ValueError("没有满足窗口长度的 evaluation_dates")

    rows: list[dict[str, object]] = []
    for endpoint_number, end_position in enumerate(endpoint_positions):
        window_dates = dates[end_position - window_days + 1 : end_position + 1]
        start_date = window_dates[0]
        end_date = window_dates[-1]
        pooled = sampled.loc[sampled[date_col].isin(window_dates)]
        for factor_number, factor in enumerate(factors):
            valid = pooled[[factor, label]].dropna()
            daily = diagnostics[factor].loc[
                diagnostics[factor][date_col].isin(window_dates)
            ]
            nominal_n = int(daily["pair_n"].sum())
            common = {
                "date": end_date,
                "start_date": start_date,
                "end_date": end_date,
                "factor": factor,
                "method": "rolling_pooled",
                "transform": transform,
                "window_days": int(window_days),
                "n_dates": int(len(window_dates)),
                "nominal_n": nominal_n,
                "used_n": int(len(valid)),
                "factor_daily_mean_std": float(daily["factor_mean"].std(ddof=1)),
                "factor_daily_scale_cv": _coefficient_of_variation(daily["factor_std"]),
                "label_daily_mean_std": float(daily["label_mean"].std(ddof=1)),
                "label_daily_scale_cv": _coefficient_of_variation(daily["label_std"]),
            }
            if len(valid) < max(min_observations, k + 1):
                rows.append(
                    {
                        **common,
                        "mi": np.nan,
                        "median_joint_radius": np.nan,
                        "jittered": False,
                        "valid": False,
                        "reason": "insufficient_observations",
                    }
                )
                continue

            x = valid[factor].to_numpy(dtype=np.float64)
            y = valid[label].to_numpy(dtype=np.float64)
            if transform == "global_zscore":
                x = _zscore(x)
                y = _zscore(y)
                finite = np.isfinite(x) & np.isfinite(y)
                x, y = x[finite], y[finite]
            try:
                estimate, radius, jittered = _estimate_pair(
                    x,
                    y,
                    k,
                    random_state + endpoint_number * 1009 + factor_number,
                )
                rows.append(
                    {
                        **common,
                        "mi": estimate,
                        "median_joint_radius": radius,
                        "jittered": jittered,
                        "valid": True,
                        "reason": "",
                    }
                )
            except ValueError as error:
                rows.append(
                    {
                        **common,
                        "mi": np.nan,
                        "median_joint_radius": np.nan,
                        "jittered": False,
                        "valid": False,
                        "reason": str(error),
                    }
                )
    return pd.DataFrame(rows)


def summarize_panel_mi(estimates: pd.DataFrame) -> pd.DataFrame:
    """按 factor、method、transform 和窗口汇总 MI 的时间稳定性。"""

    required = {"factor", "method", "transform", "window_days", "mi", "valid", "used_n"}
    missing = required.difference(estimates.columns)
    if missing:
        raise ValueError(f"estimates 缺少字段：{sorted(missing)}")
    valid = estimates.loc[estimates["valid"] & estimates["mi"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.groupby(
            ["factor", "method", "transform", "window_days"],
            sort=False,
        )
        .agg(
            n_estimates=("mi", "size"),
            mean_mi=("mi", "mean"),
            median_mi=("mi", "median"),
            std_mi=("mi", "std"),
            q05_mi=("mi", lambda x: float(np.quantile(x, 0.05))),
            q95_mi=("mi", lambda x: float(np.quantile(x, 0.95))),
            positive_ratio=("mi", lambda x: float(np.mean(x > 0.0))),
            median_used_n=("used_n", "median"),
        )
        .reset_index()
    )
