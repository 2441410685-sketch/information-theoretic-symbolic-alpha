"""Mutual Information 的置换检验与金融 panel null distribution。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .mutual_information import estimate_mi_ksg_details
from .panel_mi import transform_panel_by_date


Alternative = Literal["greater", "less", "two-sided"]
SeriesPermutationMethod = Literal["iid", "block"]
PanelPermutationMethod = Literal["within_date", "date_block"]


@dataclass(frozen=True)
class PermutationTestResult:
    """保存观测统计量、完整 null distribution 与 Monte Carlo p-value。"""

    observed: float
    null_distribution: np.ndarray
    p_value: float
    exceedances: int
    alternative: Alternative
    method: str
    n_permutations: int
    n_samples: int
    k: int
    block_length: int | None
    random_state: int
    n_dates: int | None = None
    n_stocks: int | None = None

    @property
    def null_mean(self) -> float:
        """返回 null distribution 的 Monte Carlo 均值。"""

        return float(np.mean(self.null_distribution))

    @property
    def null_std(self) -> float:
        """返回 null distribution 的 Monte Carlo 标准差。"""

        return float(np.std(self.null_distribution, ddof=1))

    @property
    def null_q95(self) -> float:
        """返回单侧检验常用的 null 95% 分位数。"""

        return float(np.quantile(self.null_distribution, 0.95))

    def to_record(self) -> dict[str, object]:
        """转换为便于保存为 CSV 的单行记录。"""

        return {
            "observed_mi": self.observed,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "null_q95": self.null_q95,
            "exceedances": self.exceedances,
            "p_value": self.p_value,
            "alternative": self.alternative,
            "method": self.method,
            "n_permutations": self.n_permutations,
            "n_samples": self.n_samples,
            "k": self.k,
            "block_length": self.block_length,
            "random_state": self.random_state,
            "n_dates": self.n_dates,
            "n_stocks": self.n_stocks,
        }


def monte_carlo_p_value(
    observed: float,
    null_distribution: np.ndarray,
    alternative: Alternative = "greater",
) -> tuple[float, int]:
    """用加一修正计算 Monte Carlo p-value。

    加一修正 ``(b + 1) / (B + 1)`` 把观测样本也视为置换集合中的一个
    成员，因此有限次模拟永远不会产生虚假的 p=0。
    """

    null = np.asarray(null_distribution, dtype=np.float64).ravel()
    if not np.isfinite(observed) or len(null) < 1 or np.any(~np.isfinite(null)):
        raise ValueError("observed 与 null_distribution 必须是有限数值")
    if alternative == "greater":
        exceedances = int(np.sum(null >= observed))
    elif alternative == "less":
        exceedances = int(np.sum(null <= observed))
    elif alternative == "two-sided":
        center = float(np.median(null))
        exceedances = int(np.sum(np.abs(null - center) >= abs(observed - center)))
    else:
        raise ValueError("alternative 必须是 greater、less 或 two-sided")
    return (exceedances + 1.0) / (len(null) + 1.0), exceedances


def block_permutation_indices(
    n_observations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """重排非重叠连续块，同时保持每个块内部的时间顺序。"""

    if not isinstance(n_observations, (int, np.integer)) or n_observations < 2:
        raise ValueError("n_observations 必须是至少2的整数")
    if not isinstance(block_length, (int, np.integer)):
        raise ValueError("block_length 必须是整数")
    block_length = int(block_length)
    if not 1 <= block_length < n_observations:
        raise ValueError("block_length 必须满足 1 <= block_length < N")
    blocks = [
        np.arange(start, min(start + block_length, n_observations), dtype=np.int64)
        for start in range(0, n_observations, block_length)
    ]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[index] for index in order])


def _finite_vector_pair(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """验证一维连续序列；block test 不允许删除缺失后伪造时间邻接。"""

    x_array = np.asarray(x, dtype=np.float64).ravel()
    y_array = np.asarray(y, dtype=np.float64).ravel()
    if len(x_array) != len(y_array) or len(x_array) < 10:
        raise ValueError("x 与 y 必须是长度相等且至少含10个样本的一维序列")
    if np.any(~np.isfinite(x_array)) or np.any(~np.isfinite(y_array)):
        raise ValueError("置换检验前必须显式处理 NaN/inf")
    return x_array, y_array


def _estimate_ksg(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    random_state: int,
) -> float:
    """统一 KSG 设置并保留可能略小于0的原始估计。"""

    return estimate_mi_ksg_details(
        x,
        y,
        k=k,
        tie_strategy="jitter",
        random_state=random_state,
    ).estimate


def permutation_test_mi(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    *,
    n_permutations: int = 199,
    method: SeriesPermutationMethod = "iid",
    block_length: int | None = None,
    alternative: Alternative = "greater",
    k: int = 5,
    random_state: int = 20260901,
) -> PermutationTestResult:
    """对一对连续序列执行 IID 或时间块 KSG MI 置换检验。"""

    x_array, y_array = _finite_vector_pair(x, y)
    if not isinstance(n_permutations, (int, np.integer)) or n_permutations < 19:
        raise ValueError("n_permutations 至少为19")
    if method not in ("iid", "block"):
        raise ValueError("method 必须是 iid 或 block")
    if method == "block" and block_length is None:
        raise ValueError("block method 必须指定 block_length")
    if method == "iid" and block_length is not None:
        raise ValueError("iid method 不应指定 block_length")

    observed = _estimate_ksg(x_array, y_array, k, random_state)
    rng = np.random.default_rng(random_state)
    null = np.empty(int(n_permutations), dtype=np.float64)
    for repetition in range(int(n_permutations)):
        if method == "iid":
            indices = rng.permutation(len(y_array))
        else:
            indices = block_permutation_indices(
                len(y_array),
                int(block_length),
                rng,
            )
        null[repetition] = _estimate_ksg(
            x_array,
            y_array[indices],
            k,
            random_state + repetition + 1,
        )
    p_value, exceedances = monte_carlo_p_value(observed, null, alternative)
    return PermutationTestResult(
        observed=observed,
        null_distribution=null,
        p_value=p_value,
        exceedances=exceedances,
        alternative=alternative,
        method=method,
        n_permutations=int(n_permutations),
        n_samples=len(x_array),
        k=k,
        block_length=block_length,
        random_state=random_state,
    )


def _balanced_panel_matrices(
    panel: pd.DataFrame,
    factor: str,
    label: str,
    date_col: str,
    stock_col: str,
    maximum_stocks: int,
    random_state: int,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """构造固定股票集合的按日 Rank-Gaussian 矩阵。

    日期块置换必须把整张 label 横截面移动到另一个日期。固定股票集合可使
    每次 null replicate 的样本数完全相同，避免缺失变化污染统计量比较。
    """

    required = {date_col, stock_col, factor, label}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少字段：{sorted(missing)}")
    if maximum_stocks < max(20, 5 + 1):
        raise ValueError("maximum_stocks 至少为20")
    frame = panel[[date_col, stock_col, factor, label]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    if frame[date_col].isna().any() or frame[stock_col].isna().any():
        raise ValueError("date 与 stock 不能缺失")
    if frame.duplicated([date_col, stock_col]).any():
        raise ValueError("(date, stock) 主键必须唯一")
    for column in [factor, label]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[~np.isfinite(frame[column]), column] = np.nan
    frame = frame.sort_values([date_col, stock_col]).reset_index(drop=True)
    dates = pd.DatetimeIndex(frame[date_col].drop_duplicates().sort_values())
    if len(dates) < 3:
        raise ValueError("panel 至少需要3个日期")

    # 必须先在完整日期截面上变换，再选择平衡股票；否则 rank 的参考股票池会
    # 被事后的完整样本筛选改变。
    transformed = transform_panel_by_date(
        frame,
        [factor, label],
        method="rankgauss_by_date",
        date_col=date_col,
    ).dropna(subset=[factor, label])
    counts = transformed.groupby(stock_col, observed=True)[date_col].nunique()
    eligible = np.asarray(counts.index[counts.eq(len(dates))], dtype=object)
    if len(eligible) < 20:
        raise ValueError("完整覆盖所有选定日期的股票少于20只")
    eligible = np.sort(eligible.astype(str))
    if len(eligible) > maximum_stocks:
        rng = np.random.default_rng(random_state)
        stocks = np.sort(
            rng.choice(eligible, size=maximum_stocks, replace=False)
        )
    else:
        stocks = eligible
    balanced = transformed.loc[
        transformed[stock_col].astype(str).isin(stocks)
    ].copy()
    balanced[stock_col] = balanced[stock_col].astype(str)
    x_matrix = (
        balanced.pivot(index=date_col, columns=stock_col, values=factor)
        .reindex(index=dates, columns=stocks)
        .to_numpy(dtype=np.float64)
    )
    y_matrix = (
        balanced.pivot(index=date_col, columns=stock_col, values=label)
        .reindex(index=dates, columns=stocks)
        .to_numpy(dtype=np.float64)
    )
    if np.any(~np.isfinite(x_matrix)) or np.any(~np.isfinite(y_matrix)):
        raise RuntimeError("平衡 panel 构造后不应含 NaN/inf")
    return dates, stocks, x_matrix, y_matrix


def _mean_daily_ksg(
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    k: int,
    random_state: int,
) -> float:
    """以日期等权汇总每日横截面 KSG MI。"""

    if x_matrix.shape != y_matrix.shape or x_matrix.ndim != 2:
        raise ValueError("x_matrix 与 y_matrix 必须是相同形状的二维矩阵")
    estimates = [
        _estimate_ksg(
            x_matrix[date_number],
            y_matrix[date_number],
            k,
            random_state + date_number * 1009,
        )
        for date_number in range(x_matrix.shape[0])
    ]
    return float(np.mean(estimates))


def panel_permutation_test_mi(
    panel: pd.DataFrame,
    factor: str,
    label: str,
    *,
    n_permutations: int = 199,
    method: PanelPermutationMethod = "within_date",
    block_length: int | None = None,
    maximum_stocks: int = 200,
    date_col: str = "date",
    stock_col: str = "stock",
    alternative: Alternative = "greater",
    k: int = 5,
    random_state: int = 20260901,
) -> PermutationTestResult:
    """对金融 panel 的平均每日横截面 MI 构造结构化 null。

    ``within_date`` 在每个日期内部重排股票标签，条件于当日边际分布；
    ``date_block`` 把完整 label 横截面按连续日期块重排，保留块内时间顺序、
    股票身份与同日横截面结构。
    """

    if not isinstance(n_permutations, (int, np.integer)) or n_permutations < 19:
        raise ValueError("n_permutations 至少为19")
    if method not in ("within_date", "date_block"):
        raise ValueError("method 必须是 within_date 或 date_block")
    if method == "date_block" and block_length is None:
        raise ValueError("date_block 必须指定 block_length")
    if method == "within_date" and block_length is not None:
        raise ValueError("within_date 不应指定 block_length")

    dates, stocks, x_matrix, y_matrix = _balanced_panel_matrices(
        panel,
        factor,
        label,
        date_col,
        stock_col,
        maximum_stocks,
        random_state,
    )
    if method == "date_block":
        # 提前验证 block 长度，让错误在昂贵的 KSG 循环前暴露。
        block_permutation_indices(
            len(dates),
            int(block_length),
            np.random.default_rng(random_state),
        )

    observed = _mean_daily_ksg(x_matrix, y_matrix, k, random_state)
    rng = np.random.default_rng(random_state)
    null = np.empty(int(n_permutations), dtype=np.float64)
    for repetition in range(int(n_permutations)):
        if method == "within_date":
            permuted_y = np.empty_like(y_matrix)
            for date_number in range(len(dates)):
                permuted_y[date_number] = y_matrix[
                    date_number,
                    rng.permutation(len(stocks)),
                ]
        else:
            date_indices = block_permutation_indices(
                len(dates),
                int(block_length),
                rng,
            )
            permuted_y = y_matrix[date_indices]
        null[repetition] = _mean_daily_ksg(
            x_matrix,
            permuted_y,
            k,
            random_state + (repetition + 1) * 100_003,
        )
    p_value, exceedances = monte_carlo_p_value(observed, null, alternative)
    return PermutationTestResult(
        observed=observed,
        null_distribution=null,
        p_value=p_value,
        exceedances=exceedances,
        alternative=alternative,
        method=method,
        n_permutations=int(n_permutations),
        n_samples=int(x_matrix.size),
        k=k,
        block_length=block_length,
        random_state=random_state,
        n_dates=len(dates),
        n_stocks=len(stocks),
    )


def pooled_rankgauss_block_permutation_test_mi(
    panel: pd.DataFrame,
    factor: str,
    label: str,
    *,
    n_permutations: int = 99,
    block_length: int = 20,
    date_col: str = "date",
    stock_col: str = "stock",
    k: int = 5,
    minimum_cross_section: int = 50,
    random_state: int = 20260918,
) -> PermutationTestResult:
    """对按日 Rank-Gaussian 后的 pooled KSG MI 做整日时间块置换。

    与 ``panel_permutation_test_mi`` 的“日度 MI 均值”统计量不同，
    本函数与 Day 23 GP fitness 使用同一个 pooled KSG 统计量。
    置换单位是完整日期截面，连续日期块内顺序保留。
    """

    if not isinstance(n_permutations, (int, np.integer)) or n_permutations < 19:
        raise ValueError("n_permutations 至少为 19")
    if minimum_cross_section < max(3, k + 1):
        raise ValueError("minimum_cross_section 过小")
    required = {date_col, stock_col, factor, label}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel 缺少字段：{sorted(missing)}")
    frame = panel[[date_col, stock_col, factor, label]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    if frame[date_col].isna().any() or frame[stock_col].isna().any():
        raise ValueError("date 与 stock 不能缺失")
    if frame.duplicated([date_col, stock_col]).any():
        raise ValueError("(date, stock) 主键必须唯一")
    transformed = transform_panel_by_date(
        frame,
        [factor, label],
        method="rankgauss_by_date",
        date_col=date_col,
    ).dropna(subset=[factor, label])
    transformed = transformed.sort_values([date_col, stock_col]).reset_index(drop=True)
    counts = transformed.groupby(date_col, sort=True).size()
    if len(counts) < 3 or counts.min() < minimum_cross_section:
        raise ValueError("有效日期或每日横截面样本不足")
    # 某些合法表达式可能在少数股票上产生 NaN。必须先在
    # 当日全部有效截面上 Rank Gaussianize，然后才能按最小日度
    # 样本数做确定性等宽截取；若先截取，rank 参考集会改变。
    width = int(counts.min())
    if counts.nunique() != 1:
        transformed = (
            transformed.groupby(date_col, sort=True, group_keys=False)
            .head(width)
            .reset_index(drop=True)
        )
    dates = pd.DatetimeIndex(counts.index)
    x_matrix = transformed[factor].to_numpy(dtype=float).reshape(len(dates), width)
    y_matrix = transformed[label].to_numpy(dtype=float).reshape(len(dates), width)
    # 在进入昂贵循环前验证 block 长度。
    block_permutation_indices(
        len(dates), block_length, np.random.default_rng(random_state)
    )
    observed = estimate_mi_ksg_details(
        x_matrix.reshape(-1),
        y_matrix.reshape(-1),
        k=k,
        tie_strategy="jitter",
        random_state=random_state,
    ).estimate
    rng = np.random.default_rng(random_state)
    null = np.empty(int(n_permutations), dtype=float)
    for repetition in range(int(n_permutations)):
        date_indices = block_permutation_indices(len(dates), block_length, rng)
        null[repetition] = estimate_mi_ksg_details(
            x_matrix.reshape(-1),
            y_matrix[date_indices].reshape(-1),
            k=k,
            tie_strategy="jitter",
            random_state=random_state + (repetition + 1) * 100_003,
        ).estimate
    p_value, exceedances = monte_carlo_p_value(observed, null, "greater")
    return PermutationTestResult(
        observed=float(observed),
        null_distribution=null,
        p_value=p_value,
        exceedances=exceedances,
        alternative="greater",
        method="pooled_rankgauss_date_block",
        n_permutations=int(n_permutations),
        n_samples=int(x_matrix.size),
        k=k,
        block_length=block_length,
        random_state=random_state,
        n_dates=len(dates),
        n_stocks=width,
    )
