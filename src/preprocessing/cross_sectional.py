"""Day 7：截尾、中性化、z-score 与 Rank Gaussianization。"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata


def _finite_matrix(values: np.ndarray | pd.DataFrame | pd.Series) -> np.ndarray:
    """统一转成二维浮点矩阵，并把 inf 视为缺失。"""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError("因子输入必须是一维或二维")
    return np.where(np.isfinite(matrix), matrix, np.nan)


def derive_log_float_market_cap(
    close: np.ndarray | pd.Series,
    volume: np.ndarray | pd.Series,
    turnover_pct: np.ndarray | pd.Series,
) -> np.ndarray:
    """由当日成交量和百分比换手率推导流通市值的对数代理。

    BaoStock 的换手率以百分数记录，因此：
    流通股本约等于 volume / (turnover_pct / 100)。
    """

    close_array = np.asarray(close, dtype=np.float64)
    volume_array = np.asarray(volume, dtype=np.float64)
    turnover_array = np.asarray(turnover_pct, dtype=np.float64)
    valid = (
        np.isfinite(close_array)
        & np.isfinite(volume_array)
        & np.isfinite(turnover_array)
        & (close_array > 0)
        & (volume_array > 0)
        & (turnover_array > 0)
    )
    result = np.full(close_array.shape, np.nan, dtype=np.float64)
    float_market_cap = close_array[valid] * volume_array[valid] * 100.0 / turnover_array[valid]
    result[valid] = np.log(float_market_cap)
    return result


def winsorize_mad_matrix(
    values: np.ndarray | pd.DataFrame | pd.Series,
    n_mad: float = 5.0,
) -> np.ndarray:
    """逐列执行 median ± n_mad × 1.4826 × MAD 截尾。"""

    matrix = _finite_matrix(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(matrix, axis=0)
        mad = np.nanmedian(np.abs(matrix - median), axis=0)
    scale = 1.4826 * mad
    lower = median - n_mad * scale
    upper = median + n_mad * scale
    result = np.clip(matrix, lower, upper)

    # 离散因子可能 MAD=0 但并非常数，此时退回分位数截尾，避免把整列压成中位数。
    fallback = (~np.isfinite(scale)) | (scale <= 1e-12)
    for column in np.flatnonzero(fallback):
        finite = matrix[np.isfinite(matrix[:, column]), column]
        if finite.size == 0 or np.ptp(finite) <= 1e-12:
            continue
        low, high = np.quantile(finite, [0.01, 0.99])
        result[:, column] = np.clip(matrix[:, column], low, high)
    return result


def winsorize_percentile_matrix(
    values: np.ndarray | pd.DataFrame | pd.Series,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> np.ndarray:
    """逐列执行分位数截尾，用作 MAD 主方案的敏感性比较。"""

    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("分位点必须满足 0 <= lower < upper <= 1")
    matrix = _finite_matrix(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lower = np.nanquantile(matrix, lower_quantile, axis=0)
        upper = np.nanquantile(matrix, upper_quantile, axis=0)
    return np.clip(matrix, lower, upper)


def neutralize_matrix(
    values: np.ndarray | pd.DataFrame | pd.Series,
    log_market_cap: np.ndarray | pd.Series,
    industry: np.ndarray | pd.Series | None = None,
    min_obs: int = 30,
) -> np.ndarray:
    """按行业去均值后回归市值，得到截距+行业+市值 OLS 残差。

    该实现利用 Frisch-Waugh-Lovell 定理，等价于显式构造行业哑变量，
    但不会建立宽而稀疏的设计矩阵。
    """

    matrix = _finite_matrix(values)
    size = np.asarray(log_market_cap, dtype=np.float64)
    if len(size) != len(matrix):
        raise ValueError("市值与因子样本长度必须一致")

    if industry is None:
        codes = np.zeros(len(matrix), dtype=np.int32)
    else:
        industry_series = pd.Series(industry, dtype="string")
        codes, _ = pd.factorize(industry_series, sort=True, use_na_sentinel=True)

    valid_size_industry = np.isfinite(size) & (codes >= 0)
    valid = np.isfinite(matrix) & valid_size_industry[:, None]
    centered_y = np.full_like(matrix, np.nan)
    centered_x = np.full_like(matrix, np.nan)

    for code in np.unique(codes[codes >= 0]):
        rows = codes == code
        y_group = matrix[rows]
        valid_group = valid[rows]
        counts = valid_group.sum(axis=0)
        safe_counts = np.where(counts > 0, counts, 1)
        y_mean = np.where(
            counts > 0,
            np.nansum(np.where(valid_group, y_group, np.nan), axis=0) / safe_counts,
            np.nan,
        )
        x_group = size[rows, None]
        x_mean = np.where(
            counts > 0,
            np.nansum(np.where(valid_group, x_group, np.nan), axis=0) / safe_counts,
            np.nan,
        )
        centered_y[rows] = np.where(valid_group, y_group - y_mean, np.nan)
        centered_x[rows] = np.where(valid_group, x_group - x_mean, np.nan)

    denominator = np.nansum(centered_x * centered_x, axis=0)
    numerator = np.nansum(centered_x * centered_y, axis=0)
    beta = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    residual = centered_y - centered_x * beta
    valid_columns = valid.sum(axis=0) >= min_obs
    residual[:, ~valid_columns] = np.nan
    return residual


def zscore_matrix(
    values: np.ndarray | pd.DataFrame | pd.Series,
    min_obs: int = 30,
) -> np.ndarray:
    """逐列进行总体标准差口径的横截面 z-score。"""

    matrix = _finite_matrix(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0, ddof=0)
    result = np.divide(
        matrix - mean,
        std,
        out=np.full_like(matrix, np.nan),
        where=std > 1e-12,
    )
    result[:, np.isfinite(matrix).sum(axis=0) < min_obs] = np.nan
    return result


def rank_gaussianize_matrix(
    values: np.ndarray | pd.DataFrame | pd.Series,
    min_obs: int = 30,
) -> np.ndarray:
    """逐列先取平均秩，再通过标准正态逆 CDF 映射。"""

    matrix = _finite_matrix(values)
    counts = np.isfinite(matrix).sum(axis=0)
    ranks = rankdata(matrix, axis=0, method="average", nan_policy="omit")
    probability = (ranks - 0.5) / np.where(counts > 0, counts, 1)
    result = ndtri(probability)
    result[~np.isfinite(matrix)] = np.nan
    result[:, counts < min_obs] = np.nan
    return result


def preprocess_matrix(
    raw_values: np.ndarray | pd.DataFrame | pd.Series,
    log_market_cap: np.ndarray | pd.Series,
    industry: np.ndarray | pd.Series | None = None,
    n_mad: float = 5.0,
    min_obs: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """执行 Raw → MAD截尾 → 中性化 → z-score/Rank-Gaussian。"""

    raw = _finite_matrix(raw_values)
    winsorized = winsorize_mad_matrix(raw, n_mad=n_mad)
    neutralized = neutralize_matrix(
        winsorized,
        log_market_cap,
        industry=industry,
        min_obs=min_obs,
    )
    zscore = zscore_matrix(neutralized, min_obs=min_obs)
    rank_gaussian = rank_gaussianize_matrix(neutralized, min_obs=min_obs)
    return raw, neutralized, zscore, rank_gaussian


def preprocess_cross_section(
    frame: pd.DataFrame,
    factor_columns: list[str],
    size_column: str = "log_float_market_cap",
    industry_column: str | None = "industry_sector",
    n_mad: float = 5.0,
    min_obs: int = 30,
) -> pd.DataFrame:
    """处理单个交易日截面，并保留四种因子表示。"""

    industry = (
        frame[industry_column].to_numpy()
        if industry_column is not None and industry_column in frame
        else None
    )
    raw, neutral, zscore, rank_gaussian = preprocess_matrix(
        frame[factor_columns],
        frame[size_column],
        industry=industry,
        n_mad=n_mad,
        min_obs=min_obs,
    )
    output = frame[[column for column in ("date", "stock") if column in frame]].copy()
    for number, factor in enumerate(factor_columns):
        output[f"{factor}__raw"] = raw[:, number].astype(np.float32)
        output[f"{factor}__neutral"] = neutral[:, number].astype(np.float32)
        output[f"{factor}__zscore"] = zscore[:, number].astype(np.float32)
        output[f"{factor}__rankgauss"] = rank_gaussian[:, number].astype(np.float32)
    return output
