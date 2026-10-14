"""离散熵、联合熵与条件熵的有限样本估计。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln


Correction = Literal["none", "miller_madow"]
DistanceMetric = Literal["euclidean", "chebyshev"]
TieStrategy = Literal["raise", "jitter"]


@dataclass(frozen=True)
class KLEntropyResult:
    """保存 KL estimator 数值与局部距离诊断。"""

    estimate: float
    n_samples: int
    dimension: int
    k: int
    metric: DistanceMetric
    log_unit_ball_volume: float
    mean_log_radius: float
    min_radius: float
    median_radius: float
    max_radius: float
    jittered: bool


def _validate_log_base(base: float) -> float:
    """检查对数底；自然对数的单位是 nat，底2的单位是 bit。"""

    if not np.isfinite(base) or base <= 0.0 or np.isclose(base, 1.0):
        raise ValueError("对数底必须为正数且不能等于1")
    return float(math.log(base))


def counts_from_values(values: np.ndarray | pd.Series) -> np.ndarray:
    """统计离散状态频数，并把缺失值排除而不是当成一个状态。"""

    series = pd.Series(values, dtype="object")
    codes, _ = pd.factorize(series, sort=False, use_na_sentinel=True)
    valid = codes >= 0
    if not valid.any():
        raise ValueError("没有可用于估计熵的非缺失样本")
    return np.bincount(codes[valid]).astype(np.int64)


def entropy_from_counts(
    counts: np.ndarray,
    correction: Correction = "none",
    base: float = math.e,
) -> float:
    """由非负频数估计离散熵，可选 Miller–Madow 修正。

    修正只使用实际出现的状态数，因此空类别不会凭空增加参数个数。
    """

    frequencies = np.asarray(counts, dtype=np.float64).ravel()
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies < 0):
        raise ValueError("频数必须是有限非负数")
    positive = frequencies[frequencies > 0]
    sample_size = float(positive.sum())
    if sample_size <= 0:
        raise ValueError("总样本数必须大于0")
    log_base = _validate_log_base(base)
    probabilities = positive / sample_size
    estimate = -float(np.sum(probabilities * np.log(probabilities))) / log_base
    if correction == "miller_madow":
        estimate += (len(positive) - 1.0) / (2.0 * sample_size * log_base)
    elif correction != "none":
        raise ValueError("correction 只能是 'none' 或 'miller_madow'")
    return float(estimate)


def discrete_entropy(
    values: np.ndarray | pd.Series,
    correction: Correction = "none",
    base: float = math.e,
) -> float:
    """估计一个离散变量的 Shannon entropy。"""

    return entropy_from_counts(counts_from_values(values), correction, base)


def joint_state_counts(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
) -> np.ndarray:
    """构造两个离散变量的非空联合状态频数。"""

    x_series = pd.Series(x, dtype="object")
    y_series = pd.Series(y, dtype="object")
    if len(x_series) != len(y_series):
        raise ValueError("x 与 y 的样本长度必须一致")
    x_codes, _ = pd.factorize(x_series, sort=False, use_na_sentinel=True)
    y_codes, _ = pd.factorize(y_series, sort=False, use_na_sentinel=True)
    valid = (x_codes >= 0) & (y_codes >= 0)
    if not valid.any():
        raise ValueError("x 与 y 没有成对的非缺失样本")
    y_states = int(y_codes[valid].max()) + 1
    joint_codes = x_codes[valid].astype(np.int64) * y_states + y_codes[valid]
    return np.bincount(joint_codes).astype(np.int64)


def joint_entropy(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    correction: Correction = "none",
    base: float = math.e,
) -> float:
    """估计联合熵 H(X,Y)。"""

    return entropy_from_counts(joint_state_counts(x, y), correction, base)


def conditional_entropy(
    x: np.ndarray | pd.Series,
    given_y: np.ndarray | pd.Series,
    correction: Correction = "none",
    base: float = math.e,
) -> float:
    """通过 H(X|Y)=H(X,Y)-H(Y) 估计条件熵。"""

    x_series = pd.Series(x, dtype="object")
    y_series = pd.Series(given_y, dtype="object")
    if len(x_series) != len(y_series):
        raise ValueError("x 与 given_y 的样本长度必须一致")
    valid = x_series.notna().to_numpy() & y_series.notna().to_numpy()
    if not valid.any():
        raise ValueError("没有成对的非缺失样本")
    return joint_entropy(
        x_series[valid], y_series[valid], correction, base
    ) - discrete_entropy(y_series[valid], correction, base)


def gaussian_differential_entropy(covariance: np.ndarray | float) -> float:
    """返回多元 Gaussian 的解析 differential entropy（nat）。"""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim == 0:
        matrix = matrix.reshape(1, 1)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance 必须是方阵")
    sign, log_determinant = np.linalg.slogdet(matrix)
    if sign <= 0 or not np.isfinite(log_determinant):
        raise ValueError("covariance 必须正定")
    dimension = matrix.shape[0]
    return float(
        0.5
        * (
            dimension * math.log(2.0 * math.pi * math.e)
            + log_determinant
        )
    )


def log_unit_ball_volume(dimension: int, metric: DistanceMetric) -> float:
    """返回指定范数下 d 维单位球体积的对数。"""

    if dimension < 1:
        raise ValueError("dimension 必须为正整数")
    if metric == "euclidean":
        return float(0.5 * dimension * math.log(math.pi) - gammaln(0.5 * dimension + 1.0))
    if metric == "chebyshev":
        return float(dimension * math.log(2.0))
    raise ValueError("metric 只能是 'euclidean' 或 'chebyshev'")


def _continuous_sample_matrix(samples: np.ndarray | pd.Series) -> np.ndarray:
    """把连续样本统一为 N×d 矩阵，并拒绝静默删除 NaN/inf。"""

    matrix = np.asarray(samples, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or len(matrix) < 2 or matrix.shape[1] < 1:
        raise ValueError("samples 必须是至少2行的 N×d 数值矩阵")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("kNN entropy 不接受 NaN/inf，请在调用前明确处理样本")
    return matrix


def knn_neighbor_distances(
    samples: np.ndarray | pd.Series,
    max_k: int = 5,
    metric: DistanceMetric = "euclidean",
    tie_strategy: TieStrategy = "raise",
    jitter_relative: float = 1e-10,
    random_state: int = 20260829,
) -> tuple[np.ndarray, bool]:
    """计算每个样本到第1至第 max_k 个其他样本的距离矩阵。

    普通 differential entropy 假设存在连续密度。重复点可能代表舍入或离散质量，
    默认直接拒绝；显式选择 jitter 时才加入相对于各维尺度的微小噪声。
    """

    matrix = _continuous_sample_matrix(samples).copy()
    n_samples = len(matrix)
    if not isinstance(max_k, (int, np.integer)) or not 1 <= int(max_k) < n_samples:
        raise ValueError("max_k 必须是满足 1 <= max_k < N 的整数")
    if tie_strategy not in ("raise", "jitter"):
        raise ValueError("tie_strategy 只能是 'raise' 或 'jitter'")
    has_duplicates = len(np.unique(matrix, axis=0)) < n_samples
    jittered = False
    if has_duplicates:
        if tie_strategy == "raise":
            raise ValueError("样本包含重复点；请检查离散化或显式使用 jitter")
        if not np.isfinite(jitter_relative) or jitter_relative <= 0:
            raise ValueError("jitter_relative 必须是有限正数")
        scale = np.std(matrix, axis=0, ddof=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        rng = np.random.default_rng(random_state)
        matrix += rng.normal(size=matrix.shape) * scale * jitter_relative
        jittered = True

    p_norm = 2.0 if metric == "euclidean" else np.inf if metric == "chebyshev" else None
    if p_norm is None:
        raise ValueError("metric 只能是 'euclidean' 或 'chebyshev'")
    # query 的第一列是样本自身（距离0），返回时去掉该列。
    distances, _ = cKDTree(matrix).query(
        matrix, k=int(max_k) + 1, p=p_norm, workers=1
    )
    neighbor_distances = np.asarray(distances[:, 1:], dtype=np.float64)
    if np.any(~np.isfinite(neighbor_distances)) or np.any(neighbor_distances <= 0):
        raise ValueError("近邻距离非正；样本不满足连续 kNN estimator 假设")
    return neighbor_distances, jittered


def kth_neighbor_distances(
    samples: np.ndarray | pd.Series,
    k: int = 5,
    metric: DistanceMetric = "euclidean",
    tie_strategy: TieStrategy = "raise",
    jitter_relative: float = 1e-10,
    random_state: int = 20260829,
) -> tuple[np.ndarray, bool]:
    """计算每个样本到第 k 个其他样本的距离。"""

    distances, jittered = knn_neighbor_distances(
        samples,
        max_k=k,
        metric=metric,
        tie_strategy=tie_strategy,
        jitter_relative=jitter_relative,
        random_state=random_state,
    )
    return distances[:, int(k) - 1], jittered


def entropy_from_kth_distances(
    radius: np.ndarray,
    n_samples: int,
    dimension: int,
    k: int,
    metric: DistanceMetric = "euclidean",
) -> float:
    """把第 k 邻居半径代入 Kozachenko–Leonenko 公式。"""

    distances = np.asarray(radius, dtype=np.float64).ravel()
    if len(distances) != n_samples or n_samples < 2:
        raise ValueError("radius 长度必须等于 n_samples")
    if not 1 <= k < n_samples:
        raise ValueError("k 必须满足 1 <= k < n_samples")
    if np.any(~np.isfinite(distances)) or np.any(distances <= 0):
        raise ValueError("radius 必须是有限正数")
    return float(
        digamma(n_samples)
        - digamma(k)
        + log_unit_ball_volume(dimension, metric)
        + dimension * np.mean(np.log(distances))
    )


def estimate_entropy_kl_details(
    samples: np.ndarray | pd.Series,
    k: int = 5,
    metric: DistanceMetric = "euclidean",
    tie_strategy: TieStrategy = "raise",
    jitter_relative: float = 1e-10,
    random_state: int = 20260829,
) -> KLEntropyResult:
    """使用 Kozachenko–Leonenko kNN 方法估计 differential entropy。"""

    matrix = _continuous_sample_matrix(samples)
    radius, jittered = kth_neighbor_distances(
        matrix,
        k=k,
        metric=metric,
        tie_strategy=tie_strategy,
        jitter_relative=jitter_relative,
        random_state=random_state,
    )
    n_samples, dimension = matrix.shape
    log_volume = log_unit_ball_volume(dimension, metric)
    mean_log_radius = float(np.mean(np.log(radius)))
    estimate = entropy_from_kth_distances(
        radius,
        n_samples=n_samples,
        dimension=dimension,
        k=k,
        metric=metric,
    )
    return KLEntropyResult(
        estimate=estimate,
        n_samples=n_samples,
        dimension=dimension,
        k=int(k),
        metric=metric,
        log_unit_ball_volume=log_volume,
        mean_log_radius=mean_log_radius,
        min_radius=float(np.min(radius)),
        median_radius=float(np.median(radius)),
        max_radius=float(np.max(radius)),
        jittered=jittered,
    )


def estimate_entropy_kl(
    samples: np.ndarray | pd.Series,
    k: int = 5,
    metric: DistanceMetric = "euclidean",
    tie_strategy: TieStrategy = "raise",
    jitter_relative: float = 1e-10,
    random_state: int = 20260829,
) -> float:
    """返回 Kozachenko–Leonenko differential entropy 的有限样本估计。"""

    return estimate_entropy_kl_details(
        samples,
        k=k,
        metric=metric,
        tie_strategy=tie_strategy,
        jitter_relative=jitter_relative,
        random_state=random_state,
    ).estimate
