"""离散 Mutual Information 与 Histogram MI 的从零实现。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import digamma

from .entropy import Correction, TieStrategy, entropy_from_counts


BinStrategy = Literal["equal_width", "quantile"]


@dataclass(frozen=True)
class HistogramMIResult:
    """保存有限样本估计及分箱稀疏性，防止只看一个 MI 数字。"""

    estimate: float
    plugin_estimate: float
    correction: Correction
    correction_amount: float
    n_samples: int
    requested_x_bins: int
    requested_y_bins: int
    effective_x_bins: int
    effective_y_bins: int
    occupied_joint_bins: int
    possible_joint_bins: int
    occupancy_ratio: float


@dataclass(frozen=True)
class KSGMIResult:
    """保存 KSG-1 有限样本估计及近邻计数诊断。"""

    estimate: float
    n_samples: int
    x_dimension: int
    y_dimension: int
    k: int
    metric: str
    min_joint_radius: float
    median_joint_radius: float
    max_joint_radius: float
    mean_x_neighbors: float
    mean_y_neighbors: float
    max_joint_boundary_ties: int
    jittered: bool


def _paired_finite(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """连续估计只使用成对有限样本；NaN/inf 不伪装成一个 bin。"""

    x_array = np.asarray(x, dtype=np.float64).ravel()
    y_array = np.asarray(y, dtype=np.float64).ravel()
    if len(x_array) != len(y_array):
        raise ValueError("x 与 y 的样本长度必须一致")
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    if valid.sum() < 2:
        raise ValueError("至少需要2个成对有限样本")
    return x_array[valid], y_array[valid]


def contingency_table(
    x_codes: np.ndarray,
    y_codes: np.ndarray,
    x_states: int | None = None,
    y_states: int | None = None,
) -> np.ndarray:
    """由从0开始的离散代码构造二维频数表。"""

    x_array = np.asarray(x_codes, dtype=np.int64).ravel()
    y_array = np.asarray(y_codes, dtype=np.int64).ravel()
    if len(x_array) != len(y_array):
        raise ValueError("x_codes 与 y_codes 长度必须一致")
    valid = (x_array >= 0) & (y_array >= 0)
    if not valid.any():
        raise ValueError("没有有效的联合状态")
    nx = int(x_states) if x_states is not None else int(x_array[valid].max()) + 1
    ny = int(y_states) if y_states is not None else int(y_array[valid].max()) + 1
    if nx <= 0 or ny <= 0:
        raise ValueError("状态数必须为正整数")
    if np.any(x_array[valid] >= nx) or np.any(y_array[valid] >= ny):
        raise ValueError("离散代码超出指定状态数")
    flat = x_array[valid] * ny + y_array[valid]
    return np.bincount(flat, minlength=nx * ny).reshape(nx, ny)


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    base: float = math.e,
) -> float:
    """计算 D_KL(P||Q)；若 P 在 Q 的零概率处有质量则返回正无穷。"""

    p_array = np.asarray(p, dtype=np.float64).ravel()
    q_array = np.asarray(q, dtype=np.float64).ravel()
    if p_array.shape != q_array.shape:
        raise ValueError("p 与 q 的形状必须一致")
    if np.any(~np.isfinite(p_array)) or np.any(~np.isfinite(q_array)):
        raise ValueError("p 与 q 必须有限")
    if np.any(p_array < 0) or np.any(q_array < 0):
        raise ValueError("概率不能为负")
    if p_array.sum() <= 0 or q_array.sum() <= 0:
        raise ValueError("概率总和必须为正")
    p_probability = p_array / p_array.sum()
    q_probability = q_array / q_array.sum()
    support = p_probability > 0
    if np.any(q_probability[support] == 0):
        return float("inf")
    log_base = math.log(base)
    if not np.isfinite(log_base) or np.isclose(log_base, 0.0):
        raise ValueError("对数底必须为正数且不能等于1")
    return float(
        np.sum(
            p_probability[support]
            * np.log(p_probability[support] / q_probability[support])
        )
        / log_base
    )


def mutual_information_from_counts(
    counts: np.ndarray,
    correction: Correction = "none",
    base: float = math.e,
) -> float:
    """由列联表计算 MI；Miller–Madow 通过三个熵的修正组合得到。"""

    table = np.asarray(counts, dtype=np.float64)
    if table.ndim != 2 or np.any(~np.isfinite(table)) or np.any(table < 0):
        raise ValueError("counts 必须是二维有限非负频数表")
    if table.sum() <= 0:
        raise ValueError("联合频数总和必须为正")
    x_counts = table.sum(axis=1)
    y_counts = table.sum(axis=0)
    return float(
        entropy_from_counts(x_counts, correction, base)
        + entropy_from_counts(y_counts, correction, base)
        - entropy_from_counts(table.ravel(), correction, base)
    )


def discrete_mutual_information(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    correction: Correction = "none",
    base: float = math.e,
) -> float:
    """把任意离散标签编码后估计 MI。"""

    x_series = pd.Series(x, dtype="object")
    y_series = pd.Series(y, dtype="object")
    if len(x_series) != len(y_series):
        raise ValueError("x 与 y 的样本长度必须一致")
    valid = x_series.notna().to_numpy() & y_series.notna().to_numpy()
    if not valid.any():
        raise ValueError("没有成对的非缺失样本")
    x_codes, _ = pd.factorize(x_series[valid], sort=False)
    y_codes, _ = pd.factorize(y_series[valid], sort=False)
    table = contingency_table(x_codes, y_codes)
    return mutual_information_from_counts(table, correction, base)


def fit_histogram_edges(
    values: np.ndarray | pd.Series,
    bins: int,
    strategy: BinStrategy = "equal_width",
) -> np.ndarray:
    """拟合一维分箱边界；边界可在训练窗保存并应用到未来数据。"""

    array = np.asarray(values, dtype=np.float64).ravel()
    finite = array[np.isfinite(array)]
    if bins < 2:
        raise ValueError("bins 至少为2")
    if finite.size < 2:
        raise ValueError("拟合分箱至少需要2个有限样本")
    if np.ptp(finite) <= 1e-15:
        return np.array([-np.inf, np.inf])
    if strategy == "equal_width":
        interior = np.linspace(finite.min(), finite.max(), bins + 1)[1:-1]
    elif strategy == "quantile":
        interior = np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    else:
        raise ValueError("strategy 只能是 'equal_width' 或 'quantile'")
    # ties 会产生重复分位点；去重意味着有效 bin 数可以小于请求值。
    interior = np.unique(interior[np.isfinite(interior)])
    return np.r_[-np.inf, interior, np.inf]


def apply_histogram_edges(
    values: np.ndarray | pd.Series,
    edges: np.ndarray,
) -> np.ndarray:
    """应用已拟合边界；有限样本映射到0开始的 bin，缺失值映射为-1。"""

    array = np.asarray(values, dtype=np.float64).ravel()
    edge_array = np.asarray(edges, dtype=np.float64).ravel()
    if len(edge_array) < 2 or not np.all(np.diff(edge_array) > 0):
        raise ValueError("edges 必须严格递增且至少包含两个端点")
    codes = np.full(len(array), -1, dtype=np.int64)
    finite = np.isfinite(array)
    codes[finite] = np.searchsorted(edge_array[1:-1], array[finite], side="right")
    return codes


def histogram_mutual_information(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    bins: int | tuple[int, int] = 8,
    strategy: BinStrategy = "equal_width",
    correction: Correction = "none",
    base: float = math.e,
    x_edges: np.ndarray | None = None,
    y_edges: np.ndarray | None = None,
) -> HistogramMIResult:
    """通过一维分箱和联合列联表估计连续变量的 Histogram MI。"""

    x_array, y_array = _paired_finite(x, y)
    requested_x, requested_y = (bins, bins) if isinstance(bins, int) else bins
    if requested_x < 2 or requested_y < 2:
        raise ValueError("两个变量的 bins 都必须至少为2")
    fitted_x_edges = (
        fit_histogram_edges(x_array, requested_x, strategy)
        if x_edges is None
        else np.asarray(x_edges, dtype=np.float64)
    )
    fitted_y_edges = (
        fit_histogram_edges(y_array, requested_y, strategy)
        if y_edges is None
        else np.asarray(y_edges, dtype=np.float64)
    )
    x_codes = apply_histogram_edges(x_array, fitted_x_edges)
    y_codes = apply_histogram_edges(y_array, fitted_y_edges)
    effective_x = len(fitted_x_edges) - 1
    effective_y = len(fitted_y_edges) - 1
    table = contingency_table(
        x_codes,
        y_codes,
        x_states=effective_x,
        y_states=effective_y,
    )
    plugin = mutual_information_from_counts(table, correction="none", base=base)
    estimate = mutual_information_from_counts(table, correction=correction, base=base)
    possible = int(effective_x * effective_y)
    occupied = int(np.count_nonzero(table))
    return HistogramMIResult(
        estimate=estimate,
        plugin_estimate=plugin,
        correction=correction,
        correction_amount=estimate - plugin,
        n_samples=len(x_array),
        requested_x_bins=int(requested_x),
        requested_y_bins=int(requested_y),
        effective_x_bins=effective_x,
        effective_y_bins=effective_y,
        occupied_joint_bins=occupied,
        possible_joint_bins=possible,
        occupancy_ratio=occupied / possible,
    )


def gaussian_mutual_information(
    covariance: np.ndarray,
    x_dimension: int = 1,
) -> float:
    """由联合 Gaussian 协方差矩阵计算解析 Mutual Information（nat）。

    covariance 的前 ``x_dimension`` 个坐标属于 X，其余坐标属于 Y。
    该解析值用于验证 estimator，而不是用 Gaussian 假设替代真实金融 MI。
    """

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance 必须是方阵")
    if np.any(~np.isfinite(matrix)) or not np.allclose(matrix, matrix.T):
        raise ValueError("covariance 必须是有限对称矩阵")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("covariance 及其边际协方差必须正定") from error
    dimension = matrix.shape[0]
    if not isinstance(x_dimension, (int, np.integer)) or not 1 <= int(x_dimension) < dimension:
        raise ValueError("x_dimension 必须把联合矩阵分成非空 X 与 Y 两部分")
    x_dimension = int(x_dimension)
    x_covariance = matrix[:x_dimension, :x_dimension]
    y_covariance = matrix[x_dimension:, x_dimension:]
    signs_and_logs = (
        np.linalg.slogdet(x_covariance),
        np.linalg.slogdet(y_covariance),
        np.linalg.slogdet(matrix),
    )
    if any(sign <= 0 or not np.isfinite(logdet) for sign, logdet in signs_and_logs):
        raise ValueError("covariance 及其边际协方差必须正定")
    logdet_x = signs_and_logs[0][1]
    logdet_y = signs_and_logs[1][1]
    logdet_joint = signs_and_logs[2][1]
    return float(0.5 * (logdet_x + logdet_y - logdet_joint))


def _continuous_matrix(values: np.ndarray | pd.Series, name: str) -> np.ndarray:
    """把连续变量整理为 N×d，并拒绝静默删除非有限样本。"""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or len(matrix) < 2 or matrix.shape[1] < 1:
        raise ValueError(f"{name} 必须是至少2行的 N×d 数值矩阵")
    if np.any(~np.isfinite(matrix)):
        raise ValueError(f"KSG 不接受 {name} 中的 NaN/inf，请在调用前明确处理")
    # 常数坐标没有连续密度；用 jitter 把常数伪装成连续信号没有统计意义。
    if np.any(np.ptp(matrix, axis=0) <= 1e-15):
        raise ValueError(f"{name} 包含常数坐标，不能使用连续 KSG estimator")
    return matrix


def _prepare_ksg_pair(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    k: int,
    tie_strategy: TieStrategy,
    jitter_relative: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """验证 KSG 输入，并在用户显式允许时对精确 ties 加微小噪声。"""

    x_matrix = _continuous_matrix(x, "x").copy()
    y_matrix = _continuous_matrix(y, "y").copy()
    if len(x_matrix) != len(y_matrix):
        raise ValueError("x 与 y 的样本长度必须一致")
    n_samples = len(x_matrix)
    if not isinstance(k, (int, np.integer)) or not 1 <= int(k) < n_samples:
        raise ValueError("k 必须是满足 1 <= k < N 的整数")
    if tie_strategy not in ("raise", "jitter"):
        raise ValueError("tie_strategy 只能是 'raise' 或 'jitter'")

    # 连续模型下精确重复的 X 或 Y 观测应为零概率事件；它通常意味着价格刻度、
    # rank ties 或离散变量。默认拒绝可防止距离边界被数据顺序任意决定。
    has_ties = (
        len(np.unique(x_matrix, axis=0)) < n_samples
        or len(np.unique(y_matrix, axis=0)) < n_samples
    )
    jittered = False
    if has_ties:
        if tie_strategy == "raise":
            raise ValueError("x 或 y 包含精确 ties；请检查离散化或显式使用 jitter")
        if not np.isfinite(jitter_relative) or jitter_relative <= 0.0:
            raise ValueError("jitter_relative 必须是有限正数")
        joint = np.column_stack([x_matrix, y_matrix])
        scale = np.std(joint, axis=0, ddof=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        rng = np.random.default_rng(random_state)
        joint += rng.normal(size=joint.shape) * scale * jitter_relative
        x_matrix = joint[:, : x_matrix.shape[1]]
        y_matrix = joint[:, x_matrix.shape[1] :]
        jittered = True
    return x_matrix, y_matrix, jittered


def estimate_mi_ksg_details(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    k: int = 5,
    tie_strategy: TieStrategy = "raise",
    jitter_relative: float = 1e-10,
    random_state: int = 20260830,
) -> KSGMIResult:
    """使用 KSG-1 k近邻方法估计连续变量间的 Mutual Information。

    联合空间采用 Chebyshev 距离。对每个样本先取得第 k 联合近邻半径
    ``epsilon_i``，再分别统计 X、Y 边际空间中严格满足距离小于该半径的
    邻居数。函数保留原始有限样本估计，即使结果略小于0也不会裁剪。
    """

    x_matrix, y_matrix, jittered = _prepare_ksg_pair(
        x,
        y,
        k=k,
        tie_strategy=tie_strategy,
        jitter_relative=jitter_relative,
        random_state=random_state,
    )
    n_samples = len(x_matrix)
    k = int(k)
    joint = np.column_stack([x_matrix, y_matrix])
    joint_tree = cKDTree(joint)

    # query 的第一列是样本自身；第 k 个其他样本位于第 k+1 列。
    joint_distances, _ = joint_tree.query(
        joint,
        k=k + 1,
        p=np.inf,
        workers=1,
    )
    radius = np.asarray(joint_distances[:, k], dtype=np.float64)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("联合近邻半径非正；样本不满足连续 KSG 假设")

    # KSG-1 公式要求边际计数使用严格不等式。nextafter 把查询半径移到
    # epsilon 的下一个浮点数，从而不会把联合球边界点错误计入两个边际。
    strict_radius = np.nextafter(radius, 0.0)
    x_neighbors = np.asarray(
        cKDTree(x_matrix).query_ball_point(
            x_matrix,
            r=strict_radius,
            p=np.inf,
            workers=1,
            return_length=True,
        ),
        dtype=np.int64,
    ) - 1
    y_neighbors = np.asarray(
        cKDTree(y_matrix).query_ball_point(
            y_matrix,
            r=strict_radius,
            p=np.inf,
            workers=1,
            return_length=True,
        ),
        dtype=np.int64,
    ) - 1
    if np.any(x_neighbors < 0) or np.any(y_neighbors < 0):
        raise RuntimeError("边际近邻计数出现负数")

    estimate = float(
        digamma(k)
        + digamma(n_samples)
        - np.mean(digamma(x_neighbors + 1) + digamma(y_neighbors + 1))
    )

    # 正常连续样本通常只有第 k 个邻居位于联合边界；更大的数提示距离 ties。
    inclusive_joint = np.asarray(
        joint_tree.query_ball_point(
            joint,
            r=radius,
            p=np.inf,
            workers=1,
            return_length=True,
        ),
        dtype=np.int64,
    ) - 1
    strict_joint = np.asarray(
        joint_tree.query_ball_point(
            joint,
            r=strict_radius,
            p=np.inf,
            workers=1,
            return_length=True,
        ),
        dtype=np.int64,
    ) - 1
    boundary_ties = inclusive_joint - strict_joint

    return KSGMIResult(
        estimate=estimate,
        n_samples=n_samples,
        x_dimension=x_matrix.shape[1],
        y_dimension=y_matrix.shape[1],
        k=k,
        metric="chebyshev",
        min_joint_radius=float(np.min(radius)),
        median_joint_radius=float(np.median(radius)),
        max_joint_radius=float(np.max(radius)),
        mean_x_neighbors=float(np.mean(x_neighbors)),
        mean_y_neighbors=float(np.mean(y_neighbors)),
        max_joint_boundary_ties=int(np.max(boundary_ties)),
        jittered=jittered,
    )


def estimate_mi_ksg(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    k: int = 5,
    tie_strategy: TieStrategy = "raise",
    jitter_relative: float = 1e-10,
    random_state: int = 20260830,
) -> float:
    """返回 KSG-1 Mutual Information 的原始有限样本估计（nat）。"""

    return estimate_mi_ksg_details(
        x,
        y,
        k=k,
        tie_strategy=tie_strategy,
        jitter_relative=jitter_relative,
        random_state=random_state,
    ).estimate
