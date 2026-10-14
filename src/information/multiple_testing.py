"""候选因子家族的多重检验工具。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BHResult:
    """Benjamini–Hochberg 检验结果，顺序与输入假设完全一致。"""

    adjusted_p_values: np.ndarray
    rejected: np.ndarray
    alpha: float
    hypothesis_count: int
    critical_rank: int
    critical_p_value: float


def bonferroni_min_p(*p_values: float) -> float:
    """将多个预先声明的 endpoint 合并成 Bonferroni omnibus p-value。"""

    if not p_values:
        raise ValueError("至少需要一个 p-value")
    values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-value 必须是 [0,1] 内的有限数")
    return float(min(1.0, len(values) * float(values.min())))


def benjamini_hochberg(
    p_values: np.ndarray | list[float],
    *,
    alpha: float = 0.10,
) -> BHResult:
    """执行 BH step-up procedure，并返回单调的 adjusted p-values。

    设排序后的 p-value 为 ``p_(1) <= ... <= p_(m)``，找到最大的
    ``r`` 使 ``p_(r) <= r * alpha / m``，并拒绝前 ``r`` 个原假设。
    adjusted p-value 使用反向累计最小值，随后映射回原输入顺序。
    """

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("p_values 必须是一维非空数组")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-value 必须是 [0,1] 内的有限数")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha 必须位于 (0,1)")

    count = len(values)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.arange(1, count + 1, dtype=float)
    passed = ordered <= ranks * alpha / count
    critical_rank = int(np.flatnonzero(passed)[-1] + 1) if passed.any() else 0
    critical_p = float(ordered[critical_rank - 1]) if critical_rank else np.nan

    ordered_adjusted = ordered * count / ranks
    ordered_adjusted = np.minimum.accumulate(ordered_adjusted[::-1])[::-1]
    ordered_adjusted = np.clip(ordered_adjusted, 0.0, 1.0)
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = ordered_adjusted
    rejected = adjusted <= alpha
    return BHResult(
        adjusted_p_values=adjusted,
        rejected=rejected,
        alpha=float(alpha),
        hypothesis_count=count,
        critical_rank=critical_rank,
        critical_p_value=critical_p,
    )
