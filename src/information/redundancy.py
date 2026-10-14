"""因子相关冗余、信息冗余与增量预测信息。"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd

from .conditional_mi import estimate_cmi_frenzel_pompe_details
from .mutual_information import estimate_mi_ksg_details


RedundancyMethod = Literal["correlation", "mutual_information"]


def gaussian_mi_equivalent(correlation: float) -> float:
    """把 Gaussian 相关阈值换算成相同强度的 MI 阈值，单位为 nat。"""

    value = float(correlation)
    if not np.isfinite(value) or not 0.0 < abs(value) < 1.0:
        raise ValueError("correlation 必须满足 0 < |correlation| < 1")
    return float(-0.5 * np.log1p(-(value**2)))


def _validate_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    minimum_count: int,
) -> list[str]:
    """检查列名存在且不重复，避免同一因子与自己构成伪冗余。"""

    names = list(columns)
    if len(names) < minimum_count or len(names) != len(set(names)):
        raise ValueError(f"columns 至少需要 {minimum_count} 个不重复列名")
    missing = set(names).difference(frame.columns)
    if missing:
        raise ValueError(f"数据缺少字段：{sorted(missing)}")
    return names


def _finite_rows(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """只保留指定列共同有限的样本，使 MI 与 CMI 使用完全相同的样本。"""

    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=np.float64)).all(axis=1)
    return frame.loc[finite].copy()


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """返回 Pearson correlation；常数或样本不足时返回 NaN。"""

    if len(x) < 3 or np.ptp(x) <= 1e-15 or np.ptp(y) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _mean_daily_correlation(
    frame: pd.DataFrame,
    factor_a: str,
    factor_b: str,
    date_col: str | None,
    min_daily_observations: int,
) -> tuple[float, float, int]:
    """计算等权日度相关均值；没有日期列时退回 pooled correlation。"""

    if date_col is None:
        value = _safe_pearson(
            frame[factor_a].to_numpy(dtype=np.float64),
            frame[factor_b].to_numpy(dtype=np.float64),
        )
        return value, value, 1

    correlations: list[float] = []
    for _, group in frame.groupby(date_col, sort=True):
        if len(group) < min_daily_observations:
            continue
        value = _safe_pearson(
            group[factor_a].to_numpy(dtype=np.float64),
            group[factor_b].to_numpy(dtype=np.float64),
        )
        if np.isfinite(value):
            correlations.append(value)
    if not correlations:
        return float("nan"), float("nan"), 0
    return (
        float(np.mean(correlations)),
        float(np.median(correlations)),
        len(correlations),
    )


def pairwise_redundancy_table(
    frame: pd.DataFrame,
    factors: Sequence[str],
    *,
    date_col: str | None = "date",
    k: int = 5,
    min_observations: int = 300,
    min_daily_observations: int = 30,
    random_state: int = 20260903,
) -> pd.DataFrame:
    """计算每对因子的日均相关和 pooled KSG MI。

    输入应已经完成研究所需的逐日 Rank Gaussianization。相关性按交易日
    等权平均，MI 使用相同日期抽样后的 pooled 样本；因此函数不会把原始
    panel 静默当作 IID，也不会再次改变调用者冻结的预处理口径。
    """

    factor_names = _validate_columns(frame, factors, minimum_count=2)
    if date_col is not None and date_col not in frame:
        raise ValueError(f"数据缺少日期字段 {date_col}")
    if min_observations < k + 1 or min_daily_observations < 3:
        raise ValueError("最小样本数与 k 的设置不合法")

    rows: list[dict[str, object]] = []
    for pair_number, (factor_a, factor_b) in enumerate(
        combinations(factor_names, 2)
    ):
        required = [factor_a, factor_b]
        if date_col is not None:
            required.append(date_col)
        valid = _finite_rows(frame, [factor_a, factor_b])
        if date_col is not None:
            valid = valid.dropna(subset=[date_col])
        n_observations = len(valid)
        if n_observations < min_observations:
            rows.append(
                {
                    "factor_a": factor_a,
                    "factor_b": factor_b,
                    "n_observations": n_observations,
                    "n_dates": 0,
                    "mean_daily_corr": np.nan,
                    "median_daily_corr": np.nan,
                    "pooled_corr": np.nan,
                    "ksg_mi": np.nan,
                    "median_joint_radius": np.nan,
                    "jittered": False,
                    "valid": False,
                    "reason": "insufficient_observations",
                }
            )
            continue

        mean_corr, median_corr, n_dates = _mean_daily_correlation(
            valid,
            factor_a,
            factor_b,
            date_col,
            min_daily_observations,
        )
        x = valid[factor_a].to_numpy(dtype=np.float64)
        y = valid[factor_b].to_numpy(dtype=np.float64)
        pooled_corr = _safe_pearson(x, y)
        try:
            mi = estimate_mi_ksg_details(
                x,
                y,
                k=k,
                tie_strategy="jitter",
                random_state=random_state + pair_number * 10_007,
            )
            rows.append(
                {
                    "factor_a": factor_a,
                    "factor_b": factor_b,
                    "n_observations": n_observations,
                    "n_dates": n_dates,
                    "mean_daily_corr": mean_corr,
                    "median_daily_corr": median_corr,
                    "pooled_corr": pooled_corr,
                    "ksg_mi": mi.estimate,
                    "median_joint_radius": mi.median_joint_radius,
                    "jittered": mi.jittered,
                    "valid": True,
                    "reason": "",
                }
            )
        except ValueError as error:
            rows.append(
                {
                    "factor_a": factor_a,
                    "factor_b": factor_b,
                    "n_observations": n_observations,
                    "n_dates": n_dates,
                    "mean_daily_corr": mean_corr,
                    "median_daily_corr": median_corr,
                    "pooled_corr": pooled_corr,
                    "ksg_mi": np.nan,
                    "median_joint_radius": np.nan,
                    "jittered": False,
                    "valid": False,
                    "reason": str(error),
                }
            )
    return pd.DataFrame(rows)


def redundancy_matrix(
    pairwise: pd.DataFrame,
    factors: Sequence[str],
    metric: Literal["mean_daily_corr", "ksg_mi"],
) -> pd.DataFrame:
    """把 long pair table 转成对称矩阵；相关对角为1，MI 对角留空。"""

    factor_names = list(factors)
    required = {"factor_a", "factor_b", metric}
    if missing := required.difference(pairwise.columns):
        raise ValueError(f"pairwise 缺少字段：{sorted(missing)}")
    values = np.full((len(factor_names), len(factor_names)), np.nan, dtype=float)
    if metric == "mean_daily_corr":
        np.fill_diagonal(values, 1.0)
    matrix = pd.DataFrame(
        values,
        index=factor_names,
        columns=factor_names,
    )
    for row in pairwise.itertuples(index=False):
        if row.factor_a not in matrix.index or row.factor_b not in matrix.index:
            continue
        value = float(getattr(row, metric))
        matrix.loc[row.factor_a, row.factor_b] = value
        matrix.loc[row.factor_b, row.factor_a] = value
    return matrix


def prune_by_redundancy(
    pairwise: pd.DataFrame,
    ordered_factors: Sequence[str],
    *,
    method: RedundancyMethod,
    threshold: float,
) -> pd.DataFrame:
    """按固定优先顺序贪心去冗余，并完整记录每个删除决定。"""

    factors = list(ordered_factors)
    if not factors or len(factors) != len(set(factors)):
        raise ValueError("ordered_factors 必须非空且不重复")
    if method not in ("correlation", "mutual_information"):
        raise ValueError("未知冗余方法")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold 必须是有限正数")
    metric = "mean_daily_corr" if method == "correlation" else "ksg_mi"
    required = {"factor_a", "factor_b", metric}
    if missing := required.difference(pairwise.columns):
        raise ValueError(f"pairwise 缺少字段：{sorted(missing)}")

    lookup: dict[frozenset[str], float] = {}
    for row in pairwise.itertuples(index=False):
        value = float(getattr(row, metric))
        if np.isfinite(value):
            lookup[frozenset((str(row.factor_a), str(row.factor_b)))] = value

    kept: list[str] = []
    rows: list[dict[str, object]] = []
    for priority, factor in enumerate(factors, start=1):
        conflicts: list[tuple[str, float]] = []
        for previous in kept:
            key = frozenset((factor, previous))
            if key not in lookup:
                raise ValueError(f"缺少因子对：{factor}, {previous}")
            raw_value = lookup[key]
            comparison = abs(raw_value) if method == "correlation" else raw_value
            if comparison >= threshold:
                conflicts.append((previous, raw_value))
        if conflicts:
            redundant_with, value = max(
                conflicts,
                key=lambda item: abs(item[1]) if method == "correlation" else item[1],
            )
            keep = False
        else:
            redundant_with, value = "", np.nan
            keep = True
            kept.append(factor)
        rows.append(
            {
                "factor": factor,
                "priority": priority,
                "method": method,
                "threshold": threshold,
                "kept": keep,
                "redundant_with": redundant_with,
                "redundancy_value": value,
            }
        )
    return pd.DataFrame(rows)


def incremental_predictive_information(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    label: str,
    existing_factors: Sequence[str],
    *,
    k: int = 5,
    min_observations: int = 500,
    random_state: int = 20260903,
) -> pd.DataFrame:
    """估计候选的边际 MI 与给定现有集合后的 Incremental Information。

    对每个候选都先取 candidate、label、existing 的共同完整样本，因此
    marginal MI 与 CMI 的差异不会来自样本口径变化。返回原始有限样本
    估计，不会把负 CMI 静默裁成零。
    """

    candidate_names = _validate_columns(frame, candidates, minimum_count=1)
    existing = list(existing_factors)
    if len(existing) != len(set(existing)):
        raise ValueError("existing_factors 不能重复")
    missing_existing = set(existing).difference(frame.columns)
    if label not in frame or missing_existing:
        absent = ([label] if label not in frame else []) + sorted(missing_existing)
        raise ValueError(f"数据缺少字段：{absent}")
    if min_observations < k + 1:
        raise ValueError("min_observations 必须至少为 k+1")

    rows: list[dict[str, object]] = []
    for candidate_number, candidate in enumerate(candidate_names):
        common_columns = [candidate, label, *existing]
        common_columns = list(dict.fromkeys(common_columns))
        valid = _finite_rows(frame, common_columns)
        n_observations = len(valid)
        common = {
            "factor": candidate,
            "label": label,
            "existing_factors": ",".join(existing),
            "conditioning_dimension": len(existing),
            "n_observations": n_observations,
            "k": k,
        }
        if n_observations < min_observations:
            rows.append(
                {
                    **common,
                    "corr_with_label": np.nan,
                    "marginal_mi": np.nan,
                    "incremental_cmi": np.nan,
                    "incremental_fraction": np.nan,
                    "median_cmi_radius": np.nan,
                    "status": "insufficient_observations",
                }
            )
            continue

        x = valid[candidate].to_numpy(dtype=np.float64)
        y = valid[label].to_numpy(dtype=np.float64)
        seed = random_state + candidate_number * 100_003
        marginal = estimate_mi_ksg_details(
            x,
            y,
            k=k,
            tie_strategy="jitter",
            random_state=seed,
        )
        if candidate in existing:
            incremental = np.nan
            radius = np.nan
            status = "conditioning_member"
        elif existing:
            z = valid[existing].to_numpy(dtype=np.float64)
            details = estimate_cmi_frenzel_pompe_details(
                x,
                y,
                z,
                k=k,
                tie_strategy="jitter",
                random_state=seed + 1,
            )
            incremental = details.estimate
            radius = details.median_joint_radius
            status = "estimated"
        else:
            incremental = marginal.estimate
            radius = marginal.median_joint_radius
            status = "unconditional"
        fraction = (
            float(incremental / marginal.estimate)
            if np.isfinite(incremental) and marginal.estimate > 0.0
            else np.nan
        )
        rows.append(
            {
                **common,
                "corr_with_label": _safe_pearson(x, y),
                "marginal_mi": marginal.estimate,
                "incremental_cmi": incremental,
                "incremental_fraction": fraction,
                "median_cmi_radius": radius,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def summarize_retained_redundancy(
    pairwise: pd.DataFrame,
    decisions: pd.DataFrame,
) -> dict[str, float | int]:
    """汇总保留集合内部的最大相关和 MI，不把单一删除数当作质量。"""

    required_decisions = {"factor", "kept"}
    if missing := required_decisions.difference(decisions.columns):
        raise ValueError(f"decisions 缺少字段：{sorted(missing)}")
    required_pairs = {"factor_a", "factor_b", "mean_daily_corr", "ksg_mi"}
    if missing := required_pairs.difference(pairwise.columns):
        raise ValueError(f"pairwise 缺少字段：{sorted(missing)}")
    kept = set(decisions.loc[decisions["kept"], "factor"].astype(str))
    inside = pairwise.loc[
        pairwise["factor_a"].isin(kept) & pairwise["factor_b"].isin(kept)
    ]
    return {
        "retained_count": len(kept),
        "retained_pair_count": len(inside),
        "max_abs_corr": (
            float(inside["mean_daily_corr"].abs().max()) if not inside.empty else np.nan
        ),
        "mean_pairwise_mi": (
            float(inside["ksg_mi"].mean()) if not inside.empty else np.nan
        ),
        "max_pairwise_mi": (
            float(inside["ksg_mi"].max()) if not inside.empty else np.nan
        ),
    }
