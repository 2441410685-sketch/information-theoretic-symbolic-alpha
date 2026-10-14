"""Day 21：符号因子的受保护算子、复杂度约束与重复控制。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from .expression import (
    Call,
    EvaluationContext,
    Expression,
    ExpressionAudit,
    ExpressionRegistry,
    Literal,
    OperatorExecutor,
    Terminal,
    audit_expression,
    canonical_expression,
    structural_hash,
)


@dataclass(frozen=True, slots=True)
class OperatorPolicy:
    """冻结数值保护规则，避免不同算子各自随意处理异常值。"""

    division_absolute_epsilon: float = 1e-12
    division_relative_epsilon: float = 1e-12
    maximum_absolute_value: float = 1e100
    minimum_cross_section_size: int = 2

    def __post_init__(self) -> None:
        if self.division_absolute_epsilon <= 0:
            raise ValueError("division_absolute_epsilon 必须为正")
        if self.division_relative_epsilon < 0:
            raise ValueError("division_relative_epsilon 不能为负")
        if not np.isfinite(self.maximum_absolute_value) or self.maximum_absolute_value <= 0:
            raise ValueError("maximum_absolute_value 必须是有限正数")
        if self.minimum_cross_section_size < 2:
            raise ValueError("minimum_cross_section_size 至少为 2")


@dataclass(frozen=True, slots=True)
class ExpressionLimits:
    """在昂贵 fitness 以前应用的表达式膨胀上限。"""

    maximum_nodes: int = 31
    maximum_depth: int = 8
    maximum_required_history: int = 60
    maximum_weighted_complexity: float = 25.0

    def __post_init__(self) -> None:
        if min(self.maximum_nodes, self.maximum_depth, self.maximum_required_history) < 1:
            raise ValueError("节点、深度和历史窗口上限必须为正")
        if self.maximum_weighted_complexity <= 0:
            raise ValueError("maximum_weighted_complexity 必须为正")


@dataclass(frozen=True, slots=True)
class OutputPolicy:
    """候选输出进入 fitness 前必须满足的最小可用性。"""

    minimum_valid_fraction: float = 0.60
    minimum_valid_observations: int = 20
    minimum_unique_values: int = 2

    def __post_init__(self) -> None:
        if not 0 < self.minimum_valid_fraction <= 1:
            raise ValueError("minimum_valid_fraction 必须位于 (0, 1]")
        if self.minimum_valid_observations < 1:
            raise ValueError("minimum_valid_observations 必须为正")
        if self.minimum_unique_values < 2:
            raise ValueError("minimum_unique_values 至少为 2")


@dataclass(frozen=True, slots=True)
class ConstraintReport:
    """表达式结构约束的机器可读检查结果。"""

    accepted: bool
    reasons: tuple[str, ...]
    canonical: str
    structural_hash: str
    node_count: int | None
    depth: int | None
    required_history: int | None
    maximum_forward_offset: int | None
    weighted_complexity: float | None


@dataclass(frozen=True, slots=True)
class OutputReport:
    """因子输出的有限值覆盖率和退化状态。"""

    accepted: bool
    reasons: tuple[str, ...]
    observation_count: int
    valid_count: int
    valid_fraction: float
    nan_count: int
    inf_count: int
    unique_valid_values: int


@dataclass(frozen=True, slots=True)
class DuplicateReport:
    """Canonical duplicate 检查结果。"""

    is_new: bool
    canonical: str
    structural_hash: str
    distinct_seen_count: int


DEFAULT_OPERATOR_WEIGHTS: dict[str, float] = {
    "add": 1.0,
    "subtract": 1.0,
    "multiply": 1.0,
    "safe_div": 1.5,
    "delay": 1.5,
    "delta": 1.5,
    "ts_mean": 2.0,
    "ts_std": 2.0,
    "ts_min": 2.0,
    "ts_max": 2.0,
    "ts_rank": 2.5,
    "ts_corr": 3.0,
    "ts_cov": 3.0,
    "rank": 1.5,
}


def _as_panel(argument: object) -> pd.Series:
    """将算子输入统一为 float Series，并把非有限输入标成缺失。"""

    if not isinstance(argument, pd.Series):
        raise TypeError("该参数必须是 pandas.Series")
    values = pd.to_numeric(argument, errors="coerce").astype(float)
    array = values.to_numpy(copy=True)
    array[~np.isfinite(array)] = np.nan
    return pd.Series(array, index=argument.index, name=argument.name)


def _as_window(argument: object) -> int:
    """读取已经由表达式类型系统验证的正整数窗口。"""

    if isinstance(argument, bool) or not isinstance(argument, int):
        raise TypeError("窗口参数必须是正整数")
    if argument < 1:
        raise ValueError("窗口参数必须是正整数")
    return argument


def _finalize(
    values: pd.Series | np.ndarray,
    index: pd.Index,
    policy: OperatorPolicy,
) -> pd.Series:
    """将 overflow/inf 和数值爆炸局部转成 NaN，不污染其他样本。"""

    if isinstance(values, pd.Series):
        if not values.index.equals(index):
            values = values.reindex(index)
        array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, copy=True)
    else:
        array = np.asarray(values, dtype=float).copy()
        if len(array) != len(index):
            raise ValueError("算子输出长度必须与输入 panel 一致")
    invalid = (~np.isfinite(array)) | (np.abs(array) > policy.maximum_absolute_value)
    array[invalid] = np.nan
    return pd.Series(array, index=index)


def _ordered_work(
    context: EvaluationContext,
    **series: pd.Series,
) -> pd.DataFrame:
    """按 stock/date 稳定排序并保存原行位置，算完后可恢复输入顺序。"""

    work = pd.DataFrame(
        {
            "__position": np.arange(len(context.frame)),
            "__stock": context.frame[context.stock_column].to_numpy(),
            "__date": context.frame[context.date_column].to_numpy(),
        }
    )
    for name, values in series.items():
        clean = _as_panel(values)
        if not clean.index.equals(context.frame.index):
            raise ValueError("算子输入 Series 必须与 panel 索引一致")
        work[name] = clean.to_numpy()
    return work.sort_values(["__stock", "__date"], kind="mergesort").reset_index(drop=True)


def _restore_order(
    ordered_values: pd.Series | np.ndarray,
    ordered_work: pd.DataFrame,
    context: EvaluationContext,
    policy: OperatorPolicy,
) -> pd.Series:
    """把按时间排序后的运算结果放回调用者原有的 panel 行顺序。"""

    result = np.full(len(context.frame), np.nan, dtype=float)
    result[ordered_work["__position"].to_numpy(dtype=int)] = np.asarray(
        ordered_values, dtype=float
    )
    return _finalize(result, context.frame.index, policy)


def execute_add(
    arguments: tuple[object, ...],
    context: EvaluationContext,
    *,
    policy: OperatorPolicy = OperatorPolicy(),
) -> pd.Series:
    """逐行相加；overflow 只使对应行失效。"""

    with np.errstate(over="ignore", invalid="ignore"):
        values = _as_panel(arguments[0]) + _as_panel(arguments[1])
    return _finalize(values, context.frame.index, policy)


def execute_subtract(
    arguments: tuple[object, ...],
    context: EvaluationContext,
    *,
    policy: OperatorPolicy = OperatorPolicy(),
) -> pd.Series:
    """逐行相减；保留方向，因此不具交换律。"""

    with np.errstate(over="ignore", invalid="ignore"):
        values = _as_panel(arguments[0]) - _as_panel(arguments[1])
    return _finalize(values, context.frame.index, policy)


def execute_multiply(
    arguments: tuple[object, ...],
    context: EvaluationContext,
    *,
    policy: OperatorPolicy = OperatorPolicy(),
) -> pd.Series:
    """逐行相乘，并将局部 overflow 转成 NaN。"""

    with np.errstate(over="ignore", invalid="ignore"):
        values = _as_panel(arguments[0]) * _as_panel(arguments[1])
    return _finalize(values, context.frame.index, policy)


def execute_safe_div(
    arguments: tuple[object, ...],
    context: EvaluationContext,
    *,
    policy: OperatorPolicy = OperatorPolicy(),
) -> pd.Series:
    """尺度自适应地保护接近零的分母，并以 NaN 表示未定义。"""

    numerator = _as_panel(arguments[0])
    denominator = _as_panel(arguments[1])
    denominator_array = denominator.to_numpy()
    finite_absolute = np.abs(denominator_array[np.isfinite(denominator_array)])
    scale = float(np.median(finite_absolute)) if finite_absolute.size else 0.0
    threshold = max(
        policy.division_absolute_epsilon,
        policy.division_relative_epsilon * scale,
    )
    valid = (
        np.isfinite(numerator.to_numpy())
        & np.isfinite(denominator_array)
        & (np.abs(denominator_array) > threshold)
    )
    output = np.full(len(numerator), np.nan, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        output[valid] = numerator.to_numpy()[valid] / denominator_array[valid]
    return _finalize(output, context.frame.index, policy)


def _unary_by_stock(
    values: pd.Series,
    context: EvaluationContext,
    transform: Any,
    policy: OperatorPolicy,
) -> pd.Series:
    """对每只股票独立应用时序变换，绝不跨股票拼接窗口。"""

    work = _ordered_work(context, value=values)
    transformed = work.groupby("__stock", sort=False)["value"].transform(transform)
    return _restore_order(transformed, work, context, policy)


def execute_delay(
    arguments: tuple[object, ...],
    context: EvaluationContext,
    *,
    policy: OperatorPolicy = OperatorPolicy(),
) -> pd.Series:
    """读取同一股票之前 d 个保留交易观测，禁止负位移。"""

    values, window = _as_panel(arguments[0]), _as_window(arguments[1])
    return _unary_by_stock(values, context, lambda group: group.shift(window), policy)


def execute_delta(
    arguments: tuple[object, ...],
    context: EvaluationContext,
    *,
    policy: OperatorPolicy = OperatorPolicy(),
) -> pd.Series:
    """计算当前值减去同一股票过去 d 个保留交易观测。"""

    values, window = _as_panel(arguments[0]), _as_window(arguments[1])

    def difference(group: pd.Series) -> pd.Series:
        return group - group.shift(window)

    return _unary_by_stock(values, context, difference, policy)


def _rolling_unary(
    arguments: tuple[object, ...],
    context: EvaluationContext,
    operation: str,
    policy: OperatorPolicy,
) -> pd.Series:
    """使用包含当日的完整过去窗口执行单变量 rolling 操作。"""

    values, window = _as_panel(arguments[0]), _as_window(arguments[1])

    def transform(group: pd.Series) -> pd.Series:
        rolling = group.rolling(window, min_periods=window)
        if operation == "mean":
            return rolling.mean()
        if operation == "std":
            return rolling.std(ddof=0)
        if operation == "min":
            return rolling.min()
        if operation == "max":
            return rolling.max()
        raise KeyError(f"未知 rolling operation: {operation}")

    return _unary_by_stock(values, context, transform, policy)


def execute_ts_mean(
    arguments: tuple[object, ...],
    context: EvaluationContext,
    *,
    policy: OperatorPolicy = OperatorPolicy(),
) -> pd.Series:
    """计算包含当日的过去 w 个保留交易观测均值。"""

    return _rolling_unary(arguments, context, "mean", policy)


def execute_ts_std(
    arguments: tuple[object, ...],
    context: EvaluationContext,
    *,
    policy: OperatorPolicy = OperatorPolicy(),
) -> pd.Series:
    """计算总体口径 rolling 标准差，常数窗口得到 0。"""

    return _rolling_unary(arguments, context, "std", policy)


def execute_ts_min(
    arguments: tuple[object, ...],
    context: EvaluationContext,
    *,
    policy: OperatorPolicy = OperatorPolicy(),
) -> pd.Series:
    """计算包含当日的完整 rolling 最小值。"""

    return _rolling_unary(arguments, context, "min", policy)


def execute_ts_max(
    arguments: tuple[object, ...],
    context: EvaluationContext,
    *,
    policy: OperatorPolicy = OperatorPolicy(),
) -> pd.Series:
    """计算包含当日的完整 rolling 最大值。"""

    return _rolling_unary(arguments, context, "max", policy)


def _last_percentile_rank(window_values: np.ndarray) -> float:
    """返回窗口最后一个有限观测的平均百分位秩。"""

    if not np.isfinite(window_values).all():
        return np.nan
    last = window_values[-1]
    less = int(np.sum(window_values < last))
    equal = int(np.sum(window_values == last))
    average_rank = less + (equal + 1.0) / 2.0
    return float(average_rank / len(window_values))


def execute_ts_rank(
    arguments: tuple[object, ...],
    context: EvaluationContext,
    *,
    policy: OperatorPolicy = OperatorPolicy(),
) -> pd.Series:
    """计算当前观测在同股过去 w 个完整观测中的平均百分位秩。"""

    values, window = _as_panel(arguments[0]), _as_window(arguments[1])

    def transform(group: pd.Series) -> pd.Series:
        return group.rolling(window, min_periods=window).apply(
            _last_percentile_rank, raw=True
        )

    return _unary_by_stock(values, context, transform, policy)


def _rolling_pair(
    arguments: tuple[object, ...],
    context: EvaluationContext,
    operation: str,
    policy: OperatorPolicy,
) -> pd.Series:
    """在同一股票、同一完整窗口内计算 pairwise correlation/covariance。"""

    left = _as_panel(arguments[0])
    right = _as_panel(arguments[1])
    window = _as_window(arguments[2])
    work = _ordered_work(context, left=left, right=right)
    ordered_output = np.full(len(work), np.nan, dtype=float)
    for positions in work.groupby("__stock", sort=False).indices.values():
        group = work.iloc[positions]
        left_group = group["left"].reset_index(drop=True)
        right_group = group["right"].reset_index(drop=True)
        if operation == "corr":
            values = left_group.rolling(window, min_periods=window).corr(right_group)
        elif operation == "cov":
            values = left_group.rolling(window, min_periods=window).cov(
                right_group, ddof=0
            )
        else:
            raise KeyError(f"未知 pair rolling operation: {operation}")
        ordered_output[np.asarray(positions, dtype=int)] = values.to_numpy()
    return _restore_order(ordered_output, work, context, policy)


def execute_ts_corr(
    arguments: tuple[object, ...],
    context: EvaluationContext,
    *,
    policy: OperatorPolicy = OperatorPolicy(),
) -> pd.Series:
    """计算同股完整窗口 Pearson correlation；常数窗口返回 NaN。"""

    return _rolling_pair(arguments, context, "corr", policy)


def execute_ts_cov(
    arguments: tuple[object, ...],
    context: EvaluationContext,
    *,
    policy: OperatorPolicy = OperatorPolicy(),
) -> pd.Series:
    """计算同股完整窗口总体 covariance（ddof=0）。"""

    return _rolling_pair(arguments, context, "cov", policy)


def execute_rank(
    arguments: tuple[object, ...],
    context: EvaluationContext,
    *,
    policy: OperatorPolicy = OperatorPolicy(),
) -> pd.Series:
    """逐日计算平均百分位秩；不足两个有效股票时返回 NaN。"""

    values = _as_panel(arguments[0])
    work = pd.DataFrame(
        {
            "date": context.frame[context.date_column].to_numpy(),
            "value": values.to_numpy(),
        },
        index=context.frame.index,
    )

    def rank_one_date(group: pd.Series) -> pd.Series:
        finite = np.isfinite(group.to_numpy())
        result = pd.Series(np.nan, index=group.index, dtype=float)
        if int(finite.sum()) >= policy.minimum_cross_section_size:
            result.loc[finite] = group.loc[finite].rank(method="average", pct=True)
        return result

    ranked = work.groupby("date", sort=False, group_keys=False)["value"].apply(
        rank_one_date
    )
    ranked = ranked.reindex(context.frame.index)
    return _finalize(ranked, context.frame.index, policy)


def make_operator_executors(
    policy: OperatorPolicy | None = None,
) -> dict[str, OperatorExecutor]:
    """返回与 Day 20 registry 一一对应的完整受保护 executor mapping。"""

    selected = policy or OperatorPolicy()
    functions = {
        "add": execute_add,
        "subtract": execute_subtract,
        "multiply": execute_multiply,
        "safe_div": execute_safe_div,
        "delay": execute_delay,
        "delta": execute_delta,
        "ts_mean": execute_ts_mean,
        "ts_std": execute_ts_std,
        "ts_min": execute_ts_min,
        "ts_max": execute_ts_max,
        "ts_rank": execute_ts_rank,
        "ts_corr": execute_ts_corr,
        "ts_cov": execute_ts_cov,
        "rank": execute_rank,
    }
    return {
        name: partial(function, policy=selected)
        for name, function in functions.items()
    }


def weighted_complexity(
    expression: Expression,
    *,
    operator_weights: dict[str, float] | None = None,
    terminal_weight: float = 0.25,
    literal_weight: float = 0.10,
) -> float:
    """计算可解释的结构成本；昂贵 rolling pair 算子的权重更高。"""

    weights = operator_weights or DEFAULT_OPERATOR_WEIGHTS
    if isinstance(expression, Terminal):
        return terminal_weight
    if isinstance(expression, Literal):
        return literal_weight
    own_weight = weights.get(expression.operator)
    if own_weight is None:
        raise KeyError(f"缺少 operator complexity weight: {expression.operator}")
    return float(
        own_weight
        + sum(
            weighted_complexity(
                argument,
                operator_weights=weights,
                terminal_weight=terminal_weight,
                literal_weight=literal_weight,
            )
            for argument in expression.arguments
        )
    )


def check_expression_constraints(
    expression: Expression,
    registry: ExpressionRegistry,
    limits: ExpressionLimits = ExpressionLimits(),
) -> ConstraintReport:
    """在执行前检查结构合法性、未来引用、窗口与 bloat。"""

    try:
        audit: ExpressionAudit = audit_expression(expression, registry)
        complexity = weighted_complexity(expression)
    except (KeyError, TypeError, ValueError) as error:
        return ConstraintReport(
            accepted=False,
            reasons=(f"invalid_expression:{type(error).__name__}:{error}",),
            canonical="",
            structural_hash="",
            node_count=None,
            depth=None,
            required_history=None,
            maximum_forward_offset=None,
            weighted_complexity=None,
        )
    reasons: list[str] = []
    if audit.node_count > limits.maximum_nodes:
        reasons.append("node_limit")
    if audit.depth > limits.maximum_depth:
        reasons.append("depth_limit")
    if audit.required_history > limits.maximum_required_history:
        reasons.append("history_limit")
    if audit.maximum_forward_offset > 0:
        reasons.append("future_reference")
    if complexity > limits.maximum_weighted_complexity:
        reasons.append("weighted_complexity_limit")
    return ConstraintReport(
        accepted=not reasons,
        reasons=tuple(reasons),
        canonical=audit.canonical,
        structural_hash=audit.structural_hash,
        node_count=audit.node_count,
        depth=audit.depth,
        required_history=audit.required_history,
        maximum_forward_offset=audit.maximum_forward_offset,
        weighted_complexity=complexity,
    )


def check_factor_output(
    values: pd.Series,
    policy: OutputPolicy = OutputPolicy(),
) -> OutputReport:
    """拒绝有限覆盖不足、全常数或几乎无变化的候选输出。"""

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    valid = numeric[finite]
    count = len(numeric)
    valid_count = int(finite.sum())
    valid_fraction = valid_count / count if count else 0.0
    unique = int(pd.Series(valid).nunique(dropna=True))
    reasons: list[str] = []
    if valid_count < policy.minimum_valid_observations:
        reasons.append("too_few_valid_observations")
    if valid_fraction < policy.minimum_valid_fraction:
        reasons.append("valid_fraction_below_limit")
    if unique < policy.minimum_unique_values:
        reasons.append("constant_or_degenerate")
    return OutputReport(
        accepted=not reasons,
        reasons=tuple(reasons),
        observation_count=count,
        valid_count=valid_count,
        valid_fraction=valid_fraction,
        nan_count=int(np.isnan(numeric).sum()),
        inf_count=int(np.isinf(numeric).sum()),
        unique_valid_values=unique,
    )


class CanonicalDeduplicator:
    """用 stable structural hash 记录已见表达式，避免重复 fitness。"""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    @property
    def distinct_seen_count(self) -> int:
        """返回截至当前见过的不同 canonical expression 数。"""

        return len(self._seen)

    def register(
        self,
        expression: Expression,
        registry: ExpressionRegistry,
    ) -> DuplicateReport:
        """登记表达式；首次出现 is_new=True，之后为 duplicate。"""

        canonical = canonical_expression(expression, registry)
        fingerprint = structural_hash(expression, registry)
        is_new = fingerprint not in self._seen
        self._seen.add(fingerprint)
        return DuplicateReport(
            is_new=is_new,
            canonical=canonical,
            structural_hash=fingerprint,
            distinct_seen_count=len(self._seen),
        )
