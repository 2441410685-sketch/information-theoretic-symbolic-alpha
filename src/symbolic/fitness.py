"""Day 22：日度横截面 RankIC fitness 与换手率统计。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.information.conditional_mi import estimate_cmi_frenzel_pompe_details
from src.information.mutual_information import estimate_mi_ksg_details
from src.information.panel_mi import transform_panel_by_date

from .operators import OutputPolicy, check_factor_output


@dataclass(frozen=True, slots=True)
class RankICFitnessOutcome:
    """单个表达式在一个时间区间上的 RankIC fitness 摘要。"""

    accepted: bool
    rejection_reason: str
    fitness: float
    mean_rank_ic: float
    absolute_mean_rank_ic: float
    rank_ic_std: float
    rank_ic_ir: float
    positive_ratio: float
    date_count: int
    observation_count: int
    valid_fraction: float
    direction: int
    evaluation_seconds: float = 0.0
    semantic_code: str = ""


@dataclass(frozen=True, slots=True)
class MIFitnessOutcome:
    """Pooled KSG MI fitness 摘要。

    ``stability_weight=0`` 且 ``stability_penalty=0`` 表示纯 MI fitness；
    非零值保留给 Day 23 的稳定性惩罚消融实验。
    """

    accepted: bool
    rejection_reason: str
    fitness: float
    mi_estimate: float
    stability_penalty: float
    stability_weight: float
    block_mi_mean: float
    block_mi_std: float
    block_mi_min: float
    block_positive_ratio: float
    gaussian_equivalent_abs_correlation: float
    date_count: int
    block_count: int
    observation_count: int
    valid_fraction: float
    k: int
    jittered: bool
    evaluation_seconds: float = 0.0
    semantic_code: str = ""


@dataclass(frozen=True, slots=True)
class CMIFitnessOutcome:
    """Day 24 的增量信息、稳定性、复杂度与换手率 fitness 摘要。

    所有信息量和惩罚最终都以 nat 为单位。复杂度与换手率先转换成
    无量纲的 ``[0, 1]`` 量，再乘以预先冻结的 nat 权重，避免把不同
    尺度的原始量直接相加。
    """

    accepted: bool
    rejection_reason: str
    fitness: float
    cmi_estimate: float
    stability_penalty: float
    stability_weight: float
    block_cmi_mean: float
    block_cmi_std: float
    block_cmi_min: float
    block_positive_ratio: float
    complexity_penalty: float
    complexity_weight: float
    normalized_complexity: float
    turnover_penalty: float
    turnover_weight: float
    mean_rank_turnover: float
    conditioning_dimension: int
    conditioning_factors: str
    date_count: int
    block_count: int
    observation_count: int
    valid_fraction: float
    k: int
    jittered: bool
    evaluation_seconds: float = 0.0


def daily_rank_ic(
    factor: pd.Series,
    frame: pd.DataFrame,
    *,
    label_column: str = "label_1d",
    date_column: str = "date",
    minimum_cross_section: int = 30,
) -> pd.DataFrame:
    """逐日计算 Spearman RankIC，不把不同日期 flatten 成一个相关系数。"""

    if len(factor) != len(frame) or not factor.index.equals(frame.index):
        raise ValueError("factor 必须与 frame 长度和索引完全一致")
    required = {date_column, label_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"frame 缺少列: {sorted(missing)}")
    if minimum_cross_section < 3:
        raise ValueError("minimum_cross_section 至少为 3")

    work = pd.DataFrame(
        {
            "date": frame[date_column].to_numpy(),
            "factor": pd.to_numeric(factor, errors="coerce").to_numpy(),
            "label": pd.to_numeric(frame[label_column], errors="coerce").to_numpy(),
        }
    )
    rows: list[dict[str, object]] = []
    for date, group in work.groupby("date", sort=True):
        finite = np.isfinite(group["factor"]) & np.isfinite(group["label"])
        valid = group.loc[finite, ["factor", "label"]]
        if len(valid) < minimum_cross_section:
            rows.append(
                {
                    "date": date,
                    "rank_ic": np.nan,
                    "observation_count": len(valid),
                    "valid": False,
                }
            )
            continue
        factor_rank = valid["factor"].rank(method="average")
        label_rank = valid["label"].rank(method="average")
        correlation = factor_rank.corr(label_rank)
        rows.append(
            {
                "date": date,
                "rank_ic": float(correlation) if np.isfinite(correlation) else np.nan,
                "observation_count": len(valid),
                "valid": bool(np.isfinite(correlation)),
            }
        )
    return pd.DataFrame(rows)


def summarize_rank_ic(
    daily: pd.DataFrame,
    *,
    minimum_dates: int = 20,
    observation_count: int,
    valid_fraction: float,
) -> RankICFitnessOutcome:
    """用绝对的时间均值 RankIC 作为 baseline fitness，而非日绝对值均值。"""

    valid = pd.to_numeric(
        daily.loc[daily["valid"], "rank_ic"], errors="coerce"
    ).dropna()
    if len(valid) < minimum_dates:
        return invalid_rank_ic_outcome(
            "too_few_valid_dates",
            observation_count=observation_count,
            valid_fraction=valid_fraction,
            date_count=len(valid),
        )
    mean_ic = float(valid.mean())
    std_ic = float(valid.std(ddof=1)) if len(valid) > 1 else np.nan
    icir = mean_ic / std_ic if np.isfinite(std_ic) and std_ic > 0 else np.nan
    direction = 1 if mean_ic > 0 else -1 if mean_ic < 0 else 0
    return RankICFitnessOutcome(
        accepted=True,
        rejection_reason="",
        fitness=abs(mean_ic),
        mean_rank_ic=mean_ic,
        absolute_mean_rank_ic=abs(mean_ic),
        rank_ic_std=std_ic,
        rank_ic_ir=float(icir),
        positive_ratio=float((valid > 0).mean()),
        date_count=len(valid),
        observation_count=observation_count,
        valid_fraction=valid_fraction,
        direction=direction,
    )


def evaluate_rank_ic_fitness(
    factor: pd.Series,
    frame: pd.DataFrame,
    *,
    label_column: str = "label_1d",
    date_column: str = "date",
    minimum_cross_section: int = 50,
    minimum_dates: int = 60,
    output_policy: OutputPolicy | None = None,
) -> tuple[RankICFitnessOutcome, pd.DataFrame]:
    """先执行输出 gate，再计算每日 RankIC 和纯 RankIC baseline fitness。"""

    policy = output_policy or OutputPolicy(
        minimum_valid_fraction=0.60,
        minimum_valid_observations=2_000,
        minimum_unique_values=2,
    )
    output = check_factor_output(factor, policy)
    if not output.accepted:
        reason = ",".join(output.reasons)
        return (
            invalid_rank_ic_outcome(
                reason,
                observation_count=output.valid_count,
                valid_fraction=output.valid_fraction,
            ),
            pd.DataFrame(
                columns=["date", "rank_ic", "observation_count", "valid"]
            ),
        )
    daily = daily_rank_ic(
        factor,
        frame,
        label_column=label_column,
        date_column=date_column,
        minimum_cross_section=minimum_cross_section,
    )
    return (
        summarize_rank_ic(
            daily,
            minimum_dates=minimum_dates,
            observation_count=output.valid_count,
            valid_fraction=output.valid_fraction,
        ),
        daily,
    )


def invalid_rank_ic_outcome(
    reason: str,
    *,
    observation_count: int = 0,
    valid_fraction: float = 0.0,
    date_count: int = 0,
) -> RankICFitnessOutcome:
    """为结构或输出失败候选生成统一的负无穷 fitness。"""

    return RankICFitnessOutcome(
        accepted=False,
        rejection_reason=reason,
        fitness=float("-inf"),
        mean_rank_ic=np.nan,
        absolute_mean_rank_ic=np.nan,
        rank_ic_std=np.nan,
        rank_ic_ir=np.nan,
        positive_ratio=np.nan,
        date_count=date_count,
        observation_count=observation_count,
        valid_fraction=valid_fraction,
        direction=0,
    )


def invalid_mi_outcome(
    reason: str,
    *,
    observation_count: int = 0,
    valid_fraction: float = 0.0,
    date_count: int = 0,
    block_count: int = 0,
    k: int = 5,
    stability_weight: float = 0.5,
) -> MIFitnessOutcome:
    """为结构、样本量或 KSG 失败候选生成统一拒绝结果。"""

    return MIFitnessOutcome(
        accepted=False,
        rejection_reason=reason,
        fitness=float("-inf"),
        mi_estimate=np.nan,
        stability_penalty=np.nan,
        stability_weight=stability_weight,
        block_mi_mean=np.nan,
        block_mi_std=np.nan,
        block_mi_min=np.nan,
        block_positive_ratio=np.nan,
        gaussian_equivalent_abs_correlation=np.nan,
        date_count=date_count,
        block_count=block_count,
        observation_count=observation_count,
        valid_fraction=valid_fraction,
        k=k,
        jittered=False,
    )


def invalid_cmi_outcome(
    reason: str,
    *,
    observation_count: int = 0,
    valid_fraction: float = 0.0,
    date_count: int = 0,
    block_count: int = 0,
    k: int = 5,
    stability_weight: float = 0.5,
    complexity_weight: float = 0.005,
    turnover_weight: float = 0.02,
    conditioning_dimension: int = 0,
    conditioning_factors: str = "",
) -> CMIFitnessOutcome:
    """为结构、样本量或 Frenzel–Pompe 失败候选生成统一拒绝结果。"""

    return CMIFitnessOutcome(
        accepted=False,
        rejection_reason=reason,
        fitness=float("-inf"),
        cmi_estimate=np.nan,
        stability_penalty=np.nan,
        stability_weight=stability_weight,
        block_cmi_mean=np.nan,
        block_cmi_std=np.nan,
        block_cmi_min=np.nan,
        block_positive_ratio=np.nan,
        complexity_penalty=np.nan,
        complexity_weight=complexity_weight,
        normalized_complexity=np.nan,
        turnover_penalty=np.nan,
        turnover_weight=turnover_weight,
        mean_rank_turnover=np.nan,
        conditioning_dimension=conditioning_dimension,
        conditioning_factors=conditioning_factors,
        date_count=date_count,
        block_count=block_count,
        observation_count=observation_count,
        valid_fraction=valid_fraction,
        k=k,
        jittered=False,
    )


def prepare_rankgaussian_mi_sample(
    factor: pd.Series,
    frame: pd.DataFrame,
    *,
    label_column: str = "label_1d",
    date_column: str = "date",
    stock_column: str = "stock",
    minimum_cross_section: int = 50,
) -> pd.DataFrame:
    """对因子与 label 逐日 Rank Gaussianize，并保留有效日期。

    这一步先条件于日期移除市场水平和尺度漂移，再为 pooled
    KSG 提供距离口径一致的连续输入。
    """

    if len(factor) != len(frame) or not factor.index.equals(frame.index):
        raise ValueError("factor 必须与 frame 长度和索引完全一致")
    required = {date_column, stock_column, label_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"frame 缺少列: {sorted(missing)}")
    if minimum_cross_section < 3:
        raise ValueError("minimum_cross_section 至少为 3")
    work = pd.DataFrame(
        {
            date_column: frame[date_column].to_numpy(),
            stock_column: frame[stock_column].to_numpy(),
            "factor": pd.to_numeric(factor, errors="coerce").to_numpy(),
            "label": pd.to_numeric(frame[label_column], errors="coerce").to_numpy(),
        }
    )
    transformed = transform_panel_by_date(
        work,
        ["factor", "label"],
        method="rankgauss_by_date",
        date_col=date_column,
    )
    transformed = transformed.dropna(subset=["factor", "label"])
    counts = transformed.groupby(date_column, sort=True).size()
    valid_dates = counts.index[counts.ge(minimum_cross_section)]
    return transformed.loc[
        transformed[date_column].isin(valid_dates)
    ].sort_values([date_column, stock_column]).reset_index(drop=True)


def evaluate_mi_fitness(
    factor: pd.Series,
    frame: pd.DataFrame,
    *,
    label_column: str = "label_1d",
    date_column: str = "date",
    stock_column: str = "stock",
    k: int = 5,
    block_count: int = 4,
    stability_weight: float = 0.5,
    minimum_cross_section: int = 50,
    minimum_dates: int = 100,
    minimum_block_observations: int = 1_500,
    output_policy: OutputPolicy | None = None,
    random_state: int = 20260918,
) -> tuple[MIFitnessOutcome, pd.DataFrame]:
    """计算 pooled KSG MI 减去连续时间块标准差惩罚的 fitness。

    返回的 ``mi_estimate`` 始终是原始有限样本估计
    :math:`\\widehat I`，允许由于估计误差略小于零；不将它裁剪成
    “真实 MI”。
    """

    if not isinstance(block_count, (int, np.integer)) or block_count < 2:
        raise ValueError("block_count 至少为 2")
    if not np.isfinite(stability_weight) or stability_weight < 0:
        raise ValueError("stability_weight 必须是有限非负数")
    policy = output_policy or OutputPolicy(
        minimum_valid_fraction=0.70,
        minimum_valid_observations=6_000,
        minimum_unique_values=2,
    )
    output = check_factor_output(factor, policy)
    if not output.accepted:
        return (
            invalid_mi_outcome(
                ",".join(output.reasons),
                observation_count=output.valid_count,
                valid_fraction=output.valid_fraction,
                k=k,
                stability_weight=stability_weight,
            ),
            pd.DataFrame(),
        )
    sample = prepare_rankgaussian_mi_sample(
        factor,
        frame,
        label_column=label_column,
        date_column=date_column,
        stock_column=stock_column,
        minimum_cross_section=minimum_cross_section,
    )
    dates = pd.DatetimeIndex(sample[date_column].drop_duplicates().sort_values())
    if len(dates) < minimum_dates or len(dates) < block_count:
        return (
            invalid_mi_outcome(
                "too_few_valid_dates",
                observation_count=len(sample),
                valid_fraction=output.valid_fraction,
                date_count=len(dates),
                block_count=block_count,
                k=k,
                stability_weight=stability_weight,
            ),
            pd.DataFrame(),
        )

    try:
        full = estimate_mi_ksg_details(
            sample["factor"].to_numpy(),
            sample["label"].to_numpy(),
            k=k,
            tie_strategy="jitter",
            random_state=random_state,
        )
        block_rows: list[dict[str, object]] = []
        block_estimates: list[float] = []
        jittered = full.jittered
        for block_number, block_dates in enumerate(np.array_split(dates, block_count)):
            block = sample.loc[sample[date_column].isin(block_dates)]
            if len(block) < max(minimum_block_observations, k + 1):
                return (
                    invalid_mi_outcome(
                        "too_few_block_observations",
                        observation_count=len(sample),
                        valid_fraction=output.valid_fraction,
                        date_count=len(dates),
                        block_count=block_count,
                        k=k,
                        stability_weight=stability_weight,
                    ),
                    pd.DataFrame(block_rows),
                )
            details = estimate_mi_ksg_details(
                block["factor"].to_numpy(),
                block["label"].to_numpy(),
                k=k,
                tie_strategy="jitter",
                random_state=random_state + (block_number + 1) * 10_007,
            )
            jittered = jittered or details.jittered
            block_estimates.append(details.estimate)
            block_rows.append(
                {
                    "block": block_number + 1,
                    "start_date": pd.Timestamp(block_dates[0]),
                    "end_date": pd.Timestamp(block_dates[-1]),
                    "date_count": len(block_dates),
                    "observation_count": len(block),
                    "mi_estimate": details.estimate,
                    "jittered": details.jittered,
                }
            )
    except ValueError as error:
        return (
            invalid_mi_outcome(
                f"ksg_error:{error}",
                observation_count=len(sample),
                valid_fraction=output.valid_fraction,
                date_count=len(dates),
                block_count=block_count,
                k=k,
                stability_weight=stability_weight,
            ),
            pd.DataFrame(),
        )

    block_array = np.asarray(block_estimates, dtype=float)
    block_std = float(np.std(block_array, ddof=1))
    penalty = float(stability_weight * block_std)
    estimate = float(full.estimate)
    equivalent = (
        float(np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * estimate))))
        if estimate >= 0
        else np.nan
    )
    return (
        MIFitnessOutcome(
            accepted=True,
            rejection_reason="",
            fitness=estimate - penalty,
            mi_estimate=estimate,
            stability_penalty=penalty,
            stability_weight=stability_weight,
            block_mi_mean=float(np.mean(block_array)),
            block_mi_std=block_std,
            block_mi_min=float(np.min(block_array)),
            block_positive_ratio=float(np.mean(block_array > 0.0)),
            gaussian_equivalent_abs_correlation=equivalent,
            date_count=len(dates),
            block_count=block_count,
            observation_count=len(sample),
            valid_fraction=output.valid_fraction,
            k=k,
            jittered=jittered,
        ),
        pd.DataFrame(block_rows),
    )


def evaluate_pure_mi_fitness(
    factor: pd.Series,
    frame: pd.DataFrame,
    *,
    label_column: str = "label_1d",
    date_column: str = "date",
    k: int = 5,
    minimum_dates: int = 100,
    output_policy: OutputPolicy | None = None,
    random_state: int = 20260921,
) -> MIFitnessOutcome:
    """以未经软惩罚的 pooled KSG ``MI-hat`` 作为唯一 fitness。

    ``factor`` 与 ``frame[label_column]`` 必须已经使用同一时点口径完成
    日内截面 Rank Gaussianization。本函数故意不计算时间块稳定性、换手率
    或表达式复杂度惩罚；这些量只在候选入围后作为诊断。结构复杂度仍由 GP
    的硬约束负责，防止表达式失控而不改变候选之间的 fitness 排序。
    """

    required = {date_column, label_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"frame 缺少列: {sorted(missing)}")
    if len(factor) != len(frame) or not factor.index.equals(frame.index):
        raise ValueError("factor 必须与 frame 长度和索引完全一致")
    if minimum_dates < 3:
        raise ValueError("minimum_dates 至少为 3")

    policy = output_policy or OutputPolicy(
        minimum_valid_fraction=0.70,
        minimum_valid_observations=6_000,
        minimum_unique_values=2,
    )
    output = check_factor_output(factor, policy)
    if not output.accepted:
        return invalid_mi_outcome(
            ",".join(output.reasons),
            observation_count=output.valid_count,
            valid_fraction=output.valid_fraction,
            k=k,
            stability_weight=0.0,
        )

    work = pd.DataFrame(
        {
            date_column: frame[date_column].to_numpy(),
            "factor": pd.to_numeric(factor, errors="coerce").to_numpy(),
            "label": pd.to_numeric(frame[label_column], errors="coerce").to_numpy(),
        }
    )
    finite = np.isfinite(work[["factor", "label"]]).all(axis=1)
    sample = work.loc[finite].sort_values(date_column).reset_index(drop=True)
    date_count = int(sample[date_column].nunique())
    if date_count < minimum_dates:
        return invalid_mi_outcome(
            "too_few_valid_dates",
            observation_count=len(sample),
            valid_fraction=output.valid_fraction,
            date_count=date_count,
            k=k,
            stability_weight=0.0,
        )

    try:
        details = estimate_mi_ksg_details(
            sample["factor"].to_numpy(),
            sample["label"].to_numpy(),
            k=k,
            tie_strategy="jitter",
            random_state=random_state,
        )
    except ValueError as error:
        return invalid_mi_outcome(
            f"ksg_error:{error}",
            observation_count=len(sample),
            valid_fraction=output.valid_fraction,
            date_count=date_count,
            k=k,
            stability_weight=0.0,
        )

    estimate = float(details.estimate)
    equivalent = (
        float(np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * estimate))))
        if estimate >= 0.0
        else np.nan
    )
    return MIFitnessOutcome(
        accepted=True,
        rejection_reason="",
        fitness=estimate,
        mi_estimate=estimate,
        stability_penalty=0.0,
        stability_weight=0.0,
        block_mi_mean=np.nan,
        block_mi_std=np.nan,
        block_mi_min=np.nan,
        block_positive_ratio=np.nan,
        gaussian_equivalent_abs_correlation=equivalent,
        date_count=date_count,
        block_count=0,
        observation_count=len(sample),
        valid_fraction=output.valid_fraction,
        k=k,
        jittered=details.jittered,
    )


def evaluate_cmi_fitness(
    factor: pd.Series,
    frame: pd.DataFrame,
    conditioning_columns: tuple[str, ...] | list[str],
    *,
    label_column: str = "label_1d",
    date_column: str = "date",
    k: int = 5,
    block_count: int = 4,
    stability_weight: float = 0.5,
    weighted_expression_complexity: float = 0.0,
    maximum_weighted_complexity: float = 25.0,
    complexity_weight: float = 0.005,
    mean_rank_turnover: float = 0.0,
    turnover_weight: float = 0.02,
    maximum_conditioning_dimension: int = 4,
    minimum_dates: int = 100,
    minimum_block_observations: int = 1_500,
    output_policy: OutputPolicy | None = None,
    random_state: int = 20260919,
) -> tuple[CMIFitnessOutcome, pd.DataFrame]:
    """计算 CMI 减去稳定性、复杂度和换手率惩罚后的 fitness。

    ``factor``、label 与 conditioning columns 必须已经使用同一时点口径
    完成日内截面变换。本函数不再次 rankgauss，避免错误地只在抽样后的
    小截面上重新定义秩。有限样本 CMI 允许为负，不裁剪成零。
    """

    conditioning = tuple(conditioning_columns)
    conditioning_text = ",".join(conditioning)
    dimension = len(conditioning)
    if dimension < 1:
        raise ValueError("conditioning_columns 至少包含一列")
    if dimension > maximum_conditioning_dimension:
        return (
            invalid_cmi_outcome(
                "conditioning_dimension_limit",
                k=k,
                stability_weight=stability_weight,
                complexity_weight=complexity_weight,
                turnover_weight=turnover_weight,
                conditioning_dimension=dimension,
                conditioning_factors=conditioning_text,
            ),
            pd.DataFrame(),
        )
    required = {date_column, label_column, *conditioning}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"frame 缺少列: {sorted(missing)}")
    if len(factor) != len(frame) or not factor.index.equals(frame.index):
        raise ValueError("factor 必须与 frame 长度和索引完全一致")
    if not isinstance(block_count, (int, np.integer)) or block_count < 2:
        raise ValueError("block_count 至少为 2")
    numeric_parameters = {
        "stability_weight": stability_weight,
        "weighted_expression_complexity": weighted_expression_complexity,
        "maximum_weighted_complexity": maximum_weighted_complexity,
        "complexity_weight": complexity_weight,
        "mean_rank_turnover": mean_rank_turnover,
        "turnover_weight": turnover_weight,
    }
    if any(not np.isfinite(value) for value in numeric_parameters.values()):
        raise ValueError("fitness 权重、复杂度和换手率必须是有限数")
    if stability_weight < 0 or complexity_weight < 0 or turnover_weight < 0:
        raise ValueError("三项惩罚权重必须非负")
    if weighted_expression_complexity < 0 or maximum_weighted_complexity <= 0:
        raise ValueError("复杂度必须非负且最大复杂度必须为正")
    if not 0.0 <= mean_rank_turnover <= 1.0:
        raise ValueError("mean_rank_turnover 必须位于 [0,1]")

    policy = output_policy or OutputPolicy(
        minimum_valid_fraction=0.70,
        minimum_valid_observations=6_000,
        minimum_unique_values=2,
    )
    output = check_factor_output(factor, policy)
    if not output.accepted:
        return (
            invalid_cmi_outcome(
                ",".join(output.reasons),
                observation_count=output.valid_count,
                valid_fraction=output.valid_fraction,
                k=k,
                stability_weight=stability_weight,
                complexity_weight=complexity_weight,
                turnover_weight=turnover_weight,
                conditioning_dimension=dimension,
                conditioning_factors=conditioning_text,
            ),
            pd.DataFrame(),
        )

    work = pd.DataFrame(
        {
            date_column: frame[date_column].to_numpy(),
            "factor": pd.to_numeric(factor, errors="coerce").to_numpy(),
            "label": pd.to_numeric(frame[label_column], errors="coerce").to_numpy(),
        }
    )
    for column in conditioning:
        work[column] = pd.to_numeric(frame[column], errors="coerce").to_numpy()
    finite_columns = ["factor", "label", *conditioning]
    finite = np.isfinite(work[finite_columns]).all(axis=1)
    sample = work.loc[finite].sort_values(date_column).reset_index(drop=True)
    dates = pd.DatetimeIndex(sample[date_column].drop_duplicates().sort_values())
    if len(dates) < minimum_dates or len(dates) < block_count:
        return (
            invalid_cmi_outcome(
                "too_few_valid_dates",
                observation_count=len(sample),
                valid_fraction=output.valid_fraction,
                date_count=len(dates),
                block_count=block_count,
                k=k,
                stability_weight=stability_weight,
                complexity_weight=complexity_weight,
                turnover_weight=turnover_weight,
                conditioning_dimension=dimension,
                conditioning_factors=conditioning_text,
            ),
            pd.DataFrame(),
        )

    try:
        full = estimate_cmi_frenzel_pompe_details(
            sample["factor"].to_numpy(),
            sample["label"].to_numpy(),
            sample.loc[:, conditioning].to_numpy(),
            k=k,
            tie_strategy="jitter",
            random_state=random_state,
        )
        block_rows: list[dict[str, object]] = []
        block_estimates: list[float] = []
        jittered = full.jittered
        for block_number, block_dates in enumerate(np.array_split(dates, block_count)):
            block = sample.loc[sample[date_column].isin(block_dates)]
            if len(block) < max(minimum_block_observations, k + 1):
                return (
                    invalid_cmi_outcome(
                        "too_few_block_observations",
                        observation_count=len(sample),
                        valid_fraction=output.valid_fraction,
                        date_count=len(dates),
                        block_count=block_count,
                        k=k,
                        stability_weight=stability_weight,
                        complexity_weight=complexity_weight,
                        turnover_weight=turnover_weight,
                        conditioning_dimension=dimension,
                        conditioning_factors=conditioning_text,
                    ),
                    pd.DataFrame(block_rows),
                )
            details = estimate_cmi_frenzel_pompe_details(
                block["factor"].to_numpy(),
                block["label"].to_numpy(),
                block.loc[:, conditioning].to_numpy(),
                k=k,
                tie_strategy="jitter",
                random_state=random_state + (block_number + 1) * 10_007,
            )
            jittered = jittered or details.jittered
            block_estimates.append(details.estimate)
            block_rows.append(
                {
                    "block": block_number + 1,
                    "start_date": pd.Timestamp(block_dates[0]),
                    "end_date": pd.Timestamp(block_dates[-1]),
                    "date_count": len(block_dates),
                    "observation_count": len(block),
                    "cmi_estimate": details.estimate,
                    "joint_dimension": details.joint_dimension,
                    "median_joint_radius": details.median_joint_radius,
                    "jittered": details.jittered,
                }
            )
    except ValueError as error:
        return (
            invalid_cmi_outcome(
                f"frenzel_pompe_error:{error}",
                observation_count=len(sample),
                valid_fraction=output.valid_fraction,
                date_count=len(dates),
                block_count=block_count,
                k=k,
                stability_weight=stability_weight,
                complexity_weight=complexity_weight,
                turnover_weight=turnover_weight,
                conditioning_dimension=dimension,
                conditioning_factors=conditioning_text,
            ),
            pd.DataFrame(),
        )

    block_array = np.asarray(block_estimates, dtype=float)
    block_std = float(np.std(block_array, ddof=1))
    stability_penalty = float(stability_weight * block_std)
    normalized_complexity = float(
        np.clip(weighted_expression_complexity / maximum_weighted_complexity, 0.0, 1.0)
    )
    complexity_penalty = float(complexity_weight * normalized_complexity)
    turnover_penalty = float(turnover_weight * mean_rank_turnover)
    estimate = float(full.estimate)
    fitness = estimate - stability_penalty - complexity_penalty - turnover_penalty
    return (
        CMIFitnessOutcome(
            accepted=True,
            rejection_reason="",
            fitness=float(fitness),
            cmi_estimate=estimate,
            stability_penalty=stability_penalty,
            stability_weight=stability_weight,
            block_cmi_mean=float(np.mean(block_array)),
            block_cmi_std=block_std,
            block_cmi_min=float(np.min(block_array)),
            block_positive_ratio=float(np.mean(block_array > 0.0)),
            complexity_penalty=complexity_penalty,
            complexity_weight=complexity_weight,
            normalized_complexity=normalized_complexity,
            turnover_penalty=turnover_penalty,
            turnover_weight=turnover_weight,
            mean_rank_turnover=float(mean_rank_turnover),
            conditioning_dimension=dimension,
            conditioning_factors=conditioning_text,
            date_count=len(dates),
            block_count=block_count,
            observation_count=len(sample),
            valid_fraction=output.valid_fraction,
            k=k,
            jittered=jittered,
        ),
        pd.DataFrame(block_rows),
    )


def align_validation_direction(
    training_mean_rank_ic: float,
    validation_mean_rank_ic: float,
) -> float:
    """只用训练期符号确定多空方向，再报告 validation 的方向一致 RankIC。"""

    if not np.isfinite(training_mean_rank_ic) or not np.isfinite(
        validation_mean_rank_ic
    ):
        return np.nan
    direction = 1.0 if training_mean_rank_ic >= 0 else -1.0
    return float(direction * validation_mean_rank_ic)


def daily_rank_turnover(
    factor: pd.Series,
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    stock_column: str = "stock",
    minimum_common_stocks: int = 30,
) -> pd.DataFrame:
    """计算相邻日期共同股票的平均绝对百分位 rank 变化。"""

    if len(factor) != len(frame) or not factor.index.equals(frame.index):
        raise ValueError("factor 必须与 frame 长度和索引完全一致")
    required = {date_column, stock_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"frame 缺少列: {sorted(missing)}")
    work = pd.DataFrame(
        {
            "date": frame[date_column].to_numpy(),
            "stock": frame[stock_column].to_numpy(),
            "factor": pd.to_numeric(factor, errors="coerce").to_numpy(),
        }
    )
    work = work.loc[np.isfinite(work["factor"])].copy()
    work["rank"] = work.groupby("date", sort=True)["factor"].rank(
        method="average", pct=True
    )
    dates = sorted(work["date"].unique())
    rows: list[dict[str, object]] = []
    previous: pd.DataFrame | None = None
    previous_date: object | None = None
    for date in dates:
        current = work.loc[work["date"].eq(date), ["stock", "rank"]]
        if previous is not None:
            common = previous.merge(current, on="stock", suffixes=("_previous", "_current"))
            count = len(common)
            turnover = (
                float((common["rank_current"] - common["rank_previous"]).abs().mean())
                if count >= minimum_common_stocks
                else np.nan
            )
            rows.append(
                {
                    "previous_date": previous_date,
                    "date": date,
                    "common_stocks": count,
                    "rank_turnover": turnover,
                    "valid": bool(np.isfinite(turnover)),
                }
            )
        previous = current
        previous_date = date
    return pd.DataFrame(rows)
