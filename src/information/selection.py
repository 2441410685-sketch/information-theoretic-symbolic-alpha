"""基于增量预测信息的前向因子选择。"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .conditional_mi import estimate_cmi_frenzel_pompe_details
from .mutual_information import estimate_mi_ksg_details
from .significance import block_permutation_indices, monte_carlo_p_value


@dataclass(frozen=True)
class ForwardSelectionResult:
    """保存前向选择结果、逐步候选分数和 max-null 抽样。"""

    selected_factors: tuple[str, ...]
    audit: pd.DataFrame
    candidate_scores: pd.DataFrame
    null_draws: pd.DataFrame
    stop_reason: str


def _validate_selection_frame(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    label: str,
    date_col: str,
) -> list[str]:
    """验证前向选择所需的列与主键级日期信息。"""

    names = list(candidates)
    if not names or len(names) != len(set(names)):
        raise ValueError("candidates 必须非空且不重复")
    required = {date_col, label, *names}
    if missing := required.difference(frame.columns):
        raise ValueError(f"数据缺少字段：{sorted(missing)}")
    if len(frame) < 20:
        raise ValueError("前向选择至少需要20行样本")
    if pd.to_datetime(frame[date_col], errors="coerce").isna().any():
        raise ValueError("日期列包含无法解析的值")
    return names


def estimate_information_increment(
    frame: pd.DataFrame,
    candidate: str,
    label: str,
    selected_factors: Sequence[str],
    *,
    k: int = 5,
    min_observations: int = 500,
    random_state: int = 20260904,
) -> dict[str, float | int | str]:
    """估计候选因子的边际或条件增量信息，保留原始有限样本值。

    当选择集合为空时，分数是 ``I(candidate; label)``；否则分数是
    ``I(candidate; label | selected_factors)``。所有变量使用共同有限样本，
    避免分数变化只是由缺失样本口径改变造成。
    """

    selected = list(selected_factors)
    columns = list(dict.fromkeys([candidate, label, *selected]))
    if missing := set(columns).difference(frame.columns):
        raise ValueError(f"数据缺少字段：{sorted(missing)}")
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=np.float64)).all(axis=1)
    values = values.loc[finite]
    if len(values) < min_observations:
        return {
            "factor": candidate,
            "score": np.nan,
            "score_type": "mi" if not selected else "cmi",
            "conditioning_dimension": len(selected),
            "n_observations": len(values),
            "status": "insufficient_observations",
        }

    x = values[candidate].to_numpy(dtype=np.float64)
    y = values[label].to_numpy(dtype=np.float64)
    if selected:
        details = estimate_cmi_frenzel_pompe_details(
            x,
            y,
            values[selected].to_numpy(dtype=np.float64),
            k=k,
            tie_strategy="jitter",
            random_state=random_state,
        )
        score = details.estimate
        score_type = "cmi"
    else:
        details = estimate_mi_ksg_details(
            x,
            y,
            k=k,
            tie_strategy="jitter",
            random_state=random_state,
        )
        score = details.estimate
        score_type = "mi"
    return {
        "factor": candidate,
        "score": float(score),
        "score_type": score_type,
        "conditioning_dimension": len(selected),
        "n_observations": len(values),
        "status": "estimated",
    }


def score_information_candidates(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    label: str,
    selected_factors: Sequence[str],
    *,
    k: int = 5,
    min_observations: int = 500,
    random_state: int = 20260904,
) -> pd.DataFrame:
    """在同一选择步骤为全部候选计算 MI 或 CMI，并按分数降序排列。"""

    rows = [
        estimate_information_increment(
            frame,
            candidate,
            label,
            selected_factors,
            k=k,
            min_observations=min_observations,
            random_state=random_state + number * 10_007,
        )
        for number, candidate in enumerate(candidates)
    ]
    result = pd.DataFrame(rows).sort_values(
        ["score", "factor"],
        ascending=[False, True],
        na_position="last",
    )
    result["rank"] = np.arange(1, len(result) + 1)
    return result.reset_index(drop=True)


def permute_equal_count_date_blocks(
    values: np.ndarray | pd.Series,
    dates: np.ndarray | pd.Series,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """按完整日期截面重排连续日期块，并保持块内日期顺序。

    每个日期必须具有相同样本数。重排单位是完整横截面而不是单行，因此
    日内 label 分布不会被打碎；连续日期块内部的顺序也被保留。
    """

    vector = np.asarray(values, dtype=np.float64).ravel()
    date_values = pd.to_datetime(pd.Series(dates), errors="coerce")
    if len(vector) != len(date_values) or len(vector) < 2:
        raise ValueError("values 与 dates 必须等长且至少含2行")
    if np.any(~np.isfinite(vector)) or date_values.isna().any():
        raise ValueError("values 与 dates 不能包含 NaN/inf")

    # 稳定排序确保每个日期内的横截面位置在重排前后口径一致。
    order = np.argsort(date_values.to_numpy(), kind="stable")
    sorted_dates = date_values.iloc[order].to_numpy()
    unique_dates, counts = np.unique(sorted_dates, return_counts=True)
    if len(unique_dates) < 3 or len(set(counts.tolist())) != 1:
        raise ValueError("日期块重排要求至少3日且每日样本数完全相同")
    date_order = block_permutation_indices(len(unique_dates), block_length, rng)
    width = int(counts[0])
    matrix = vector[order].reshape(len(unique_dates), width)
    permuted_sorted = matrix[date_order].reshape(-1)
    result = np.empty_like(vector)
    result[order] = permuted_sorted
    return result


def _chronological_halves(
    frame: pd.DataFrame,
    date_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把训练日期按时间切为前后两半，用于训练期内稳定性检查。"""

    dates = np.sort(pd.to_datetime(frame[date_col]).unique())
    if len(dates) < 20:
        raise ValueError("稳定性检查至少需要20个交易日")
    split = len(dates) // 2
    first_dates = set(dates[:split])
    first = frame.loc[pd.to_datetime(frame[date_col]).isin(first_dates)]
    second = frame.loc[~pd.to_datetime(frame[date_col]).isin(first_dates)]
    return first, second


def forward_information_selection(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    label: str,
    *,
    date_col: str = "date",
    max_factors: int = 4,
    k: int = 5,
    min_observations: int = 500,
    n_permutations: int = 19,
    block_length: int = 5,
    null_quantile: float = 0.95,
    require_positive_halves: bool = True,
    random_state: int = 20260904,
) -> ForwardSelectionResult:
    """使用 CMI、训练期半窗稳定性和 block max-null 逐步选择因子。

    null 每次重排完整 label 日期块，并记录全部剩余候选分数的最大值，
    从而校正“每一步挑最大者”的 winner's curse。它是金融 panel 的实用
    screening null，不冒充精确的 conditional-randomization test。
    """

    candidate_names = _validate_selection_frame(
        frame, candidates, label, date_col
    )
    if not 1 <= max_factors <= 8:
        raise ValueError("max_factors 必须位于1到8；高维 CMI 不可靠")
    if n_permutations < 19:
        raise ValueError("n_permutations 至少为19")
    if not 0.5 < null_quantile < 1.0:
        raise ValueError("null_quantile 必须位于 (0.5, 1)")

    working = frame.copy()
    working[date_col] = pd.to_datetime(working[date_col])
    selected: list[str] = []
    audit_rows: list[dict[str, object]] = []
    candidate_parts: list[pd.DataFrame] = []
    null_rows: list[dict[str, object]] = []
    first_half, second_half = _chronological_halves(working, date_col)
    rng = np.random.default_rng(random_state)
    stop_reason = "maximum_factor_count_reached"

    for step in range(1, max_factors + 1):
        remaining = [name for name in candidate_names if name not in selected]
        observed = score_information_candidates(
            working,
            remaining,
            label,
            selected,
            k=k,
            min_observations=min_observations,
            random_state=random_state + step * 1_000_003,
        )
        observed.insert(0, "step", step)
        observed.insert(1, "selected_before", ",".join(selected))
        candidate_parts.append(observed)
        valid = observed.loc[
            observed["status"].eq("estimated") & observed["score"].notna()
        ]
        if valid.empty:
            stop_reason = "no_estimable_candidate"
            break
        winner = str(valid.iloc[0]["factor"])
        winner_score = float(valid.iloc[0]["score"])

        half_scores: list[float] = []
        for half_number, half in enumerate((first_half, second_half), start=1):
            result = estimate_information_increment(
                half,
                winner,
                label,
                selected,
                k=k,
                min_observations=max(k + 1, min_observations // 3),
                random_state=random_state + step * 100_003 + half_number,
            )
            half_scores.append(float(result["score"]))

        maximum_null = np.empty(n_permutations, dtype=np.float64)
        for repetition in range(n_permutations):
            permuted = working.copy()
            permuted[label] = permute_equal_count_date_blocks(
                working[label].to_numpy(dtype=np.float64),
                working[date_col],
                block_length,
                rng,
            )
            null_scores = score_information_candidates(
                permuted,
                remaining,
                label,
                selected,
                k=k,
                min_observations=min_observations,
                random_state=(
                    random_state
                    + step * 10_000_019
                    + (repetition + 1) * 100_003
                ),
            )
            maximum_null[repetition] = float(null_scores["score"].max())
            null_rows.append(
                {
                    "step": step,
                    "repetition": repetition + 1,
                    "maximum_null_score": maximum_null[repetition],
                }
            )

        # 有限 B 下不能使用线性插值分位数后又把它误当作精确 alpha 检验。
        # ``higher`` 取真实 order statistic，并与加一修正 p-value 同时约束。
        null_threshold = float(
            np.quantile(maximum_null, null_quantile, method="higher")
        )
        p_value, exceedances = monte_carlo_p_value(
            winner_score,
            maximum_null,
            alternative="greater",
        )
        alpha = 1.0 - null_quantile
        passes_null = bool(
            winner_score > null_threshold
            and p_value <= alpha + np.finfo(float).eps
        )
        passes_stability = (
            min(half_scores) > 0.0 if require_positive_halves else True
        )
        accepted = bool(passes_null and passes_stability)
        if not passes_null:
            decision = "stop_max_null"
            stop_reason = "winner_did_not_exceed_block_max_null"
        elif not passes_stability:
            decision = "stop_unstable_halves"
            stop_reason = "winner_not_positive_in_both_training_halves"
        else:
            decision = "accept"

        audit_rows.append(
            {
                "step": step,
                "conditioning_dimension": len(selected),
                "selected_before": ",".join(selected),
                "winner": winner,
                "winner_score": winner_score,
                "first_half_score": half_scores[0],
                "second_half_score": half_scores[1],
                "null_mean": float(np.mean(maximum_null)),
                "null_q95": null_threshold,
                "null_alpha": alpha,
                "max_null_p_value": p_value,
                "null_exceedances": exceedances,
                "candidate_count": len(remaining),
                "passes_max_null": passes_null,
                "passes_training_stability": passes_stability,
                "accepted": accepted,
                "decision": decision,
            }
        )
        if not accepted:
            break
        selected.append(winner)

    audit = pd.DataFrame(audit_rows)
    candidate_scores = (
        pd.concat(candidate_parts, ignore_index=True)
        if candidate_parts
        else pd.DataFrame()
    )
    null_draws = pd.DataFrame(null_rows)
    return ForwardSelectionResult(
        selected_factors=tuple(selected),
        audit=audit,
        candidate_scores=candidate_scores,
        null_draws=null_draws,
        stop_reason=stop_reason,
    )


def sequential_information_path(
    frame: pd.DataFrame,
    ordered_factors: Sequence[str],
    label: str,
    *,
    k: int = 5,
    min_observations: int = 500,
    random_state: int = 20260904,
) -> pd.DataFrame:
    """按冻结顺序计算每个因子的增量信息及其链式累计值。"""

    selected: list[str] = []
    rows: list[dict[str, object]] = []
    cumulative = 0.0
    for step, factor in enumerate(ordered_factors, start=1):
        result = estimate_information_increment(
            frame,
            factor,
            label,
            selected,
            k=k,
            min_observations=min_observations,
            random_state=random_state + step * 100_003,
        )
        increment = float(result["score"])
        cumulative += increment
        rows.append(
            {
                "step": step,
                "factor": factor,
                "conditioning_factors": ",".join(selected),
                "incremental_information": increment,
                "cumulative_information": cumulative,
                "score_type": result["score_type"],
                "n_observations": result["n_observations"],
            }
        )
        selected.append(factor)
    return pd.DataFrame(rows)
