"""全 A 股截面 LightGBM 的目标、fold 与训练函数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtri


@dataclass(frozen=True)
class WalkForwardFold:
    """保存一个不重叠 OOS fold 的全部日期边界。"""

    fold: int
    inner_train_dates: tuple[pd.Timestamp, ...]
    inner_validation_dates: tuple[pd.Timestamp, ...]
    final_train_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]


def rank_gaussian_target_by_date(
    dates: pd.Series | np.ndarray,
    values: pd.Series | np.ndarray,
    *,
    minimum_cross_section: int = 50,
) -> np.ndarray:
    """把每日未来收益的截面秩映射为标准正态分位数。

    该变换只作用于监督学习标签，不进入特征。每天独立转换可以消除市场
    公共涨跌和不同波动状态造成的尺度漂移，同时保留完整截面顺序。
    """

    date_values = pd.to_datetime(np.asarray(dates))
    numeric = np.asarray(values, dtype=float)
    if len(date_values) != len(numeric):
        raise ValueError("dates 与 values 长度必须一致")
    if minimum_cross_section < 3:
        raise ValueError("minimum_cross_section 至少为 3")

    result = np.full(len(numeric), np.nan, dtype=np.float64)
    frame = pd.DataFrame({"date": date_values, "value": numeric})
    for indices in frame.groupby("date", sort=True, observed=True).indices.values():
        position = np.asarray(indices, dtype=np.int64)
        block = numeric[position]
        finite = np.isfinite(block)
        count = int(finite.sum())
        if count < minimum_cross_section:
            continue
        ranks = pd.Series(block[finite]).rank(method="average").to_numpy(dtype=float)
        probability = (ranks - 0.5) / count
        result[position[finite]] = ndtri(probability)
    return result


def date_equal_sample_weights(
    dates: pd.Series | np.ndarray,
    valid: pd.Series | np.ndarray | None = None,
) -> np.ndarray:
    """让每个交易日的有效样本总权重严格等于 1。

    全 A 股股票数量随时间增长。如果每行权重都等于 1，后期日期会仅因股票
    更多而在损失中占据更大权重。日期等权避免这种无关的时间偏置。
    """

    date_values = pd.to_datetime(np.asarray(dates))
    mask = (
        np.ones(len(date_values), dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if len(mask) != len(date_values):
        raise ValueError("valid 与 dates 长度必须一致")
    result = np.zeros(len(date_values), dtype=np.float64)
    frame = pd.DataFrame({"date": date_values, "valid": mask})
    for indices in frame.groupby("date", sort=False, observed=True).indices.values():
        position = np.asarray(indices, dtype=np.int64)
        selected = position[mask[position]]
        if len(selected):
            result[selected] = 1.0 / len(selected)
    return result


def date_membership_mask(
    dates: pd.Series | pd.DatetimeIndex | np.ndarray,
    selected_dates: tuple[pd.Timestamp, ...] | pd.DatetimeIndex | np.ndarray,
) -> np.ndarray:
    """跨 Pandas/NumPy 版本稳定地判断日期是否属于冻结窗口。

    Parquet 在不同环境中可能把日期还原为 ``datetime64[ns]``、``[us]``，
    而 fold 保存的是 ``Timestamp``。直接 ``np.isin`` 在部分版本组合下会把
    单位差异当作对象类型差异，得到全 False。这里先统一为自然日整数编码。
    """

    date_index = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates), errors="coerce"))
    selected_index = pd.DatetimeIndex(
        pd.to_datetime(np.asarray(selected_dates), errors="coerce")
    )
    if date_index.hasnans:
        raise ValueError("dates 含无法解析的日期")
    if selected_index.hasnans or selected_index.empty:
        raise ValueError("selected_dates 不能为空或含无效日期")
    date_codes = date_index.to_numpy(dtype="datetime64[D]").astype(np.int64)
    selected_codes = selected_index.to_numpy(dtype="datetime64[D]").astype(np.int64)
    return np.isin(date_codes, selected_codes)


def build_walk_forward_folds(
    all_dates: pd.Series | pd.DatetimeIndex | np.ndarray,
    *,
    oos_year: int,
    rolling_train_dates: int = 720,
    inner_validation_dates: int = 60,
    inner_gap_dates: int = 2,
    outer_gap_dates: int = 2,
    test_dates: int = 20,
    step_dates: int = 20,
) -> list[WalkForwardFold]:
    """建立两层时间隔离的 walk-forward folds。

    inner train 与 inner validation 之间留出 ``inner_gap_dates``；用于最终重训
    的720日窗口与真正 OOS test 之间再留出 ``outer_gap_dates``。这样训练样本
    的 t+2 开盘标签不会穿过测试信号日边界。
    """

    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(all_dates))).drop_duplicates().sort_values()
    if dates.empty:
        raise ValueError("all_dates 不能为空")
    for name, value in {
        "rolling_train_dates": rolling_train_dates,
        "inner_validation_dates": inner_validation_dates,
        "inner_gap_dates": inner_gap_dates,
        "outer_gap_dates": outer_gap_dates,
        "test_dates": test_dates,
        "step_dates": step_dates,
    }.items():
        if value < 1:
            raise ValueError(f"{name} 必须为正整数")

    oos = dates[dates.year == int(oos_year)]
    folds: list[WalkForwardFold] = []
    for fold_number, test_start in enumerate(range(0, len(oos), step_dates), start=1):
        test_block = oos[test_start : test_start + test_dates]
        if test_block.empty:
            continue
        first_test_position = int(dates.get_loc(test_block[0]))
        final_train_end = first_test_position - outer_gap_dates
        final_train_start = final_train_end - rolling_train_dates
        inner_valid_end = final_train_end
        inner_valid_start = inner_valid_end - inner_validation_dates
        inner_train_end = inner_valid_start - inner_gap_dates
        inner_train_start = inner_train_end - rolling_train_dates
        if min(final_train_start, inner_valid_start, inner_train_start) < 0:
            raise ValueError(
                f"{test_block[0].date()} 之前历史不足，无法建立冻结窗口"
            )
        fold = WalkForwardFold(
            fold=fold_number,
            inner_train_dates=tuple(dates[inner_train_start:inner_train_end]),
            inner_validation_dates=tuple(dates[inner_valid_start:inner_valid_end]),
            final_train_dates=tuple(dates[final_train_start:final_train_end]),
            test_dates=tuple(test_block),
        )
        # 明确断言时间顺序，防止切片修改后悄悄引入泄漏。
        if not (
            max(fold.inner_train_dates) < min(fold.inner_validation_dates)
            and max(fold.final_train_dates) < min(fold.test_dates)
        ):
            raise AssertionError("walk-forward 日期顺序错误")
        folds.append(fold)
    return folds


def fit_lightgbm_with_inner_early_stopping(
    x_inner_train: np.ndarray,
    y_inner_train: np.ndarray,
    w_inner_train: np.ndarray,
    x_inner_valid: np.ndarray,
    y_inner_valid: np.ndarray,
    w_inner_valid: np.ndarray,
    x_final_train: np.ndarray,
    y_final_train: np.ndarray,
    w_final_train: np.ndarray,
    x_test: np.ndarray,
    *,
    parameters: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[np.ndarray, int, dict[str, list[float]]]:
    """先以验证 L2 选择树数，再在最终720日窗口重训并预测 OOS。

    LightGBM 延迟导入，使不安装 LightGBM 的本地环境仍可运行日期、目标和
    组合单元测试；正式集群任务会在启动前执行依赖预检。
    """

    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("正式训练需要安装 lightgbm") from error

    inner_train = lgb.Dataset(
        x_inner_train,
        label=y_inner_train,
        weight=w_inner_train,
        free_raw_data=True,
    )
    inner_valid = lgb.Dataset(
        x_inner_valid,
        label=y_inner_valid,
        weight=w_inner_valid,
        reference=inner_train,
        free_raw_data=True,
    )
    history: dict[str, dict[str, list[float]]] = {}
    search_model = lgb.train(
        parameters,
        inner_train,
        num_boost_round=num_boost_round,
        valid_sets=[inner_valid],
        valid_names=["inner_valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, first_metric_only=True),
            lgb.record_evaluation(history),
        ],
    )
    best_iteration = int(search_model.best_iteration or num_boost_round)

    final_dataset = lgb.Dataset(
        x_final_train,
        label=y_final_train,
        weight=w_final_train,
        free_raw_data=True,
    )
    final_model = lgb.train(
        parameters,
        final_dataset,
        num_boost_round=best_iteration,
    )
    prediction = final_model.predict(x_test, num_iteration=best_iteration)
    compact_history = {
        metric: list(values)
        for metric, values in history.get("inner_valid", {}).items()
    }
    return np.asarray(prediction, dtype=float), best_iteration, compact_history
