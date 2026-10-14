"""最终 LightGBM 实验使用的多年度滚动切分与训练函数。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.cross_sectional_lightgbm import WalkForwardFold


def build_walk_forward_folds_range(
    all_dates: pd.Series | pd.DatetimeIndex | np.ndarray,
    *,
    oos_start: str | pd.Timestamp,
    oos_end: str | pd.Timestamp,
    rolling_train_dates: int = 720,
    inner_validation_dates: int = 60,
    inner_gap_dates: int = 2,
    outer_gap_dates: int = 2,
    test_dates: int = 20,
    step_dates: int = 20,
) -> list[WalkForwardFold]:
    """构造跨年度、固定长度的严格 walk-forward folds。

    最终训练窗始终只保留测试日前最近 ``rolling_train_dates`` 个交易日，
    所以旧制度会自然退出样本。inner validation 只负责选择树数，真实 OOS
    test 从不参与早停或调参。两个 gap 均按交易日计，用来隔离 t+2 标签。
    """

    # Arrow/Pandas 版本不同会把同一 parquet 日期读成 datetime64[us]
    # 或 datetime64[ns]。后续校验只关心交易日值，因此先统一为纳秒精度，
    # 避免日期完全相同却因底层时间单位不同而被严格比较误判。
    parsed_dates = pd.to_datetime(np.asarray(all_dates), errors="coerce")
    dates = pd.DatetimeIndex(
        np.asarray(parsed_dates, dtype="datetime64[ns]")
    ).drop_duplicates().sort_values()
    if dates.empty or dates.hasnans:
        raise ValueError("all_dates 不能为空或包含无效日期")

    parameters = {
        "rolling_train_dates": rolling_train_dates,
        "inner_validation_dates": inner_validation_dates,
        "inner_gap_dates": inner_gap_dates,
        "outer_gap_dates": outer_gap_dates,
        "test_dates": test_dates,
        "step_dates": step_dates,
    }
    if any(int(value) < 1 for value in parameters.values()):
        raise ValueError("所有窗口参数必须为正整数")

    start = pd.Timestamp(oos_start)
    end = pd.Timestamp(oos_end)
    if start > end:
        raise ValueError("oos_start 不得晚于 oos_end")
    oos = dates[(dates >= start) & (dates <= end)]
    if oos.empty:
        raise ValueError("指定 OOS 区间没有交易日")

    folds: list[WalkForwardFold] = []
    for fold_number, test_start in enumerate(
        range(0, len(oos), int(step_dates)), start=1
    ):
        test_block = oos[test_start : test_start + int(test_dates)]
        if test_block.empty:
            continue
        first_test_position = int(dates.get_loc(test_block[0]))

        # Python 切片右端不包含：减去 outer gap 后，训练窗末日与测试首日
        # 之间严格留下指定数量的交易日。
        final_train_end = first_test_position - int(outer_gap_dates)
        final_train_start = final_train_end - int(rolling_train_dates)
        inner_valid_end = final_train_end
        inner_valid_start = inner_valid_end - int(inner_validation_dates)
        inner_train_end = inner_valid_start - int(inner_gap_dates)
        inner_train_start = inner_train_end - int(rolling_train_dates)
        if min(final_train_start, inner_valid_start, inner_train_start) < 0:
            raise ValueError(
                f"{test_block[0].date()} 之前历史不足，无法建立固定滚动窗口"
            )

        fold = WalkForwardFold(
            fold=fold_number,
            inner_train_dates=tuple(dates[inner_train_start:inner_train_end]),
            inner_validation_dates=tuple(dates[inner_valid_start:inner_valid_end]),
            final_train_dates=tuple(dates[final_train_start:final_train_end]),
            test_dates=tuple(test_block),
        )
        if len(fold.inner_train_dates) != int(rolling_train_dates):
            raise AssertionError("inner train 长度错误")
        if len(fold.final_train_dates) != int(rolling_train_dates):
            raise AssertionError("final train 长度错误")
        if not (
            max(fold.inner_train_dates) < min(fold.inner_validation_dates)
            and max(fold.final_train_dates) < min(fold.test_dates)
        ):
            raise AssertionError("walk-forward 日期顺序错误")
        folds.append(fold)

    covered = pd.DatetimeIndex(
        np.asarray(
            [date for fold in folds for date in fold.test_dates],
            dtype="datetime64[ns]",
        )
    )
    expected = pd.DatetimeIndex(np.asarray(oos, dtype="datetime64[ns]"))
    if len(covered) != len(expected) or not np.array_equal(
        covered.to_numpy(dtype="datetime64[ns]"),
        expected.to_numpy(dtype="datetime64[ns]"),
    ):
        missing = expected.difference(covered)
        unexpected = covered.difference(expected)
        duplicate_count = int(covered.duplicated().sum())
        raise AssertionError(
            "OOS folds 没有恰好覆盖目标区间: "
            f"expected={len(expected)}, covered={len(covered)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"duplicates={duplicate_count}"
        )
    return folds


def fit_lightgbm_final(
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
    feature_names: list[str],
    parameters: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[np.ndarray, int, dict[str, list[float]], np.ndarray, np.ndarray]:
    """用 inner validation 选树数，再在最近 720 日重训并预测。

    早停继续使用 V1 冻结的、日期等权的 Rank-Gaussian 标签 L2。函数额外
    返回最终模型的 gain/split importance，便于判断 MI 因子是否真正被模型
    使用；importance 不参与任何筛选，因此不会反向污染最终 holdout。
    """

    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("正式训练需要集群上的 LightGBM 3.3.5") from error

    if x_inner_train.shape[1] != len(feature_names):
        raise ValueError("feature_names 与特征矩阵列数不一致")
    for name, matrix in {
        "inner_train": x_inner_train,
        "inner_valid": x_inner_valid,
        "final_train": x_final_train,
        "test": x_test,
    }.items():
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError(f"{name} 必须是非空二维矩阵")

    inner_train = lgb.Dataset(
        x_inner_train,
        label=y_inner_train,
        weight=w_inner_train,
        feature_name=feature_names,
        free_raw_data=True,
    )
    inner_valid = lgb.Dataset(
        x_inner_valid,
        label=y_inner_valid,
        weight=w_inner_valid,
        feature_name=feature_names,
        reference=inner_train,
        free_raw_data=True,
    )
    history: dict[str, dict[str, list[float]]] = {}
    search_model = lgb.train(
        parameters,
        inner_train,
        num_boost_round=int(num_boost_round),
        valid_sets=[inner_valid],
        valid_names=["inner_valid"],
        callbacks=[
            lgb.early_stopping(int(early_stopping_rounds), first_metric_only=True),
            lgb.record_evaluation(history),
        ],
    )
    best_iteration = int(search_model.best_iteration or num_boost_round)

    final_dataset = lgb.Dataset(
        x_final_train,
        label=y_final_train,
        weight=w_final_train,
        feature_name=feature_names,
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
    gain = final_model.feature_importance(
        importance_type="gain", iteration=best_iteration
    ).astype(float)
    split = final_model.feature_importance(
        importance_type="split", iteration=best_iteration
    ).astype(float)
    return (
        np.asarray(prediction, dtype=float),
        best_iteration,
        compact_history,
        gain,
        split,
    )
