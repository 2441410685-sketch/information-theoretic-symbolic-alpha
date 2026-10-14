"""横截面预处理与严格时间切分的核心测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import build_walk_forward_folds_range
from src.preprocessing import neutralize_matrix, rank_gaussianize_matrix


def test_neutralization_removes_size_and_industry_exposure() -> None:
    """中性残差必须与市值列正交，且行业内均值为零。"""

    rng = np.random.default_rng(21)
    size = rng.normal(size=400)
    industry = np.repeat(["A", "B", "C", "D"], 100)
    industry_effect = pd.Series(industry).map(
        {"A": -2.0, "B": 1.0, "C": 3.0, "D": -1.0}
    ).to_numpy()
    factor = 2.5 * size + industry_effect + rng.normal(scale=0.1, size=400)
    residual = neutralize_matrix(factor, size, industry)[:, 0]
    assert abs(np.corrcoef(residual, size)[0, 1]) < 1e-12
    assert pd.Series(residual).groupby(industry).mean().abs().max() < 1e-12


def test_rank_gaussianization_is_monotone_and_well_scaled() -> None:
    """Rank Gaussian 保序，并产生近似零均值单位尺度。"""

    transformed = rank_gaussianize_matrix(np.arange(1.0, 201.0))[:, 0]
    assert np.all(np.diff(transformed) > 0)
    assert abs(transformed.mean()) < 1e-12
    assert 0.97 < transformed.std(ddof=0) < 1.03


def test_walk_forward_has_gap_and_exact_oos_coverage() -> None:
    """测试期不能进入训练或早停窗口，且 OOS 日期不重不漏。"""

    dates = pd.bdate_range("2020-01-01", periods=1_100)
    folds = build_walk_forward_folds_range(
        dates,
        oos_start=dates[900],
        oos_end=dates[999],
        rolling_train_dates=500,
        inner_validation_dates=40,
        inner_gap_dates=2,
        outer_gap_dates=2,
        test_dates=20,
        step_dates=20,
    )
    covered = [date for fold in folds for date in fold.test_dates]
    assert covered == list(dates[900:1000])
    for fold in folds:
        first_test = dates.get_loc(min(fold.test_dates))
        last_train = dates.get_loc(max(fold.final_train_dates))
        assert first_test - last_train - 1 == 2
        assert set(fold.final_train_dates).isdisjoint(fold.test_dates)
