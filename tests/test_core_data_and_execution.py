"""数据时间语义与执行状态机的最小回归测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import build_stock_frames
from src.portfolio.execution_constraints import simulate_topk_open_execution


def _raw_stock_fixture() -> tuple[pd.DataFrame, pd.DatetimeIndex, list[pd.Timestamp]]:
    calendar = pd.bdate_range("2025-09-01", "2026-01-15")
    dates = list(calendar[-6:])
    rows = [
        [dates[0], 10.0, 10.0, 10.0, 10.0, 10.0, 1000, 10000, 1.0, 1, 0],
        [dates[1], 11.0, 11.0, 11.0, 11.0, 10.0, 1000, 11000, 1.0, 1, 0],
        [dates[2], 9.9, 9.9, 9.9, 9.9, 11.0, 1000, 9900, 1.0, 1, 0],
        [dates[3], 10.0, 10.1, 9.9, 10.0, 9.9, 1000, 10000, 1.0, 1, 0],
        [dates[4], np.nan, np.nan, np.nan, np.nan, 10.0, np.nan, np.nan, np.nan, 0, 0],
        [dates[5], 10.2, 10.3, 10.1, 10.2, 10.0, 1000, 10200, 1.0, 1, 0],
    ]
    raw = pd.DataFrame(
        rows,
        columns=[
            "date",
            "open",
            "high",
            "low",
            "close",
            "preclose",
            "volume",
            "amount",
            "turn",
            "tradestatus",
            "isST",
        ],
    )
    return raw, calendar, dates


def test_label_is_built_before_future_execution_filtering() -> None:
    """未来涨跌停只能影响执行，不能删除信号日或改写标签。"""

    raw, calendar, dates = _raw_stock_fixture()
    clean, _ = build_stock_frames(
        raw,
        calendar,
        stock="000001.SZ",
        ipo_date="2020-01-01",
        output_start=str(calendar.min().date()),
        signal_end=str(calendar.max().date()),
        execution_start=str(dates[0].date()),
        fetch_end=str(calendar.max().date()),
    )
    row = clean.loc[clean["date"].eq(dates[0])].iloc[0]
    assert np.isclose(row["label_1d"], 9.9 / 11.0 - 1.0)
    assert not bool(row["can_buy_t1"])
    assert not bool(row["can_sell_t2"])


def test_future_suspension_keeps_oos_signal_row() -> None:
    """未来停牌导致标签缺失时，当前信号仍必须留在完整 OOS 截面。"""

    raw, calendar, dates = _raw_stock_fixture()
    clean, _ = build_stock_frames(
        raw,
        calendar,
        stock="000001.SZ",
        ipo_date="2020-01-01",
        output_start=str(calendar.min().date()),
        signal_end=str(calendar.max().date()),
        execution_start=str(dates[0].date()),
        fetch_end=str(calendar.max().date()),
    )
    row = clean.loc[clean["date"].eq(dates[3])].iloc[0]
    assert pd.isna(row["label_1d"])
    assert not bool(row["can_buy_t1"])


def test_blocked_buy_stays_cash_and_blocked_sell_carries_position() -> None:
    """买不到时不补未来赢家，卖不出时持仓继续占用资金。"""

    calendar = pd.DatetimeIndex(
        pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])
    )
    targets = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "stock": ["A", "B"],
            "rank": [1, 1],
        }
    )
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-06", "2026-01-07", "2026-01-07", "2026-01-08", "2026-01-08"]
            ),
            "stock": ["A", "A", "B", "A", "B"],
            "open": [10.0, 9.0, 10.0, 8.0, 11.0],
            "has_valid_open": [True] * 5,
            "can_buy_open": [True] * 5,
            "can_sell_open": [True, False, True, True, True],
        }
    )
    result = simulate_topk_open_execution(targets, bars, calendar, top_k=1)
    assert result.events.iloc[1]["n_blocked_sell"] == 1
    assert result.events.iloc[1]["n_target_held_after"] == 0
    assert np.isclose(result.daily.iloc[1]["portfolio_return"], 8.0 / 9.0 - 1.0)
