"""从单股未过滤日线构造信号 panel、标签与执行行情。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.baostock import limit_rate
from src.portfolio.execution_constraints import add_open_execution_flags


CLEAN_COLUMNS = [
    "date",
    "stock",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "turnover_pct",
    "return_1d",
    "listing_age",
    "label_1d",
    "can_buy_t1",
    "can_sell_t2",
]

EXECUTION_COLUMNS = [
    "date",
    "stock",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "tradestatus",
    "isST",
    "limit_rate",
    "limit_up_price",
    "limit_down_price",
    "has_valid_open",
    "can_buy_open",
    "can_sell_open",
]


def build_stock_frames(
    raw: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    *,
    stock: str,
    ipo_date: str,
    output_start: str,
    signal_end: str,
    execution_start: str,
    fetch_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造单股信号 panel 与执行行情，且不按未来状态筛信号行。

    ``label_1d`` 必须先在完整交易日历上按 ``t+1``、``t+2`` 开盘价计算，
    再应用信号日 ``t`` 的股票池规则。未来的买卖限制只保存为执行字段；
    若未来停牌导致标签缺失，信号行仍保留供 OOS 排名。
    """

    period = trading_calendar[
        (trading_calendar >= pd.Timestamp(output_start))
        & (trading_calendar <= pd.Timestamp(fetch_end))
    ]
    if len(period) == 0:
        raise ValueError("指定区间没有交易日")

    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    numeric = [
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
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.set_index("date").reindex(period)
    frame["stock"] = stock
    frame["date"] = frame.index

    calendar_positions = np.searchsorted(trading_calendar.values, period.values)
    ipo_position = np.searchsorted(
        trading_calendar.values, np.datetime64(pd.Timestamp(ipo_date))
    )
    frame["listing_age"] = calendar_positions - ipo_position + 1

    execution_all = add_open_execution_flags(
        frame.reset_index(drop=True)[
            [
                "date",
                "stock",
                "open",
                "high",
                "low",
                "close",
                "preclose",
                "tradestatus",
                "isST",
            ]
        ]
    )
    execution_all.index = period

    # 股票池只使用 t 日已知信息。触及涨跌停作为用户冻结的简化 eligibility
    # 口径；未来开盘是否可成交则留给执行状态机处理。
    rates = limit_rate(stock, period)
    tolerance = 0.005
    up_move = frame["high"] / frame["preclose"] - 1.0
    down_move = frame["low"] / frame["preclose"] - 1.0
    hit_limit = (up_move >= rates - tolerance) | (down_move <= -rates + tolerance)
    price_columns = ["open", "high", "low", "close", "preclose"]
    finite = np.isfinite(frame[price_columns + ["volume", "amount", "turn"]]).all(
        axis=1
    )
    valid_ohlc = (
        frame[price_columns].gt(0).all(axis=1)
        & frame["volume"].ge(0)
        & frame["amount"].ge(0)
        & frame["high"].ge(frame[["open", "low", "close"]].max(axis=1))
        & frame["low"].le(frame[["open", "high", "close"]].min(axis=1))
    )
    clean_day = (
        frame["tradestatus"].eq(1)
        & frame["isST"].eq(0)
        & frame["listing_age"].ge(60)
        & finite
        & valid_ohlc
        & ~hit_limit.fillna(True)
    )

    frame["label_1d"] = frame["open"].shift(-2) / frame["open"].shift(-1) - 1.0
    frame["can_buy_t1"] = execution_all["can_buy_open"].shift(
        -1, fill_value=False
    ).astype(bool)
    frame["can_sell_t2"] = execution_all["can_sell_open"].shift(
        -2, fill_value=False
    ).astype(bool)
    frame["turnover_pct"] = frame["turn"]
    frame["return_1d"] = frame["close"] / frame["preclose"] - 1.0

    keep = clean_day & frame["date"].between(output_start, signal_end)
    clean = frame.loc[keep, CLEAN_COLUMNS].reset_index(drop=True)
    execution = execution_all.loc[
        execution_all["date"].between(execution_start, fetch_end), EXECUTION_COLUMNS
    ].reset_index(drop=True)
    return clean, execution
