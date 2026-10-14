"""A 股开盘执行约束与有状态 Top-K 组合模拟。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _round_half_up_cent(values: np.ndarray) -> np.ndarray:
    """把正价格按半分向上舍入到人民币分，近似交易所涨跌停价取整。"""

    return np.floor(values * 100.0 + 0.5 + 1e-10) / 100.0


def add_open_execution_flags(bars: pd.DataFrame) -> pd.DataFrame:
    """根据未过滤日线生成开盘可买、可卖标记。

    普通主板按 10%，科创板按 20%，创业板自 2020-08-24 起按 20%，
    北交所按 30%，ST 按 5%。这里只模拟已经上市 60 个交易日之后的候选，
    因而不处理 IPO 初期无涨跌停的特殊阶段。
    """

    required = {"date", "stock", "open", "preclose", "tradestatus", "isST"}
    missing = required.difference(bars.columns)
    if missing:
        raise KeyError(f"执行行情缺少字段: {sorted(missing)}")

    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["open", "preclose", "tradestatus", "isST"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    stock = frame["stock"].astype(str)
    number = stock.str.split(".", regex=False).str[0]
    market = stock.str.split(".", regex=False).str[-1]
    rate = np.full(len(frame), 0.10, dtype=float)
    rate[market.eq("BJ").to_numpy()] = 0.30
    rate[number.str.startswith(("688", "689")).to_numpy()] = 0.20
    gem = number.str.startswith(("300", "301")) & frame["date"].ge("2020-08-24")
    rate[gem.to_numpy()] = 0.20
    # ST 的 5% 限制优先于所属板块；ST 股票不能新买，但已有仓位仍可卖。
    rate[frame["isST"].eq(1).to_numpy()] = 0.05

    preclose = frame["preclose"].to_numpy(dtype=float)
    frame["limit_rate"] = rate
    frame["limit_up_price"] = _round_half_up_cent(preclose * (1.0 + rate))
    frame["limit_down_price"] = _round_half_up_cent(preclose * (1.0 - rate))

    valid_open = (
        frame["date"].notna()
        & frame["tradestatus"].eq(1)
        & frame["open"].gt(0)
        & frame["preclose"].gt(0)
    )
    # 半分容差只用于处理浮点表示；不是把“接近涨停”都当作封板。
    at_limit_up = frame["open"].ge(frame["limit_up_price"] - 0.0051)
    at_limit_down = frame["open"].le(frame["limit_down_price"] + 0.0051)
    frame["has_valid_open"] = valid_open
    frame["can_buy_open"] = valid_open & frame["isST"].eq(0) & ~at_limit_up
    frame["can_sell_open"] = valid_open & ~at_limit_down
    return frame


@dataclass
class ExecutionResult:
    """一次有状态回测的逐日收益、执行诊断与最终持仓。"""

    daily: pd.DataFrame
    events: pd.DataFrame
    terminal_positions: pd.DataFrame


def _next_session(date: pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """返回严格晚于给定日期的下一交易日。"""

    position = int(calendar.searchsorted(date, side="right"))
    if position >= len(calendar):
        raise ValueError(f"交易日历在 {date.date()} 之后没有足够日期")
    return pd.Timestamp(calendar[position])


def simulate_topk_open_execution(
    targets: pd.DataFrame,
    execution_bars: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    *,
    top_k: int = 100,
) -> ExecutionResult:
    """模拟 t 收盘选股、t+1 开盘调仓、t+2 开盘估值的等权组合。

    规则如下：

    1. Top-K 中开盘涨停、ST 或停牌的股票不能新买，1% 席位留在现金；
    2. 非目标股票若开盘跌停或停牌则不能卖，原持仓继续占用资金；
    3. 被困持仓导致现金不足时，所有可买缺口按同一比例缩小，绝不加杠杆；
    4. 停牌持仓用最近一次有效开盘价估值，不虚构零收益或强制卖出；
    5. 本函数不扣交易成本，用来隔离“执行可行性”本身的影响。
    """

    required_targets = {"date", "stock", "rank"}
    missing_targets = required_targets.difference(targets.columns)
    if missing_targets:
        raise KeyError(f"目标持仓缺少字段: {sorted(missing_targets)}")
    required_bars = {
        "date",
        "stock",
        "open",
        "has_valid_open",
        "can_buy_open",
        "can_sell_open",
    }
    missing_bars = required_bars.difference(execution_bars.columns)
    if missing_bars:
        raise KeyError(f"执行行情缺少字段: {sorted(missing_bars)}")
    if top_k < 1:
        raise ValueError("top_k 必须为正整数")

    target_frame = targets.copy()
    target_frame["date"] = pd.to_datetime(target_frame["date"])
    target_frame["stock"] = target_frame["stock"].astype(str)
    target_frame = target_frame.sort_values(["date", "rank", "stock"])
    if target_frame.duplicated(["date", "stock"]).any():
        raise ValueError("目标持仓的 (date, stock) 必须唯一")
    counts = target_frame.groupby("date", observed=True)["stock"].size()
    if not counts.eq(top_k).all():
        bad = counts.loc[~counts.eq(top_k)].head().to_dict()
        raise ValueError(f"每个信号日必须恰有 {top_k} 只股票: {bad}")

    bars = execution_bars.copy()
    bars["date"] = pd.to_datetime(bars["date"])
    bars["stock"] = bars["stock"].astype(str)
    if bars.duplicated(["date", "stock"]).any():
        raise ValueError("执行行情的 (date, stock) 必须唯一")
    bar_by_date = {
        pd.Timestamp(date): block.set_index("stock")
        for date, block in bars.groupby("date", sort=False, observed=True)
    }

    calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).drop_duplicates().sort_values()
    signal_dates = pd.DatetimeIndex(target_frame["date"].drop_duplicates()).sort_values()
    if len(signal_dates) == 0:
        raise ValueError("目标持仓为空")
    execution_dates = [_next_session(date, calendar) for date in signal_dates]
    final_exit_date = _next_session(execution_dates[-1], calendar)

    positions: dict[str, float] = {}  # stock -> shares
    last_prices: dict[str, float] = {}
    cash = 1.0
    previous_nav = 1.0
    events: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []

    def row_for(date: pd.Timestamp, stock_name: str) -> pd.Series | None:
        block = bar_by_date.get(pd.Timestamp(date))
        if block is None or stock_name not in block.index:
            return None
        row = block.loc[stock_name]
        if isinstance(row, pd.DataFrame):
            raise ValueError(f"执行行情重复: {date.date()} {stock_name}")
        return row

    def mark_to_open(date: pd.Timestamp) -> tuple[float, list[str]]:
        stale: list[str] = []
        value = cash
        for stock_name, shares in positions.items():
            row = row_for(date, stock_name)
            price = np.nan if row is None else float(row["open"])
            if row is not None and bool(row["has_valid_open"]) and np.isfinite(price) and price > 0:
                last_prices[stock_name] = price
            else:
                stale.append(stock_name)
                price = last_prices.get(stock_name, np.nan)
            if not np.isfinite(price) or price <= 0:
                raise ValueError(f"持仓 {stock_name} 在 {date.date()} 无可用估值价格")
            value += shares * price
        return float(value), stale

    def rebalance(signal_date: pd.Timestamp, execution_date: pd.Timestamp, nav: float) -> dict[str, object]:
        nonlocal cash
        target_block = target_frame.loc[target_frame["date"].eq(signal_date)]
        target_names = target_block.sort_values("rank")["stock"].tolist()
        target_set = set(target_names)
        desired_value = nav / top_k
        sells = 0.0
        buys = 0.0
        blocked_sells: list[str] = []
        blocked_buys: list[str] = []

        # 先卖出不再入选或超过目标权重的持仓；跌停/停牌时原股数保持不变。
        for stock_name in list(positions):
            row = row_for(execution_date, stock_name)
            price = last_prices[stock_name]
            current_value = positions[stock_name] * price
            wanted = desired_value if stock_name in target_set else 0.0
            if current_value <= wanted + 1e-14:
                continue
            can_sell = row is not None and bool(row["can_sell_open"])
            if not can_sell:
                blocked_sells.append(stock_name)
                continue
            sell_value = current_value - wanted
            positions[stock_name] -= sell_value / price
            cash += sell_value
            sells += sell_value
            if positions[stock_name] <= 1e-14:
                positions.pop(stock_name, None)

        # 再计算可买缺口；现金不足时同比例缩小，避免被困仓位诱发隐含杠杆。
        deficits: dict[str, float] = {}
        for stock_name in target_names:
            row = row_for(execution_date, stock_name)
            price = np.nan if row is None else float(row["open"])
            current_value = positions.get(stock_name, 0.0) * (
                last_prices.get(stock_name, price) if positions.get(stock_name, 0.0) > 0 else price
            )
            deficit = max(0.0, desired_value - current_value)
            if deficit <= 1e-14:
                continue
            can_buy = (
                row is not None
                and bool(row["can_buy_open"])
                and np.isfinite(price)
                and price > 0
            )
            if can_buy:
                deficits[stock_name] = deficit
            else:
                blocked_buys.append(stock_name)

        requested = float(sum(deficits.values()))
        scale = min(1.0, cash / requested) if requested > 0 else 0.0
        for stock_name, deficit in deficits.items():
            row = row_for(execution_date, stock_name)
            assert row is not None
            price = float(row["open"])
            buy_value = deficit * scale
            if buy_value <= 1e-14:
                continue
            positions[stock_name] = positions.get(stock_name, 0.0) + buy_value / price
            last_prices[stock_name] = price
            cash -= buy_value
            buys += buy_value
        if cash < -1e-10:
            raise AssertionError("执行后现金为负，出现隐含杠杆")
        cash = max(0.0, cash)

        trapped_non_target = [name for name in blocked_sells if name not in target_set]
        trapped_value = sum(
            positions[name] * last_prices[name] for name in trapped_non_target
        )
        invested_value = sum(
            shares * last_prices[name] for name, shares in positions.items()
        )
        held_targets = sum(name in positions for name in target_names)
        return {
            "signal_date": signal_date,
            "entry_date": execution_date,
            "n_target": top_k,
            "n_held_after": len(positions),
            "n_target_held_after": held_targets,
            "n_blocked_buy": len(blocked_buys),
            "n_blocked_sell": len(blocked_sells),
            "n_trapped_non_target": len(trapped_non_target),
            "blocked_buy_names": ";".join(blocked_buys),
            "blocked_sell_names": ";".join(blocked_sells),
            "trapped_non_target_names": ";".join(trapped_non_target),
            "cash_weight_after": cash / nav,
            "invested_weight_after": invested_value / nav,
            "trapped_non_target_weight": trapped_value / nav,
            "buy_notional_weight": buys / nav,
            "sell_notional_weight": sells / nav,
            "one_way_turnover": max(buys, sells) / nav,
        }

    previous_signal: pd.Timestamp | None = None
    previous_entry: pd.Timestamp | None = None
    previous_event: dict[str, object] | None = None
    for signal_date, execution_date in zip(signal_dates, execution_dates):
        nav, stale = mark_to_open(execution_date)
        if previous_signal is not None and previous_entry is not None and previous_event is not None:
            daily_rows.append(
                {
                    **previous_event,
                    "exit_date": execution_date,
                    "portfolio_return": nav / previous_nav - 1.0,
                    "nav": nav,
                    "n_stale_marks_at_exit": len(stale),
                    "stale_mark_names": ";".join(stale),
                }
            )
        event = rebalance(pd.Timestamp(signal_date), execution_date, nav)
        events.append(event)
        previous_nav = nav
        previous_signal = pd.Timestamp(signal_date)
        previous_entry = execution_date
        previous_event = event

    nav, stale = mark_to_open(final_exit_date)
    assert previous_event is not None
    daily_rows.append(
        {
            **previous_event,
            "exit_date": final_exit_date,
            "portfolio_return": nav / previous_nav - 1.0,
            "nav": nav,
            "n_stale_marks_at_exit": len(stale),
            "stale_mark_names": ";".join(stale),
        }
    )

    terminal_rows = []
    for stock_name, shares in positions.items():
        row = row_for(final_exit_date, stock_name)
        terminal_rows.append(
            {
                "stock": stock_name,
                "shares": shares,
                "mark_price": last_prices[stock_name],
                "market_value": shares * last_prices[stock_name],
                "weight": shares * last_prices[stock_name] / nav,
                "can_sell_at_final_exit": bool(row["can_sell_open"]) if row is not None else False,
            }
        )
    terminal_columns = [
        "stock",
        "shares",
        "mark_price",
        "market_value",
        "weight",
        "can_sell_at_final_exit",
    ]
    return ExecutionResult(
        daily=pd.DataFrame(daily_rows),
        events=pd.DataFrame(events),
        terminal_positions=pd.DataFrame(terminal_rows, columns=terminal_columns),
    )


def summarize_execution_backtest(daily: pd.DataFrame) -> dict[str, float]:
    """汇总执行约束后的 0 bps 组合表现和受阻频率。"""

    if daily.empty or "portfolio_return" not in daily:
        raise ValueError("逐日执行结果为空")
    returns = daily["portfolio_return"].to_numpy(dtype=float)
    wealth = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    years = len(returns) / 252.0
    annual = float(wealth[-1] ** (1.0 / years) - 1.0)
    volatility = float(np.std(returns, ddof=1))
    sharpe = float(np.mean(returns) / volatility * np.sqrt(252.0)) if volatility > 0 else np.nan
    max_drawdown = float(drawdown.min())
    return {
        "n_dates": float(len(daily)),
        "period_return": float(wealth[-1] - 1.0),
        "annual_return": annual,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": annual / abs(max_drawdown) if max_drawdown < 0 else np.nan,
        "win_rate": float(np.mean(returns > 0.0)),
        "mean_daily_return": float(np.mean(returns)),
        "mean_one_way_turnover": float(daily["one_way_turnover"].mean()),
        "mean_cash_weight": float(daily["cash_weight_after"].mean()),
        "mean_invested_weight": float(daily["invested_weight_after"].mean()),
        "mean_blocked_buy_count": float(daily["n_blocked_buy"].mean()),
        "days_with_blocked_buy_ratio": float(daily["n_blocked_buy"].gt(0).mean()),
        "mean_blocked_sell_count": float(daily["n_blocked_sell"].mean()),
        "days_with_blocked_sell_ratio": float(daily["n_blocked_sell"].gt(0).mean()),
        "mean_trapped_non_target_weight": float(daily["trapped_non_target_weight"].mean()),
        "days_with_stale_mark_ratio": float(daily["n_stale_marks_at_exit"].gt(0).mean()),
    }
