"""Day 6：用可配置窗口生成第一版传统 OHLCV 候选因子库。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


FactorFunction = Callable[[pd.DataFrame], pd.Series]

MOMENTUM_WINDOWS = (2, 3, 5, 10, 20, 40, 60, 120, 250)
ROLLING_WINDOWS = (5, 10, 20, 40, 60, 120)
CHANGE_WINDOWS = (1, 5, 10, 20, 60)
PRICE_VOLUME_WINDOWS = (5, 10, 20, 40, 60)
INTRADAY_WINDOWS = (5, 10, 20)
QUANTILE_WINDOWS = (10, 20, 60)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)

REQUIRED_COLUMNS = {
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
}


@dataclass(frozen=True)
class FactorSpec:
    """保存一个候选因子的定义、经济解释和计算函数。"""

    name: str
    category: str
    formula: str
    window: int
    min_periods: int
    direction: str
    compute: FactorFunction = field(repr=False, compare=False)


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    epsilon: float = 1e-12,
) -> pd.Series:
    """安全相除；分母过小意味着比例没有稳定经济含义，返回 NaN。"""

    valid_denominator = denominator.where(denominator.abs() > epsilon)
    return numerator / valid_denominator


def _rolling_product_return(series: pd.Series, window: int) -> pd.Series:
    """复合最近 window 日收益，而不是直接相加近似。"""

    return (
        (1.0 + series)
        .rolling(window, min_periods=window)
        .apply(np.prod, raw=True)
        - 1.0
    )


def factor_specs() -> list[FactorSpec]:
    """定义 205 个候选因子；所有窗口只向历史方向滚动并包含当日。"""

    specs: list[FactorSpec] = []

    def add(
        name: str,
        category: str,
        formula: str,
        window: int,
        direction: str,
        compute: FactorFunction,
        min_periods: int | None = None,
    ) -> None:
        specs.append(
            FactorSpec(
                name=f"factor__{name}",
                category=category,
                formula=formula,
                window=window,
                min_periods=window if min_periods is None else min_periods,
                direction=direction,
                compute=compute,
            )
        )

    for window in MOMENTUM_WINDOWS:
        add(
            f"momentum_{window:03d}",
            "momentum",
            f"prod(1+r[t-j], j=0..{window - 1})-1",
            window,
            "趋势或反转均可能，交由样本外检验",
            lambda g, w=window: _rolling_product_return(g["_ret"], w),
        )

    for window in CHANGE_WINDOWS:
        add(
            f"reversal_{window:03d}",
            "reversal",
            f"-momentum_{window}",
            window,
            "数值越大代表过去跌幅越大",
            lambda g, w=window: -_rolling_product_return(g["_ret"], w),
        )

    for window in ROLLING_WINDOWS:
        add(
            f"volatility_{window:03d}",
            "volatility",
            f"std(r, {window})",
            window,
            "通常预期负向，但需验证",
            lambda g, w=window: g["_ret"].rolling(w, min_periods=w).std(),
        )
        add(
            f"downside_volatility_{window:03d}",
            "volatility",
            f"std(min(r,0), {window})",
            window,
            "通常预期负向",
            lambda g, w=window: g["_down_ret"].rolling(w, min_periods=w).std(),
        )
        add(
            f"mean_abs_return_{window:03d}",
            "volatility",
            f"mean(abs(r), {window})",
            window,
            "高值代表价格波动活跃",
            lambda g, w=window: g["_abs_ret"].rolling(w, min_periods=w).mean(),
        )
        add(
            f"realized_volatility_{window:03d}",
            "volatility",
            f"sqrt(sum(r^2, {window}))",
            window,
            "通常预期负向",
            lambda g, w=window: np.sqrt(
                g["_ret_sq"].rolling(w, min_periods=w).sum()
            ),
        )
        add(
            f"parkinson_volatility_{window:03d}",
            "volatility",
            f"sqrt(mean(log(high/low)^2,{window})/(4log(2)))",
            window,
            "日内区间越宽，波动越高",
            lambda g, w=window: np.sqrt(
                g["_log_high_low_sq"].rolling(w, min_periods=w).mean()
                / (4.0 * np.log(2.0))
            ),
        )

    for window in ROLLING_WINDOWS:
        add(
            f"turnover_mean_{window:03d}",
            "turnover",
            f"mean(turnover, {window})",
            window,
            "交易活跃度，方向不预设",
            lambda g, w=window: g["_turn"].rolling(w, min_periods=w).mean(),
        )
        add(
            f"turnover_std_{window:03d}",
            "turnover",
            f"std(turnover, {window})",
            window,
            "换手不稳定性，方向不预设",
            lambda g, w=window: g["_turn"].rolling(w, min_periods=w).std(),
        )
        add(
            f"turnover_ratio_{window:03d}",
            "turnover",
            f"turnover[t]/mean(turnover,{window})",
            window,
            "大于1表示当日异常活跃",
            lambda g, w=window: safe_divide(
                g["_turn"], g["_turn"].rolling(w, min_periods=w).mean()
            ),
        )

    for window in CHANGE_WINDOWS:
        add(
            f"turnover_change_{window:03d}",
            "turnover",
            f"turnover[t]/turnover[t-{window}]-1",
            window + 1,
            "高值表示换手上升",
            lambda g, w=window: safe_divide(g["_turn"], g["_turn"].shift(w)) - 1.0,
        )

    add(
        "log_volume",
        "volume",
        "log(1+volume)",
        1,
        "规模型活跃度，后续需中性化",
        lambda g: g["_log_volume"],
    )
    for window in ROLLING_WINDOWS:
        add(
            f"volume_mean_{window:03d}",
            "volume",
            f"mean(log(1+volume), {window})",
            window,
            "成交活跃度，方向不预设",
            lambda g, w=window: g["_log_volume"].rolling(w, min_periods=w).mean(),
        )
        add(
            f"volume_std_{window:03d}",
            "volume",
            f"std(log(1+volume), {window})",
            window,
            "成交量不稳定性",
            lambda g, w=window: g["_log_volume"].rolling(w, min_periods=w).std(),
        )
        add(
            f"volume_ratio_{window:03d}",
            "volume",
            f"volume[t]/mean(volume,{window})",
            window,
            "大于1表示相对放量",
            lambda g, w=window: safe_divide(
                g["_volume"], g["_volume"].rolling(w, min_periods=w).mean()
            ),
        )

    for window in CHANGE_WINDOWS:
        add(
            f"log_volume_change_{window:03d}",
            "volume",
            f"log(1+volume[t])-log(1+volume[t-{window}])",
            window + 1,
            "高值表示相对放量",
            lambda g, w=window: g["_log_volume"] - g["_log_volume"].shift(w),
        )

    for window in ROLLING_WINDOWS:
        add(
            f"amihud_{window:03d}",
            "liquidity",
            f"mean(abs(r)/amount, {window})*1e8",
            window,
            "高值代表流动性较差",
            lambda g, w=window: g["_amihud"].rolling(w, min_periods=w).mean(),
        )
        add(
            f"log_amount_mean_{window:03d}",
            "liquidity",
            f"mean(log(1+amount), {window})",
            window,
            "高值通常代表流动性较好",
            lambda g, w=window: g["_log_amount"].rolling(w, min_periods=w).mean(),
        )
        add(
            f"amount_ratio_{window:03d}",
            "liquidity",
            f"amount[t]/mean(amount,{window})",
            window,
            "大于1表示成交额异常放大",
            lambda g, w=window: safe_divide(
                g["_amount"], g["_amount"].rolling(w, min_periods=w).mean()
            ),
        )
        add(
            f"turnover_impact_{window:03d}",
            "liquidity",
            f"mean(abs(r)/turnover, {window})",
            window,
            "高值代表单位换手价格冲击更大",
            lambda g, w=window: g["_turnover_impact"]
            .rolling(w, min_periods=w)
            .mean(),
        )

    intraday = (
        ("open_close_return", "(close-open)/open", "收盘相对开盘强弱", "_oc_return"),
        ("overnight_gap", "open/preclose-1", "隔夜跳空方向", "_gap"),
        ("intraday_range", "(high-low)/preclose", "日内振幅", "_range"),
        ("body_over_range", "(close-open)/(high-low)", "K线实体方向", "_body"),
        ("close_location", "(close-low)/(high-low)", "收盘在日内区间的位置", "_close_location"),
        ("upper_shadow", "(high-max(open,close))/preclose", "上影线强度", "_upper_shadow"),
        ("lower_shadow", "(min(open,close)-low)/preclose", "下影线强度", "_lower_shadow"),
        ("vwap_deviation", "amount/volume/close-1", "成交均价相对收盘偏离", "_vwap_deviation"),
    )
    for name, formula, direction, column in intraday:
        add(
            name,
            "intraday_structure",
            formula,
            1,
            direction,
            lambda g, c=column: g[c],
        )

    for window in INTRADAY_WINDOWS:
        for name, column in (
            ("body_mean", "_body"),
            ("range_mean", "_range"),
            ("close_location_mean", "_close_location"),
            ("gap_mean", "_gap"),
        ):
            add(
                f"{name}_{window:03d}",
                "intraday_structure",
                f"mean({name.removesuffix('_mean')},{window})",
                window,
                "聚合日内价格结构，方向不预设",
                lambda g, w=window, c=column: g[c].rolling(w, min_periods=w).mean(),
            )

    for window in PRICE_VOLUME_WINDOWS:
        add(
            f"return_volume_corr_{window:03d}",
            "price_volume",
            f"corr(r,delta_log_volume,{window})",
            window,
            "价量同向性，方向不预设",
            lambda g, w=window: g["_ret"]
            .rolling(w, min_periods=w)
            .corr(g["_log_volume_change_1d"]),
        )
        add(
            f"return_volume_cov_{window:03d}",
            "price_volume",
            f"cov(r,delta_log_volume,{window})",
            window,
            "价量联合变化，受尺度影响",
            lambda g, w=window: g["_ret"]
            .rolling(w, min_periods=w)
            .cov(g["_log_volume_change_1d"]),
        )
        add(
            f"signed_turnover_{window:03d}",
            "price_volume",
            f"mean(sign(r)*turnover,{window})",
            window,
            "正值代表上涨日换手占优",
            lambda g, w=window: g["_signed_turnover"]
            .rolling(w, min_periods=w)
            .mean(),
        )
        add(
            f"abs_return_turnover_corr_{window:03d}",
            "price_volume",
            f"corr(abs(r),turnover,{window})",
            window,
            "衡量大波动是否伴随高换手",
            lambda g, w=window: g["_abs_ret"]
            .rolling(w, min_periods=w)
            .corr(g["_turn"]),
        )

    for window in ROLLING_WINDOWS:
        add(
            f"return_sharpe_{window:03d}",
            "relative_strength",
            f"mean(r,{window})/std(r,{window})",
            window,
            "高值代表单位波动收益较高",
            lambda g, w=window: safe_divide(
                g["_ret"].rolling(w, min_periods=w).mean(),
                g["_ret"].rolling(w, min_periods=w).std(),
            ),
        )
        add(
            f"up_day_fraction_{window:03d}",
            "relative_strength",
            f"mean(1(r>0),{window})",
            window,
            "高值代表上涨日占比较高",
            lambda g, w=window: g["_up_day"].rolling(w, min_periods=w).mean(),
        )
        add(
            f"gain_loss_ratio_{window:03d}",
            "relative_strength",
            f"mean(max(r,0),{window})/mean(abs(min(r,0)),{window})",
            window,
            "高值代表上涨收益相对占优",
            lambda g, w=window: safe_divide(
                g["_gain"].rolling(w, min_periods=w).mean(),
                g["_loss"].rolling(w, min_periods=w).mean(),
            ),
        )

    for window in ROLLING_WINDOWS:
        add(
            f"return_skew_{window:03d}",
            "distribution",
            f"skew(r,{window})",
            window,
            "收益分布不对称性，可能非单调",
            lambda g, w=window: g["_ret"].rolling(w, min_periods=w).skew(),
        )
        add(
            f"return_range_{window:03d}",
            "distribution",
            f"max(r,{window})-min(r,{window})",
            window,
            "高值代表近期极端收益跨度大",
            lambda g, w=window: g["_ret"].rolling(w, min_periods=w).max()
            - g["_ret"].rolling(w, min_periods=w).min(),
        )

    for window in (10, 20, 40, 60, 120):
        add(
            f"return_kurtosis_{window:03d}",
            "distribution",
            f"kurtosis(r,{window})",
            window,
            "高值代表厚尾与极端收益更强",
            lambda g, w=window: g["_ret"].rolling(w, min_periods=w).kurt(),
        )

    for window in QUANTILE_WINDOWS:
        for quantile in QUANTILES:
            code = int(round(quantile * 100))
            add(
                f"return_q{code:02d}_{window:03d}",
                "distribution",
                f"quantile(r,{quantile:.2f},{window})",
                window,
                "描述局部收益分布位置，方向不预设",
                lambda g, w=window, q=quantile: g["_ret"]
                .rolling(w, min_periods=w)
                .quantile(q),
            )

    return specs


def _prepare_one_stock(group: pd.DataFrame) -> pd.DataFrame:
    """为一只股票准备可复用中间量，减少重复公式计算。"""

    g = group.sort_values("date").copy()
    g["_ret"] = pd.to_numeric(g["return_1d"], errors="coerce")
    g["_abs_ret"] = g["_ret"].abs()
    g["_down_ret"] = g["_ret"].clip(upper=0.0)
    g["_ret_sq"] = g["_ret"].pow(2)
    g["_gain"] = g["_ret"].clip(lower=0.0)
    g["_loss"] = -g["_ret"].clip(upper=0.0)
    g["_up_day"] = g["_ret"].gt(0).astype(float)

    for column in (
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "turnover_pct",
    ):
        g[column] = pd.to_numeric(g[column], errors="coerce")

    g["_volume"] = g["volume"]
    g["_amount"] = g["amount"]
    g["_turn"] = g["turnover_pct"]
    g["_log_volume"] = np.log1p(g["volume"].clip(lower=0.0))
    g["_log_amount"] = np.log1p(g["amount"].clip(lower=0.0))
    g["_log_volume_change_1d"] = g["_log_volume"].diff()
    g["_amihud"] = safe_divide(g["_abs_ret"], g["amount"]) * 1e8
    g["_turnover_impact"] = safe_divide(g["_abs_ret"], g["_turn"])
    g["_signed_turnover"] = np.sign(g["_ret"]) * g["_turn"]

    g["_log_high_low_sq"] = np.log(safe_divide(g["high"], g["low"])).pow(2)
    g["_oc_return"] = safe_divide(g["close"] - g["open"], g["open"])
    g["_gap"] = safe_divide(g["open"], g["preclose"]) - 1.0
    g["_range"] = safe_divide(g["high"] - g["low"], g["preclose"])
    g["_body"] = safe_divide(g["close"] - g["open"], g["high"] - g["low"])
    g["_close_location"] = safe_divide(
        g["close"] - g["low"], g["high"] - g["low"]
    )
    g["_upper_shadow"] = safe_divide(
        g["high"] - g[["open", "close"]].max(axis=1), g["preclose"]
    )
    g["_lower_shadow"] = safe_divide(
        g[["open", "close"]].min(axis=1) - g["low"], g["preclose"]
    )
    average_price = safe_divide(g["amount"], g["volume"])
    g["_vwap_deviation"] = safe_divide(average_price, g["close"]) - 1.0
    return g


def build_factor_catalog() -> pd.DataFrame:
    """返回因子元数据表，不需要读取或计算股票数据。"""

    return pd.DataFrame(
        [
            {
                "factor": spec.name,
                "category": spec.category,
                "formula": spec.formula,
                "window": spec.window,
                "min_periods": spec.min_periods,
                "direction": spec.direction,
            }
            for spec in factor_specs()
        ]
    )


def generate_factor_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """按股票、按日期生成传统因子；原始行顺序不会用于时间计算。"""

    missing = REQUIRED_COLUMNS.difference(panel.columns)
    if missing:
        raise ValueError(f"缺少因子计算字段：{sorted(missing)}")

    base = panel.copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base = base.dropna(subset=["date", "stock"])
    base = base.sort_values(["stock", "date"]).reset_index(drop=True)
    specs = factor_specs()
    pieces: list[pd.DataFrame] = []

    for _, group in base.groupby("stock", sort=False):
        working = _prepare_one_stock(group)
        factor_values: dict[str, np.ndarray] = {}
        for spec in specs:
            values = pd.to_numeric(spec.compute(working), errors="coerce")
            values = values.where(np.isfinite(values))
            # float32 足以保存因子排序信息，并显著降低全市场宽表内存占用。
            factor_values[spec.name] = values.to_numpy(dtype=np.float32)
        factor_frame = pd.DataFrame(factor_values, index=group.index)
        pieces.append(pd.concat([group, factor_frame], axis=1))

    if not pieces:
        return pd.DataFrame(columns=[*base.columns, *(spec.name for spec in specs)])
    return pd.concat(pieces, ignore_index=True).sort_values(["date", "stock"])


def generate_selected_factor_panel(
    panel: pd.DataFrame,
    factor_names: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """只计算指定因子，适合从宽因子库中复算少量候选表达式。

    该函数与 :func:`generate_factor_panel` 使用完全相同的因子定义，区别只是
    不再生成未入选的列。Day 15 因此可以在不重建 205 因子宽表的前提下，
    用原始 OHLCV 数据严格复算候选因子值。
    """

    requested = list(factor_names)
    if not requested:
        raise ValueError("factor_names 不能为空")
    if len(requested) != len(set(requested)):
        raise ValueError("factor_names 不能包含重复名称")

    missing = REQUIRED_COLUMNS.difference(panel.columns)
    if missing:
        raise ValueError(f"缺少因子计算字段：{sorted(missing)}")

    spec_map = {spec.name: spec for spec in factor_specs()}
    unknown = sorted(set(requested).difference(spec_map))
    if unknown:
        raise ValueError(f"未知因子：{unknown}")

    base = panel.copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base = base.dropna(subset=["date", "stock"])
    base = base.sort_values(["stock", "date"]).reset_index(drop=True)
    pieces: list[pd.DataFrame] = []

    for _, group in base.groupby("stock", sort=False):
        working = _prepare_one_stock(group)
        factor_values: dict[str, np.ndarray] = {}
        for name in requested:
            values = pd.to_numeric(spec_map[name].compute(working), errors="coerce")
            values = values.where(np.isfinite(values))
            # 候选列仍保存为 float32，与 Day 6 全量因子文件保持完全一致。
            factor_values[name] = values.to_numpy(dtype=np.float32)
        factor_frame = pd.DataFrame(factor_values, index=group.index)
        pieces.append(pd.concat([group, factor_frame], axis=1))

    if not pieces:
        return pd.DataFrame(columns=[*base.columns, *requested])
    return pd.concat(pieces, ignore_index=True).sort_values(["date", "stock"])
