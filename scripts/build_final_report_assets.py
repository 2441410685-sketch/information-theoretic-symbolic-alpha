"""从冻结的 V1.1 结果生成研究报告图与派生统计量。

这个脚本只读取 ``results/final`` 中的结果，不重新训练、不重新选因子。
这样可以把“模型产生结果”和“报告展示结果”严格分开。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "final"
FIGURE_DIR = RESULT_DIR / "figures"
TRADING_DAYS = 252

DISPLAY_NAMES = {
    "traditional": "Traditional",
    "traditional_plus_rankic": "Traditional + RankIC-GP",
    "traditional_plus_mi": "Traditional + MI-GP",
    "traditional_plus_union": "Traditional + Union",
    "traditional_plus_cmi_core": "Traditional + CMI core",
    "union_only": "Union only",
    "mi_only": "MI only",
}


def newey_west_mean_t(values: np.ndarray, max_lag: int = 20) -> float:
    """计算均值的 Newey-West t 统计量。

    日收益差存在序列相关与波动聚集，因此不能把普通 IID 标准误当作最终
    不确定性。这里使用 Bartlett 权重估计均值的长期方差。
    """

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    centered = x - x.mean()
    long_run_variance = float(np.dot(centered, centered) / len(x))
    for lag in range(1, min(max_lag, len(x) - 1) + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        covariance = float(
            np.dot(centered[lag:], centered[:-lag]) / len(x)
        )
        long_run_variance += 2.0 * weight * covariance
    standard_error = np.sqrt(max(long_run_variance, 0.0) / len(x))
    return float(x.mean() / standard_error) if standard_error > 0 else float("nan")


def moving_block_mean_interval(
    values: np.ndarray,
    *,
    block_length: int = 20,
    repetitions: int = 5_000,
    seed: int = 20261009,
) -> tuple[float, float]:
    """对日收益差做循环 moving-block bootstrap 均值区间。

    整块抽样保留块内时间依赖。区间乘以 252 后解释为算术年化增量，
    而不是复利收益之差。
    """

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < block_length:
        raise ValueError("样本长度必须大于 block_length")
    rng = np.random.default_rng(seed)
    starts = np.arange(len(x))
    offsets = np.arange(block_length)
    blocks_needed = int(np.ceil(len(x) / block_length))
    means = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        indices = (chosen[:, None] + offsets[None, :]) % len(x)
        sample = x[indices.ravel()[: len(x)]]
        means[repetition] = sample.mean() * TRADING_DAYS
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def paired_statistics(daily: pd.DataFrame) -> dict[str, dict[str, float]]:
    """计算预先指定模型对之间的配对日收益差。"""

    pivot = daily.pivot(index="date", columns="feature_group", values="portfolio_return")
    comparisons = {
        "cmi_core_minus_traditional": (
            "traditional_plus_cmi_core",
            "traditional",
        ),
        "rankic_minus_traditional": (
            "traditional_plus_rankic",
            "traditional",
        ),
        "mi_minus_traditional": ("traditional_plus_mi", "traditional"),
        "cmi_core_minus_rankic": (
            "traditional_plus_cmi_core",
            "traditional_plus_rankic",
        ),
    }
    output: dict[str, dict[str, float]] = {}
    for name, (left, right) in comparisons.items():
        difference = (pivot[left] - pivot[right]).dropna().to_numpy(dtype=float)
        lower, upper = moving_block_mean_interval(difference)
        output[name] = {
            "observations": int(len(difference)),
            "mean_daily_bps": float(difference.mean() * 10_000),
            "arithmetic_annual_difference": float(difference.mean() * TRADING_DAYS),
            "newey_west_t_lag20": newey_west_mean_t(difference, max_lag=20),
            "moving_block_95pct_annual_lower": lower,
            "moving_block_95pct_annual_upper": upper,
        }
    return output


def cumulative_panel(ax: plt.Axes, daily: pd.DataFrame) -> None:
    """绘制四个主对照组与内部等权基准的累计净值。"""

    selected = [
        "traditional",
        "traditional_plus_rankic",
        "traditional_plus_mi",
        "traditional_plus_cmi_core",
    ]
    pivot = daily.pivot(index="date", columns="feature_group", values="portfolio_return")
    nav = (1.0 + pivot[selected]).cumprod()
    for group in selected:
        width = 2.4 if group == "traditional_plus_cmi_core" else 1.4
        ax.plot(nav.index, nav[group], label=DISPLAY_NAMES[group], linewidth=width)
    benchmark = (
        daily.loc[daily["feature_group"].eq("traditional")]
        .set_index("date")["benchmark_return"]
        .sort_index()
    )
    ax.plot(
        benchmark.index,
        (1.0 + benchmark).cumprod(),
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Eligible-universe mean",
    )
    ax.set_title("OOS cumulative return (0 bps, execution constrained)")
    ax.set_ylabel("Growth of 1")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)


def overall_bars(ax: plt.Axes, scoreboard: pd.DataFrame) -> None:
    """绘制七组模型的年化收益点估计。"""

    ordered = scoreboard.sort_values("annual_return")
    labels = [DISPLAY_NAMES[item] for item in ordered["feature_group"]]
    values = ordered["annual_return"].to_numpy(dtype=float) * 100.0
    colors = [
        "#c44e52" if item == "traditional_plus_cmi_core" else "#4c72b0"
        for item in ordered["feature_group"]
    ]
    ax.barh(labels, values, color=colors, alpha=0.9)
    ax.set_title("Annualized return by feature set")
    ax.set_xlabel("Annualized return (%)")
    ax.grid(axis="x", alpha=0.2)


def yearly_bars(ax: plt.Axes, yearly: pd.DataFrame) -> None:
    """展示主模型在不同年份的稳定性。"""

    selected = [
        "traditional",
        "traditional_plus_rankic",
        "traditional_plus_mi",
        "traditional_plus_cmi_core",
    ]
    table = yearly.loc[yearly["feature_group"].isin(selected)].pivot(
        index="year", columns="feature_group", values="annual_return"
    )
    table = table[selected] * 100.0
    table.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title("Annualized return by calendar segment")
    ax.set_xlabel("")
    ax.set_ylabel("Annualized return (%)")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(
        [DISPLAY_NAMES[column] for column in selected],
        frameon=False,
        fontsize=8,
    )


def importance_bars(ax: plt.Axes, importance: pd.DataFrame) -> None:
    """绘制 CMI-core 模型的前十个平均 gain share。"""

    top = importance.nlargest(10, "mean_gain_share").sort_values("mean_gain_share")
    labels = top["feature"].str.replace("factor__", "", regex=False).str.replace(
        "gp__", "", regex=False
    )
    colors = ["#c44e52" if item in {"gp__C084", "gp__C182"} else "#55a868" for item in top["feature"]]
    ax.barh(labels, top["mean_gain_share"] * 100.0, color=colors)
    ax.set_title("Traditional + CMI core: mean gain share")
    ax.set_xlabel("Gain share (%)")
    ax.grid(axis="x", alpha=0.2)


def main() -> None:
    """校验冻结结果并生成最终报告所需的两个小型文件。"""

    daily = pd.read_csv(RESULT_DIR / "model_daily.csv", parse_dates=["date"])
    scoreboard = pd.read_csv(RESULT_DIR / "model_scoreboard.csv")
    yearly = pd.read_csv(RESULT_DIR / "model_yearly.csv")
    importance = pd.read_csv(RESULT_DIR / "cmi_core_feature_importance.csv")

    expected_groups = set(DISPLAY_NAMES)
    if set(scoreboard["feature_group"]) != expected_groups:
        raise ValueError("scoreboard 的七组特征集合与冻结协议不一致")
    counts = daily.groupby("feature_group")["date"].nunique()
    if not counts.eq(637).all():
        raise ValueError("每组模型必须恰好覆盖 637 个 OOS 交易日")
    if daily.duplicated(["feature_group", "date"]).any():
        raise ValueError("model_daily 存在重复的 (feature_group, date)")

    statistics = {
        "source": "V1.1 frozen data-contract repair sensitivity run",
        "transaction_cost_bps": 0.0,
        "paired_comparisons": paired_statistics(daily),
    }
    (RESULT_DIR / "derived_statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    cumulative_panel(axes[0, 0], daily)
    overall_bars(axes[0, 1], scoreboard)
    yearly_bars(axes[1, 0], yearly)
    importance_bars(axes[1, 1], importance)
    figure.suptitle(
        "Information-Theoretic Symbolic Alpha Mining — Frozen V1.1 evidence",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(FIGURE_DIR / "model_overview.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
