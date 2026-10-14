"""对 V1.1 的两个 CMI-core 因子做独立的 OOS 十分位检验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import derive_log_float_market_cap, preprocess_matrix
from src.symbolic.expression import (
    EvaluationContext,
    ExpressionFactory,
    evaluate_expression,
    make_default_registry,
    parse_canonical_expression,
)
from src.symbolic.large_scale import add_dimensionless_terminals
from src.symbolic.operators import make_operator_executors


CORE_IDS = ("C084", "C182")


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    """计算有限样本 Pearson 相关，常数或样本不足时返回 NaN。"""

    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return np.nan
    if np.std(left[valid]) <= 1e-12 or np.std(right[valid]) <= 1e-12:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def _rank_ic(factor: pd.Series, label: pd.Series) -> float:
    """计算单日 Spearman RankIC，只在因子和标签都可观测时评价。"""

    valid = factor.notna() & label.notna()
    if int(valid.sum()) < 30:
        return np.nan
    return float(factor.loc[valid].corr(label.loc[valid], method="spearman"))


def _newey_west_t(values: pd.Series, lag: int = 20) -> float:
    """计算均值的 Newey-West t 值，减轻日度收益自相关的影响。"""

    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(array) <= lag + 2:
        return np.nan
    centered = array - array.mean()
    long_run_variance = float(centered @ centered / len(array))
    for offset in range(1, lag + 1):
        covariance = float(centered[offset:] @ centered[:-offset] / len(array))
        weight = 1.0 - offset / (lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    if long_run_variance <= 0:
        return np.nan
    return float(array.mean() / np.sqrt(long_run_variance / len(array)))


def _performance(returns: pd.Series) -> dict[str, float]:
    """汇总一个每日等权调仓收益序列，不扣交易成本。"""

    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    wealth = np.cumprod(1.0 + values)
    volatility = float(np.std(values, ddof=1))
    period_return = float(wealth[-1] - 1.0)
    annual_return = float(wealth[-1] ** (252.0 / len(values)) - 1.0)
    return {
        "n_dates": float(len(values)),
        "mean_daily_return": float(np.mean(values)),
        "annualized_mean_return": float(np.mean(values) * 252.0),
        "period_return": period_return,
        "annual_return": annual_return,
        "sharpe": (
            float(np.mean(values) / volatility * np.sqrt(252.0))
            if volatility > 0
            else np.nan
        ),
        "win_rate": float(np.mean(values > 0)),
    }


def _assign_deciles(block: pd.DataFrame) -> pd.Series:
    """按标准化因子从低到高稳定分成 Q1--Q10。"""

    valid = block["normalized_factor"].notna()
    result = pd.Series(pd.NA, index=block.index, dtype="Int8")
    if int(valid.sum()) < 100:
        return result
    # 用 stock 作为相同因子值的确定性次序，不读取未来收益。
    ordered = block.loc[valid].sort_values(
        ["normalized_factor", "stock"], kind="mergesort"
    )
    position = np.arange(len(ordered), dtype=np.int64)
    decile = np.floor(position * 10 / len(ordered)).astype(np.int8) + 1
    result.loc[ordered.index] = decile
    return result


def load_and_evaluate_raw_factors(
    clean_panel_path: Path,
    industry_path: Path,
    representatives_path: Path,
    *,
    warmup_start: str,
    oos_end: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """读取必要历史并从 canonical 表达式重算两个原始因子。"""

    required_columns = [
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
        "label_1d",
    ]
    panel = pd.read_parquet(
        clean_panel_path,
        columns=required_columns,
        filters=[
            ("date", ">=", pd.Timestamp(warmup_start)),
            ("date", "<=", pd.Timestamp(oos_end)),
        ],
    )
    panel["date"] = pd.to_datetime(panel["date"])
    panel["stock"] = panel["stock"].astype(str)
    panel = add_dimensionless_terminals(panel)
    panel["month"] = panel["date"].dt.to_period("M").astype(str)

    industry = pd.read_csv(industry_path, parse_dates=["snapshot_date"])
    industry["month"] = industry["snapshot_date"].dt.to_period("M").astype(str)
    industry = industry[
        ["month", "stock", "industry_sector"]
    ].drop_duplicates(["month", "stock"], keep="last")
    panel = panel.merge(
        industry,
        on=["month", "stock"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    panel["log_float_market_cap"] = derive_log_float_market_cap(
        panel["close"], panel["volume"], panel["turnover_pct"]
    )
    panel = panel.sort_values(["date", "stock"]).reset_index(drop=True)

    representatives = pd.read_csv(representatives_path).set_index("candidate_id")
    missing = set(CORE_IDS).difference(representatives.index)
    if missing:
        raise ValueError(f"冻结候选表缺少核心因子: {sorted(missing)}")
    expressions = {
        factor: str(representatives.loc[factor, "canonical"])
        for factor in CORE_IDS
    }
    registry = make_default_registry()
    factory = ExpressionFactory(registry)
    executors = make_operator_executors()
    context = EvaluationContext(panel)
    for factor in CORE_IDS:
        expression = parse_canonical_expression(expressions[factor], factory)
        raw = evaluate_expression(expression, context, registry, executors)
        if not isinstance(raw, pd.Series):
            raise TypeError(f"{factor} 没有输出 panel Series")
        panel[f"{factor}__raw"] = pd.to_numeric(raw, errors="coerce")
    return panel, expressions


def preprocess_oos(
    panel: pd.DataFrame,
    *,
    oos_start: str,
    oos_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐日执行 MAD5→行业/市值中性化→Rank Gaussian。"""

    oos = panel.loc[
        panel["date"].between(oos_start, oos_end),
        [
            "date",
            "stock",
            "label_1d",
            "log_float_market_cap",
            "industry_sector",
            "C084__raw",
            "C182__raw",
        ],
    ].copy()
    oos = oos.sort_values(["date", "stock"]).reset_index(drop=True)
    raw_matrix = oos[["C084__raw", "C182__raw"]].to_numpy(dtype=float)
    neutral = np.full_like(raw_matrix, np.nan)
    zscore = np.full_like(raw_matrix, np.nan)
    rankgauss = np.full_like(raw_matrix, np.nan)
    audits: list[dict[str, object]] = []

    for date, positions in oos.groupby("date", sort=True).indices.items():
        position = np.asarray(positions, dtype=np.int64)
        block = oos.iloc[position]
        raw_day, neutral_day, zscore_day, rankgauss_day = preprocess_matrix(
            raw_matrix[position],
            block["log_float_market_cap"].to_numpy(dtype=float),
            industry=block["industry_sector"].to_numpy(),
            n_mad=5.0,
            min_obs=30,
        )
        neutral[position] = neutral_day
        zscore[position] = zscore_day
        rankgauss[position] = rankgauss_day
        size = block["log_float_market_cap"].to_numpy(dtype=float)
        industry = block["industry_sector"].astype("string")
        for column, factor in enumerate(CORE_IDS):
            neutral_values = neutral_day[:, column]
            normalized_values = rankgauss_day[:, column]
            industry_means = (
                pd.DataFrame(
                    {"factor": neutral_values, "industry": industry.to_numpy()}
                )
                .dropna()
                .groupby("industry", observed=True)["factor"]
                .mean()
                .abs()
            )
            finite_normalized = normalized_values[np.isfinite(normalized_values)]
            audits.append(
                {
                    "date": pd.Timestamp(date),
                    "factor": factor,
                    "raw_valid_count": int(np.isfinite(raw_day[:, column]).sum()),
                    "neutral_valid_count": int(np.isfinite(neutral_values).sum()),
                    "neutral_size_corr": _correlation(neutral_values, size),
                    "neutral_max_abs_industry_mean": (
                        float(industry_means.max())
                        if not industry_means.empty
                        else np.nan
                    ),
                    "rankgauss_mean": (
                        float(finite_normalized.mean())
                        if len(finite_normalized)
                        else np.nan
                    ),
                    "rankgauss_std": (
                        float(finite_normalized.std(ddof=0))
                        if len(finite_normalized)
                        else np.nan
                    ),
                    "rankgauss_size_corr": _correlation(normalized_values, size),
                }
            )

    for column, factor in enumerate(CORE_IDS):
        oos[f"{factor}__neutral"] = neutral[:, column].astype(np.float32)
        oos[f"{factor}__zscore"] = zscore[:, column].astype(np.float32)
        oos[f"{factor}__rankgauss"] = rankgauss[:, column].astype(np.float32)
    return oos, pd.DataFrame(audits)


def evaluate_deciles(
    oos: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """计算日度十分位收益、总体曲线、RankIC 和分年度多空表现。"""

    daily_deciles: list[pd.DataFrame] = []
    ic_rows: list[dict[str, object]] = []
    for factor in CORE_IDS:
        frame = oos[
            [
                "date",
                "stock",
                "label_1d",
                f"{factor}__raw",
                f"{factor}__neutral",
                f"{factor}__rankgauss",
            ]
        ].rename(
            columns={
                f"{factor}__raw": "raw_factor",
                f"{factor}__neutral": "neutralized_factor",
                f"{factor}__rankgauss": "normalized_factor",
            }
        )
        frame["decile"] = frame.groupby("date", group_keys=False).apply(
            _assign_deciles, include_groups=False
        )
        for date, block in frame.groupby("date", sort=True):
            ic_rows.append(
                {
                    "date": pd.Timestamp(date),
                    "factor": factor,
                    "raw_rank_ic": _rank_ic(block["raw_factor"], block["label_1d"]),
                    "neutral_rank_ic": _rank_ic(
                        block["neutralized_factor"], block["label_1d"]
                    ),
                    "normalized_rank_ic": _rank_ic(
                        block["normalized_factor"], block["label_1d"]
                    ),
                }
            )
        valid = frame.dropna(subset=["decile", "label_1d"]).copy()
        valid["decile"] = valid["decile"].astype(int)
        daily = (
            valid.groupby(["date", "decile"], as_index=False, observed=True)
            .agg(return_1d=("label_1d", "mean"), n_stocks=("stock", "size"))
        )
        universe = valid.groupby("date", observed=True)["label_1d"].mean()
        daily["universe_return"] = daily["date"].map(universe)
        daily["excess_return"] = daily["return_1d"] - daily["universe_return"]
        daily.insert(0, "factor", factor)
        daily_deciles.append(daily)

    daily_frame = pd.concat(daily_deciles, ignore_index=True)
    summary_rows: list[dict[str, object]] = []
    long_short_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    for factor, factor_daily in daily_frame.groupby("factor", sort=False):
        for decile, block in factor_daily.groupby("decile", sort=True):
            record = _performance(block["return_1d"])
            record.update(
                {
                    "factor": factor,
                    "decile": int(decile),
                    "mean_daily_excess_return": float(block["excess_return"].mean()),
                    "annualized_excess_return": float(
                        block["excess_return"].mean() * 252.0
                    ),
                    "mean_n_stocks": float(block["n_stocks"].mean()),
                }
            )
            summary_rows.append(record)

        pivot = factor_daily.pivot(
            index="date", columns="decile", values="return_1d"
        ).sort_index()
        spread = pivot[10] - pivot[1]
        record = _performance(spread)
        record.update(
            {
                "factor": factor,
                "spread": "Q10-Q1",
                "newey_west_t_lag20": _newey_west_t(spread, lag=20),
            }
        )
        long_short_rows.append(record)
        for year, year_spread in spread.groupby(spread.index.year):
            year_record = _performance(year_spread)
            year_record.update(
                {
                    "factor": factor,
                    "year": int(year),
                    "spread": "Q10-Q1",
                    "newey_west_t_lag20": _newey_west_t(year_spread, lag=20),
                }
            )
            yearly_rows.append(year_record)

    ic_daily = pd.DataFrame(ic_rows)
    ic_summary_rows: list[dict[str, object]] = []
    for factor, block in ic_daily.groupby("factor", sort=False):
        record: dict[str, object] = {"factor": factor}
        for column in ["raw_rank_ic", "neutral_rank_ic", "normalized_rank_ic"]:
            values = block[column].dropna()
            record[f"mean_{column}"] = float(values.mean())
            record[f"icir_{column}"] = float(values.mean() / values.std(ddof=1))
            record[f"positive_ratio_{column}"] = float(values.gt(0).mean())
        ic_summary_rows.append(record)
    return (
        daily_frame,
        pd.DataFrame(summary_rows),
        pd.DataFrame(long_short_rows),
        pd.DataFrame(yearly_rows),
        pd.DataFrame(ic_summary_rows),
    )


def save_figure(summary: pd.DataFrame, output_path: Path) -> None:
    """绘制两个核心因子的日均年化超额十分位曲线。"""

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, factor in zip(axes, CORE_IDS):
        block = summary.loc[summary["factor"].eq(factor)].sort_values("decile")
        values = block["annualized_excess_return"].to_numpy() * 100.0
        colors = ["#4472C4" if value >= 0 else "#C55A11" for value in values]
        axis.bar(block["decile"], values, color=colors, width=0.75)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"{factor}: neutralized Rank-Gaussian deciles")
        axis.set_xlabel("Decile (Q1 low, Q10 high)")
        axis.set_xticks(range(1, 11))
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Annualized mean excess return (%)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "lightgbm_final_v11.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "lightgbm_final_v11"
        / "cmi_core_deciles",
    )
    parser.add_argument("--warmup-start", default="2023-06-01")
    args = parser.parse_args()
    started = perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel, expressions = load_and_evaluate_raw_factors(
        PROJECT_ROOT / str(data["clean_panel"]),
        PROJECT_ROOT / str(data["industry_monthly_pit"]),
        PROJECT_ROOT / str(data["candidate_representatives"]),
        warmup_start=args.warmup_start,
        oos_end=str(data["oos_end"]),
    )
    oos, preprocessing_audit = preprocess_oos(
        panel,
        oos_start=str(data["oos_start"]),
        oos_end=str(data["oos_end"]),
    )
    del panel
    daily, summary, long_short, yearly, ic_summary = evaluate_deciles(oos)

    preprocessing_audit.to_csv(
        args.output_dir / "preprocessing_audit_daily.csv", index=False
    )
    daily.to_csv(args.output_dir / "daily_decile_returns.csv", index=False)
    summary.to_csv(args.output_dir / "decile_summary.csv", index=False)
    long_short.to_csv(args.output_dir / "long_short_summary.csv", index=False)
    yearly.to_csv(args.output_dir / "yearly_long_short.csv", index=False)
    ic_summary.to_csv(args.output_dir / "rank_ic_summary.csv", index=False)
    save_figure(summary, args.output_dir / "cmi_core_deciles.png")

    audit_summary = (
        preprocessing_audit.groupby("factor", as_index=False)
        .agg(
            dates=("date", "nunique"),
            minimum_neutral_valid_count=("neutral_valid_count", "min"),
            mean_abs_neutral_size_corr=("neutral_size_corr", lambda x: x.abs().mean()),
            maximum_abs_neutral_size_corr=("neutral_size_corr", lambda x: x.abs().max()),
            mean_neutral_max_abs_industry_mean=(
                "neutral_max_abs_industry_mean",
                "mean",
            ),
            maximum_neutral_abs_industry_mean=(
                "neutral_max_abs_industry_mean",
                "max",
            ),
            mean_rankgauss_mean=("rankgauss_mean", "mean"),
            mean_rankgauss_std=("rankgauss_std", "mean"),
            mean_abs_rankgauss_size_corr=(
                "rankgauss_size_corr",
                lambda x: x.abs().mean(),
            ),
        )
    )
    audit_summary.to_csv(
        args.output_dir / "preprocessing_audit_summary.csv", index=False
    )
    manifest = {
        "status": "complete",
        "purpose": "CMI core individual OOS decile diagnostic",
        "factors": list(CORE_IDS),
        "expressions": expressions,
        "preprocessing": "raw -> MAD5 -> industry/log-float-market-cap neutralization -> rank Gaussian",
        "decile_input": "normalized Rank-Gaussian factor",
        "label": "Open(t+2)/Open(t+1)-1",
        "oos_start": str(data["oos_start"]),
        "oos_end": str(data["oos_end"]),
        "future_execution_state_filter": False,
        "transaction_cost_bps": 0.0,
        "execution_constraints": "not applied; predictive decile diagnostic only",
        "elapsed_seconds": perf_counter() - started,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("PREPROCESSING AUDIT")
    print(audit_summary.to_string(index=False))
    print("\nDECILES")
    print(
        summary[
            [
                "factor",
                "decile",
                "mean_daily_return",
                "annualized_excess_return",
                "sharpe",
            ]
        ].to_string(index=False)
    )
    print("\nLONG-SHORT")
    print(long_short.to_string(index=False))
    print("\nRANK IC")
    print(ic_summary.to_string(index=False))
    print(f"\ncompleted in {manifest['elapsed_seconds']:.1f}s: {args.output_dir}")


if __name__ == "__main__":
    main()
