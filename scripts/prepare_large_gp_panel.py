"""为大规模 GP 准备 2022 selection 与 2023 validation 面板。

默认 ``--mode plan`` 只打印冻结配置，不读取 1.28GB clean panel。只有显式传入
``--mode prepare`` 才会生成数据，避免在未确认前误启动耗时任务。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.cross_sectional import derive_log_float_market_cap  # noqa: E402
from src.symbolic.large_scale import (  # noqa: E402
    add_dimensionless_terminals,
    deterministic_stratified_sample,
    rank_gaussianize_label_by_date,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "large_scale_gp.json"
DEFAULT_CLEAN_PANEL = PROJECT_ROOT / "data" / "processed" / "clean_panel.csv"
DEFAULT_INDUSTRY = PROJECT_ROOT / "data" / "processed" / "industry_monthly_pit.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "large_scale_gp" / "data"
MARKET_COLUMNS = [
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


def load_config(path: Path) -> dict[str, object]:
    """读取冻结 JSON 配置。"""

    return json.loads(path.read_text(encoding="utf-8"))


def exchange_dates(path: Path, year: int) -> pd.DatetimeIndex:
    """从 clean panel 提取某年的真实交易日，不使用自然日近似。"""

    values: set[str] = set()
    prefix = f"{year}-"
    for chunk in pd.read_csv(path, usecols=["date"], dtype="string", chunksize=500_000):
        dates = chunk["date"]
        values.update(dates.loc[dates.str.startswith(prefix)].dropna().tolist())
    return pd.DatetimeIndex(pd.to_datetime(sorted(values)))


def read_date_range(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """按块读取需要的日期范围，避免一次载入全部历史。"""

    pieces: list[pd.DataFrame] = []
    start_text, end_text = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    for chunk in pd.read_csv(
        path,
        usecols=MARKET_COLUMNS,
        dtype={"date": "string", "stock": "string"},
        chunksize=400_000,
    ):
        mask = chunk["date"].between(start_text, end_text)
        if mask.any():
            pieces.append(chunk.loc[mask].copy())
    if not pieces:
        raise RuntimeError(f"clean panel 在 {start_text} 至 {end_text} 没有数据")
    frame = pd.concat(pieces, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["date", "stock"]).reset_index(drop=True)


def attach_monthly_industry(frame: pd.DataFrame, industry_path: Path) -> pd.DataFrame:
    """按当月 PIT snapshot 合并行业，并推导对数流通市值代理。"""

    industry = pd.read_csv(
        industry_path,
        usecols=["snapshot_date", "stock", "industry_sector", "industry_code"],
        parse_dates=["snapshot_date"],
        dtype={"stock": "string", "industry_sector": "string", "industry_code": "string"},
    )
    industry["month"] = industry["snapshot_date"].dt.to_period("M").astype(str)
    lookup = industry[["month", "stock", "industry_sector", "industry_code"]]
    output = frame.copy()
    output["month"] = output["date"].dt.to_period("M").astype(str)
    output = output.merge(
        lookup,
        on=["month", "stock"],
        how="left",
        validate="many_to_one",
        sort=False,
    ).drop(columns="month")
    output["log_float_market_cap"] = derive_log_float_market_cap(
        output["close"], output["volume"], output["turnover_pct"]
    )
    return output.sort_values(["date", "stock"]).reset_index(drop=True)


def build_period_panel(
    clean_panel: Path,
    industry_path: Path,
    score_dates: pd.DatetimeIndex,
    *,
    sample_size: int,
    size_buckets: int,
    random_seed: int,
    minimum_cross_section: int,
) -> pd.DataFrame:
    """构造一个含 60 日以上 warm-up 的完整截面 period panel。"""

    history_start = score_dates.min() - pd.Timedelta(days=150)
    frame = read_date_range(clean_panel, history_start, score_dates.max())
    frame = attach_monthly_industry(frame, industry_path)
    # V2 语法直接读取无量纲终端；rolling 只按 stock 使用截至当日的数据。
    frame = add_dimensionless_terminals(frame)
    frame["is_score_date"] = frame["date"].isin(score_dates)
    frame["label_rankgauss"] = np.nan
    score_mask = frame["is_score_date"].to_numpy(dtype=bool)
    score_frame = frame.loc[score_mask].copy()
    frame.loc[score_mask, "label_rankgauss"] = rank_gaussianize_label_by_date(
        score_frame,
        minimum_cross_section=minimum_cross_section,
    ).to_numpy()

    eligible = frame.loc[
        score_mask
        & frame["label_rankgauss"].notna()
        & frame["industry_sector"].notna()
        & np.isfinite(frame["log_float_market_cap"])
    ]
    sampled = deterministic_stratified_sample(
        eligible,
        sample_size=sample_size,
        size_buckets=size_buckets,
        random_seed=random_seed,
    )
    frame["is_fitness_sample"] = False
    frame.loc[sampled.index[sampled], "is_fitness_sample"] = True
    counts = frame.loc[frame["is_fitness_sample"]].groupby("date").size()
    if len(counts) != len(score_dates) or not counts.eq(sample_size).all():
        raise RuntimeError("冻结 PIT 样本没有达到每个交易日固定股票数")
    return frame


def prepare(config: dict[str, object], clean_panel: Path, industry: Path, output: Path) -> None:
    """生成两个 parquet 和一份简洁元数据。"""

    split = config["research_split"]
    sampling = config["sampling"]
    preprocessing = config["preprocessing"]
    output.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, object]] = []
    periods = (
        ("selection", int(split["selection_year"]), int(split["selection_sessions"])),
        ("validation", int(split["validation_year"]), int(split["validation_sessions"])),
    )
    for number, (name, year, session_count) in enumerate(periods):
        dates = exchange_dates(clean_panel, year)
        if len(dates) < session_count:
            raise RuntimeError(f"{year} 只有 {len(dates)} 个交易日，少于 {session_count}")
        score_dates = dates[:session_count]
        panel = build_period_panel(
            clean_panel,
            industry,
            score_dates,
            sample_size=int(sampling["stocks_per_date"]),
            size_buckets=int(sampling["size_buckets"]),
            random_seed=int(sampling["random_seed"]) + number,
            minimum_cross_section=int(preprocessing["minimum_cross_section"]),
        )
        path = output / f"{name}_panel.parquet"
        panel.to_parquet(path, index=False, compression="zstd")
        sample = panel.loc[panel["is_fitness_sample"]]
        metadata.append(
            {
                "period": name,
                "year": year,
                "history_start": panel["date"].min().strftime("%Y-%m-%d"),
                "score_start": score_dates.min().strftime("%Y-%m-%d"),
                "score_end": score_dates.max().strftime("%Y-%m-%d"),
                "score_dates": len(score_dates),
                "sample_rows": len(sample),
                "stocks_per_date": int(sample.groupby("date").size().median()),
                "full_panel_rows": len(panel),
                "industry_coverage": float(panel["industry_sector"].notna().mean()),
                "size_coverage": float(panel["log_float_market_cap"].notna().mean()),
            }
        )
    pd.DataFrame(metadata).to_csv(output / "panel_metadata.csv", index=False)


def main() -> None:
    """默认只展示计划；必须显式 ``--mode prepare`` 才写面板。"""

    parser = argparse.ArgumentParser(description="Prepare large-scale GP panels")
    parser.add_argument("--mode", choices=("plan", "prepare"), default="plan")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--clean-panel", type=Path, default=DEFAULT_CLEAN_PANEL)
    parser.add_argument("--industry", type=Path, default=DEFAULT_INDUSTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    print(json.dumps(config, ensure_ascii=False, indent=2))
    if arguments.mode == "plan":
        print("PLAN ONLY: 未读取 clean panel，未生成搜索数据。")
        return
    prepare(config, arguments.clean_panel, arguments.industry, arguments.output)
    print(f"prepared: {arguments.output}")


if __name__ == "__main__":
    main()
