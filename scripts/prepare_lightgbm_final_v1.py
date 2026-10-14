"""为 2024--2026 最终 OOS 构造传统因子与冻结 GP 因子面板。"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.factors import build_factor_catalog, generate_factor_panel
from src.preprocessing import build_sorted_exposure_base, preprocess_matrix
from src.symbolic.expression import (
    EvaluationContext,
    ExpressionFactory,
    evaluate_expression,
    make_default_registry,
    parse_canonical_expression,
)
from src.symbolic.large_scale import (
    add_dimensionless_terminals,
    preprocess_candidate_by_date,
)
from src.symbolic.operators import make_operator_executors


def load_final_config(path: Path) -> dict[str, object]:
    """读取已经冻结的最终评估配置。"""

    config = json.loads(path.read_text(encoding="utf-8"))
    data = config["data"]
    if pd.Timestamp(data["oos_start"]) < pd.Timestamp(
        data["locked_final_holdout_start"]
    ):
        raise ValueError("最终 OOS 不得早于 Locked Final Holdout")
    if pd.Timestamp(data["oos_end"]) != pd.Timestamp(data["final_data_end"]):
        raise ValueError("本轮最终 OOS 必须一直覆盖到最新可用标签日")
    return config


def filter_model_panel(
    input_csv: Path,
    output_parquet: Path,
    *,
    start_date: str,
    end_date: str,
    chunksize: int = 500_000,
) -> None:
    """流式截取模型所需历史，保留标签缺失行供滚动因子计算。"""

    if output_parquet.exists():
        return
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_parquet.with_suffix(".tmp.parquet")
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    try:
        if input_csv.suffix.lower() in {".parquet", ".pq"}:
            # V1.1 的修复面板使用 parquet，避免十余年全市场 CSV 体积膨胀；
            # 后续因子、预处理和模型代码完全复用 V1。
            source_chunks = [pd.read_parquet(input_csv)]
        else:
            source_chunks = pd.read_csv(input_csv, chunksize=chunksize)
        for chunk in source_chunks:
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
            selected = chunk.loc[chunk["date"].between(start, end)].copy()
            if selected.empty:
                continue
            table = pa.Table.from_pandas(selected, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary, table.schema, compression="zstd"
                )
            writer.write_table(table, row_group_size=100_000)
    finally:
        if writer is not None:
            writer.close()
    if not temporary.exists():
        raise ValueError("指定日期内没有模型数据")
    temporary.replace(output_parquet)


def _stock_batches(panel: pd.DataFrame, batch_stocks: int) -> list[tuple[int, int]]:
    """按完整股票分批，避免时间序列 rolling 被批次边界截断。"""

    codes = panel["stock"].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1, len(panel)]
    return [
        (
            int(boundaries[start]),
            int(boundaries[min(start + batch_stocks, len(boundaries) - 1)]),
        )
        for start in range(0, len(boundaries) - 1, batch_stocks)
    ]


def generate_traditional_factors(
    model_panel: Path,
    output_path: Path,
    catalog_path: Path,
    *,
    expected_count: int,
    batch_stocks: int = 25,
) -> None:
    """按股票批次重建传统 OHLCV 因子，所有 rolling 只回看历史。"""

    if output_path.exists():
        return
    panel = pd.read_parquet(model_panel).sort_values(["stock", "date"])
    catalog = build_factor_catalog()
    if len(catalog) != expected_count:
        raise ValueError(f"传统因子应为 {expected_count} 个，实际 {len(catalog)}")
    catalog.to_csv(catalog_path, index=False, encoding="utf-8-sig")

    temporary = output_path.with_suffix(".tmp.parquet")
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        batches = _stock_batches(panel, batch_stocks)
        for number, (start, end) in enumerate(batches, start=1):
            factor_panel = generate_factor_panel(panel.iloc[start:end])
            table = pa.Table.from_pandas(factor_panel, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary, table.schema, compression="zstd"
                )
            writer.write_table(table, row_group_size=100_000)
            print(f"traditional batch {number}/{len(batches)}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output_path)


def preprocess_traditional_rankgauss(
    input_path: Path,
    exposure: pd.DataFrame,
    order: np.ndarray,
    output_dir: Path,
    manifest_path: Path,
    *,
    expected_count: int,
    factor_batch: int = 10,
    min_obs: int = 30,
) -> None:
    """执行 MAD→中性化→Rank Gaussian，仅落盘模型需要的最终表示。

    Day 7 教学产物仍保存 raw/neutral/zscore/rankgauss 四层；最终大样本训练为
    节省约四分之三磁盘，只另存 rankgauss。统计处理顺序没有改变。
    """

    schema_names = pq.ParquetFile(input_path).schema_arrow.names
    factors = [name for name in schema_names if name.startswith("factor__")]
    if len(factors) != expected_count:
        raise ValueError(f"传统因子应为 {expected_count} 个，实际 {len(factors)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    date_array = exposure["date"].to_numpy()
    stock_array = exposure["stock"].to_numpy()
    size_array = exposure["log_float_market_cap"].to_numpy(dtype=np.float64)
    industry_array = exposure["industry_sector"].to_numpy()
    years = exposure["date"].dt.year.to_numpy()
    year_boundaries = np.r_[
        0, np.flatnonzero(years[1:] != years[:-1]) + 1, len(years)
    ]
    manifest_rows: list[dict[str, object]] = []

    for chunk_number, factor_start in enumerate(
        range(0, len(factors), factor_batch)
    ):
        factor_columns = factors[factor_start : factor_start + factor_batch]
        final_path = output_dir / f"factor_chunk_{chunk_number:03d}.parquet"
        for factor in factor_columns:
            manifest_rows.append(
                {
                    "factor": factor,
                    "chunk": chunk_number,
                    "file": final_path.name,
                    "representation": "MAD5_neutralized_rankgauss",
                }
            )
        if final_path.exists():
            print(f"traditional preprocess chunk {chunk_number + 1}: cached", flush=True)
            continue

        raw_frame = pq.read_table(input_path, columns=factor_columns).to_pandas()
        raw_sorted = raw_frame.to_numpy(dtype=np.float64, copy=False)[order]
        del raw_frame
        gc.collect()
        temporary = final_path.with_suffix(".tmp.parquet")
        temporary.unlink(missing_ok=True)
        writer: pq.ParquetWriter | None = None
        try:
            for year_start, year_end in zip(
                year_boundaries[:-1], year_boundaries[1:]
            ):
                raw_year = raw_sorted[year_start:year_end]
                dates_year = date_array[year_start:year_end]
                size_year = size_array[year_start:year_end]
                industry_year = industry_array[year_start:year_end]
                rankgauss_year = np.full_like(raw_year, np.nan)
                date_boundaries = np.r_[
                    0,
                    np.flatnonzero(dates_year[1:] != dates_year[:-1]) + 1,
                    len(dates_year),
                ]
                for date_start, date_end in zip(
                    date_boundaries[:-1], date_boundaries[1:]
                ):
                    _, _, _, rankgauss = preprocess_matrix(
                        raw_year[date_start:date_end],
                        size_year[date_start:date_end],
                        industry=industry_year[date_start:date_end],
                        min_obs=min_obs,
                    )
                    rankgauss_year[date_start:date_end] = rankgauss

                data: dict[str, object] = {
                    "date": dates_year,
                    "stock": stock_array[year_start:year_end],
                }
                for column, factor in enumerate(factor_columns):
                    data[f"{factor}__rankgauss"] = rankgauss_year[
                        :, column
                    ].astype(np.float32)
                table = pa.Table.from_pydict(data)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary, table.schema, compression="zstd"
                    )
                writer.write_table(table, row_group_size=100_000)
        finally:
            if writer is not None:
                writer.close()
        temporary.replace(final_path)
        print(
            f"traditional preprocess chunk {chunk_number + 1}/"
            f"{int(np.ceil(len(factors) / factor_batch))}",
            flush=True,
        )
        del raw_sorted
        gc.collect()

    pd.DataFrame(manifest_rows).to_csv(
        manifest_path, index=False, encoding="utf-8-sig"
    )


def generate_gp_factors(
    model_panel: Path,
    exposure_path: Path,
    representatives_path: Path,
    output_path: Path,
    *,
    expected_count: int,
) -> None:
    """在每日完整截面上重算并预处理冻结的 57 个 GP 表达式。"""

    if output_path.exists():
        return
    panel = pd.read_parquet(model_panel)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = add_dimensionless_terminals(panel)
    exposure = pd.read_parquet(exposure_path)[
        ["date", "stock", "log_float_market_cap", "industry_sector"]
    ]
    panel = panel.merge(
        exposure,
        on=["date", "stock"],
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values(["date", "stock"]).reset_index(drop=True)

    representatives = pd.read_csv(representatives_path).sort_values("candidate_id")
    if len(representatives) != expected_count:
        raise ValueError(
            f"冻结 GP 代表应为 {expected_count} 个，实际 {len(representatives)}"
        )
    registry = make_default_registry()
    factory = ExpressionFactory(registry)
    executors = make_operator_executors()
    context = EvaluationContext(panel)
    output: dict[str, object] = {
        "date": panel["date"].to_numpy(),
        "stock": panel["stock"].to_numpy(),
    }
    for number, row in enumerate(representatives.itertuples(index=False), start=1):
        expression = parse_canonical_expression(str(row.canonical), factory)
        raw = evaluate_expression(expression, context, registry, executors)
        if not isinstance(raw, pd.Series):
            raise TypeError("GP 表达式必须输出 Series")
        processed = preprocess_candidate_by_date(
            raw,
            panel,
            mad_multiple=5.0,
            minimum_cross_section=30,
        )
        output[str(row.candidate_id)] = processed.to_numpy(dtype=np.float32)
        print(
            f"gp factor {number}/{expected_count}: {row.candidate_id}", flush=True
        )
        del raw, processed
        gc.collect()
    pd.DataFrame(output).to_parquet(
        output_path, index=False, compression="zstd"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "lightgbm_final_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "lightgbm_final_v1" / "data",
    )
    args = parser.parse_args()
    started = perf_counter()
    config = load_final_config(args.config)
    data = config["data"]
    features = config["features"]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    model_panel = output / "clean_panel_2018_2026.parquet"
    traditional_panel = output / "traditional_raw_2018_2026.parquet"
    traditional_catalog = output / "traditional_catalog.csv"
    preprocessed = output / "traditional_rankgauss"
    gp_panel = output / "gp57_rankgauss_2018_2026.parquet"

    filter_model_panel(
        PROJECT_ROOT / str(data["clean_panel"]),
        model_panel,
        start_date=str(data["history_start"]),
        end_date=str(data["final_data_end"]),
    )
    generate_traditional_factors(
        model_panel,
        traditional_panel,
        traditional_catalog,
        expected_count=int(features["traditional_count"]),
    )
    exposure, order, coverage = build_sorted_exposure_base(
        traditional_panel,
        PROJECT_ROOT / str(data["industry_monthly_pit"]),
        preprocessed,
    )
    preprocess_traditional_rankgauss(
        traditional_panel,
        exposure,
        order,
        preprocessed,
        output / "traditional_manifest.csv",
        expected_count=int(features["traditional_count"]),
    )
    del exposure, order
    gc.collect()
    generate_gp_factors(
        model_panel,
        preprocessed / "base.parquet",
        PROJECT_ROOT / str(data["candidate_representatives"]),
        gp_panel,
        expected_count=int(features["union_gp_count"]),
    )

    base = pd.read_parquet(
        preprocessed / "base.parquet", columns=["date", "label_1d"]
    )
    valid = base["label_1d"].notna()
    manifest = {
        "status": "prepared",
        "research_version": config["research_version"],
        "history_start": str(data["history_start"]),
        "final_data_end": str(data["final_data_end"]),
        "prepared_min_date": str(pd.Timestamp(base["date"].min()).date()),
        "prepared_max_date": str(pd.Timestamp(base["date"].max()).date()),
        "valid_label_max_date": str(
            pd.Timestamp(base.loc[valid, "date"].max()).date()
        ),
        "rows": int(len(base)),
        "traditional_factors": int(features["traditional_count"]),
        "gp_factors": int(features["union_gp_count"]),
        "size_coverage": coverage["size_coverage"],
        "industry_coverage": coverage["industry_coverage"],
        "locked_final_holdout_opened": True,
        "holdout_open_reason": "one_time_final_evaluation",
        "elapsed_seconds": perf_counter() - started,
    }
    (output / "prepare_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
