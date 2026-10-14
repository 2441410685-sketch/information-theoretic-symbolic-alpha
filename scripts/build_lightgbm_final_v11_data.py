"""下载全 A 股日线并构建 V1.1 PIT 信号 panel 与执行行情。

集群不能访问 BaoStock，因此本脚本必须先在本地运行。下载按股票分片保存，
中断后直接重跑即可续传；最终只生成训练 panel、执行行情和简洁数据摘要。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    CLEAN_COLUMNS,
    EXECUTION_COLUMNS,
    FIELDS,
    a_share_master,
    build_stock_frames,
    login_baostock,
    query_with_retry,
    stock_code,
)


def _stream_parquet(
    parts: list[Path],
    output: Path,
    *,
    columns: list[str],
    boolean_columns: set[str],
) -> int:
    """按股票分片流式合并，避免一次把十余年全市场数据装入内存。"""

    temporary = output.with_suffix(".tmp.parquet")
    temporary.unlink(missing_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for path in parts:
            frame = pd.read_parquet(path).reindex(columns=columns)
            if frame.empty:
                continue
            # 单股列若从未出现缺失值，pandas 可能推断为 int64；有上市前空行的
            # 股票则会推断为 float64。写入前统一类型，避免 Arrow schema 冲突。
            frame["date"] = pd.to_datetime(frame["date"])
            frame["stock"] = frame["stock"].astype(str)
            for column in columns:
                if column in {"date", "stock"}:
                    continue
                if column in boolean_columns:
                    frame[column] = frame[column].fillna(False).astype(bool)
                else:
                    frame[column] = pd.to_numeric(
                        frame[column], errors="coerce"
                    ).astype(np.float64)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            elif table.schema != writer.schema:
                table = table.cast(writer.schema)
            writer.write_table(table, row_group_size=100_000)
            rows += len(frame)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("没有可合并的数据分片")
    temporary.replace(output)
    return rows


def _summarize_clean_parts(
    clean_parts: list[Path], *, oos_start: str, oos_end: str
) -> dict[str, int | str]:
    """汇总最终面板规模与 OOS 执行状态，不建立独立审计体系。"""

    total_rows = 0
    oos_rows = 0
    oos_missing_label = 0
    oos_blocked_buy = 0
    oos_blocked_sell = 0
    minimum_date: pd.Timestamp | None = None
    maximum_date: pd.Timestamp | None = None
    for path in clean_parts:
        frame = pd.read_parquet(
            path, columns=["date", "label_1d", "can_buy_t1", "can_sell_t2"]
        )
        if frame.empty:
            continue
        dates = pd.to_datetime(frame["date"])
        total_rows += len(frame)
        minimum_date = dates.min() if minimum_date is None else min(minimum_date, dates.min())
        maximum_date = dates.max() if maximum_date is None else max(maximum_date, dates.max())
        mask = dates.between(oos_start, oos_end)
        block = frame.loc[mask]
        oos_rows += len(block)
        oos_missing_label += int(block["label_1d"].isna().sum())
        oos_blocked_buy += int((~block["can_buy_t1"].astype(bool)).sum())
        oos_blocked_sell += int((~block["can_sell_t2"].astype(bool)).sum())

    return {
        "clean_rows": total_rows,
        "start_date": str(minimum_date.date()) if minimum_date is not None else "",
        "end_date": str(maximum_date.date()) if maximum_date is not None else "",
        "oos_rows": oos_rows,
        "oos_missing_label_rows_retained": oos_missing_label,
        "oos_blocked_buy_rows_retained": oos_blocked_buy,
        "oos_blocked_sell_rows_retained": oos_blocked_sell,
    }


def main() -> None:
    """下载、断点续传并合并 V1.1 数据。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-start", default="2015-01-01")
    parser.add_argument("--signal-end", default="2026-08-21")
    parser.add_argument("--execution-start", default="2024-01-02")
    parser.add_argument("--oos-end", default="2026-08-19")
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "lightgbm_v11_parts",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="不联网，只合并已经下载完成的单股分片",
    )
    args = parser.parse_args()

    clean_dir = args.parts_dir / "clean"
    execution_dir = args.parts_dir / "execution"
    clean_dir.mkdir(parents=True, exist_ok=True)
    execution_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def finalize(master: pd.DataFrame) -> None:
        """合并完整分片并生成最终 parquet；下载完成后可独立重跑。"""

        clean_parts = sorted(clean_dir.glob("*.parquet"))
        execution_parts = sorted(execution_dir.glob("*.parquet"))
        if len(clean_parts) != len(master) or len(execution_parts) != len(master):
            raise ValueError(
                f"分片不完整: clean={len(clean_parts)}, "
                f"execution={len(execution_parts)}, master={len(master)}"
            )
        summary = _summarize_clean_parts(
            clean_parts, oos_start=args.execution_start, oos_end=args.oos_end
        )
        summary["stock_count"] = int(len(master))
        summary["execution_rows"] = _stream_parquet(
            execution_parts,
            args.output_dir / "execution_bars_v11.parquet",
            columns=EXECUTION_COLUMNS,
            boolean_columns={"has_valid_open", "can_buy_open", "can_sell_open"},
        )
        clean_rows = _stream_parquet(
            clean_parts,
            args.output_dir / "clean_panel_v11.parquet",
            columns=CLEAN_COLUMNS,
            boolean_columns={"can_buy_t1", "can_sell_t2"},
        )
        if clean_rows != int(summary["clean_rows"]):
            raise AssertionError("clean panel 合并行数与分片汇总不一致")
        (args.output_dir / "data_summary_v11.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if args.finalize_only:
        master_path = args.output_dir / "stock_master_v11.csv"
        if not master_path.exists():
            raise FileNotFoundError(master_path)
        finalize(pd.read_csv(master_path))
        return

    import baostock as bs

    login_baostock(bs)
    failures: list[str] = []
    started = time.time()
    try:
        master = a_share_master(
            query_with_retry(bs, lambda: bs.query_stock_basic(), "证券基础表", args.retries),
            args.download_start,
            args.signal_end,
        ).sort_values("code")
        calendar_raw = query_with_retry(
            bs,
            lambda: bs.query_trade_dates(
                start_date="1990-12-19", end_date=args.signal_end
            ),
            "交易日历",
            args.retries,
        )
        trading_calendar = pd.DatetimeIndex(
            pd.to_datetime(
                calendar_raw.loc[
                    calendar_raw["is_trading_day"].eq("1"), "calendar_date"
                ]
            )
        ).sort_values()
        master.to_csv(args.output_dir / "stock_master_v11.csv", index=False)
        calendar_raw.to_csv(args.output_dir / "trading_calendar_v11.csv", index=False)

        for number, item in enumerate(master.itertuples(index=False), start=1):
            stock = stock_code(item.code)
            stem = stock.replace(".", "_")
            clean_path = clean_dir / f"{stem}.parquet"
            execution_path = execution_dir / f"{stem}.parquet"
            if clean_path.exists() and execution_path.exists():
                continue
            try:
                raw = query_with_retry(
                    bs,
                    lambda code=item.code: bs.query_history_k_data_plus(
                        code,
                        FIELDS,
                        start_date=args.download_start,
                        end_date=args.signal_end,
                        frequency="d",
                        adjustflag="3",
                    ),
                    stock,
                    args.retries,
                )
                clean, execution = build_stock_frames(
                    raw,
                    trading_calendar,
                    stock=stock,
                    ipo_date=pd.Timestamp(item.ipoDate).date().isoformat(),
                    output_start=args.download_start,
                    signal_end=args.signal_end,
                    execution_start=args.execution_start,
                    fetch_end=args.signal_end,
                )
                clean.to_parquet(clean_path, index=False)
                execution.to_parquet(execution_path, index=False)
            except Exception as exc:
                failures.append(f"{stock}: {exc}")
            if number % 50 == 0 or number == len(master):
                elapsed = (time.time() - started) / 60.0
                print(
                    f"stocks {number}/{len(master)}, failures={len(failures)}, "
                    f"elapsed_min={elapsed:.1f}",
                    flush=True,
                )
    finally:
        bs.logout()

    if failures:
        (args.parts_dir / "download_failures.txt").write_text(
            "\n".join(failures), encoding="utf-8"
        )
        raise RuntimeError(f"仍有 {len(failures)} 只股票失败；直接重跑即可续传")

    finalize(master)


if __name__ == "__main__":
    main()
