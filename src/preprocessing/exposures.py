"""为逐日因子预处理构造市值与行业暴露。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.preprocessing.cross_sectional import derive_log_float_market_cap


def build_sorted_exposure_base(
    input_path: Path,
    industry_path: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float]]:
    """合并 PIT 月度行业与流通市值代理，并按 ``date-stock`` 排序。"""

    columns = ["date", "stock", "close", "volume", "turnover_pct", "label_1d"]
    base = pq.read_table(input_path, columns=columns).to_pandas()
    base["date"] = pd.to_datetime(base["date"])
    base["month"] = base["date"].dt.to_period("M").astype(str)
    base["_row_id"] = np.arange(len(base), dtype=np.int64)

    industry = pd.read_csv(industry_path, parse_dates=["snapshot_date"])
    industry["month"] = industry["snapshot_date"].dt.to_period("M").astype(str)
    lookup = industry[["month", "stock", "industry_sector", "industry_code"]]
    base = base.merge(
        lookup,
        on=["month", "stock"],
        how="left",
        sort=False,
        validate="many_to_one",
    ).sort_values("_row_id")
    base["log_float_market_cap"] = derive_log_float_market_cap(
        base["close"], base["volume"], base["turnover_pct"]
    )

    order = (
        base.sort_values(["date", "stock"], kind="mergesort")["_row_id"]
        .to_numpy(dtype=np.int64)
    )
    exposure = base.iloc[order][
        [
            "date",
            "stock",
            "label_1d",
            "log_float_market_cap",
            "industry_sector",
            "industry_code",
        ]
    ].reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    exposure.to_parquet(output_dir / "base.parquet", index=False, compression="zstd")
    coverage = {
        "rows": float(len(exposure)),
        "size_coverage": float(exposure["log_float_market_cap"].notna().mean()),
        "industry_coverage": float(exposure["industry_sector"].notna().mean()),
    }
    return exposure, order, coverage
