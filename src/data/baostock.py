"""BaoStock 下载所需的最小公共函数。

本模块只负责把供应商结果转成表格、筛选 A 股证券并处理重试；
因子、标签和模型逻辑不放在下载层。
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd


FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "adjustflag,turn,tradestatus,pctChg,isST"
)


def result_to_frame(result: object) -> pd.DataFrame:
    """把 BaoStock 查询游标完整转为 ``DataFrame``。"""

    if result is None or str(result.error_code) != "0":  # type: ignore[attr-defined]
        raise RuntimeError("BaoStock 查询失败")
    rows: list[list[str]] = []
    while result.next():  # type: ignore[attr-defined]
        rows.append(result.get_row_data())  # type: ignore[attr-defined]
    return pd.DataFrame(rows, columns=result.fields)  # type: ignore[attr-defined]


def stock_code(code: str) -> str:
    """把供应商代码 ``sh.600000`` 统一为 ``600000.SH``。"""

    market, number = code.lower().split(".")
    return f"{number}.{market.upper()}"


def limit_rate(stock: str, dates: pd.DatetimeIndex) -> np.ndarray:
    """返回普通 A 股按板块和日期适用的涨跌停比例。"""

    number, market = stock.split(".")
    rates = np.full(len(dates), 0.10)
    if market == "BJ":
        rates[:] = 0.30
    elif number.startswith(("688", "689")):
        rates[:] = 0.20
    elif number.startswith(("300", "301")):
        rates[dates >= pd.Timestamp("2020-08-24")] = 0.20
    return rates


def a_share_master(
    raw: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """保留研究区间内实际存在过的沪、深、北 A 股。"""

    frame = raw.copy()
    frame["ipoDate"] = pd.to_datetime(frame["ipoDate"], errors="coerce")
    frame["outDate"] = pd.to_datetime(frame["outDate"], errors="coerce")
    code = frame["code"].astype(str).str.lower()
    is_a_share = (
        code.str.match(r"sh\.6\d{5}$")
        | code.str.match(r"sz\.(?:00|30)\d{4}$")
        | code.str.match(r"bj\.[489]\d{5}$")
    )
    overlaps = frame["ipoDate"].le(pd.Timestamp(end_date)) & (
        frame["outDate"].isna()
        | frame["outDate"].ge(pd.Timestamp(start_date))
    )
    return frame.loc[is_a_share & overlaps & frame["type"].eq("1")].copy()


def login_baostock(bs: object) -> None:
    """登录 BaoStock；失败时显式终止，避免写入不完整结果。"""

    login = bs.login()  # type: ignore[attr-defined]
    if str(login.error_code) != "0":
        raise RuntimeError("BaoStock 登录失败")


def query_with_retry(
    bs: object,
    query: Callable[[], object],
    name: str,
    retries: int = 3,
) -> pd.DataFrame:
    """连接中断时重新登录并有限次重试。"""

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return result_to_frame(query())
        except (RuntimeError, OSError, ConnectionError) as error:
            last_error = error
            if attempt == retries:
                break
            try:
                bs.logout()  # type: ignore[attr-defined]
            except Exception:
                pass
            time.sleep(2**attempt)
            login_baostock(bs)
    raise RuntimeError(f"{name} 连续 {retries} 次查询失败") from last_error
