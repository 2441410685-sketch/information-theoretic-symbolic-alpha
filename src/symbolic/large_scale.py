"""大规模 GP 使用的 PIT 抽样、中性化与双通道 fitness。

本模块刻意只做三件事：冻结每日研究样本、把候选输出按统一顺序预处理，
以及建立纯 RankIC / 纯 KSG-MI evaluator。稳定性、换手率与 CMI 不进入搜索。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from time import perf_counter

import numpy as np
import pandas as pd

from src.preprocessing.cross_sectional import preprocess_matrix, rank_gaussianize_matrix

from .expression import (
    EvaluationContext,
    Expression,
    evaluate_expression,
    make_default_registry,
    structural_hash,
)
from .fitness import (
    MIFitnessOutcome,
    RankICFitnessOutcome,
    evaluate_pure_mi_fitness,
    evaluate_rank_ic_fitness,
)
from .operators import OutputPolicy, make_operator_executors


def add_dimensionless_terminals(frame: pd.DataFrame) -> pd.DataFrame:
    """从 PIT OHLCV 构造 V2 的无量纲基础终端。

    所有 rolling 都按股票、按时间只使用截至当日 ``t`` 的数据。价格位置与
    相对量能消除了股票价格和成交额绝对尺度，避免 GP 因数值量级而反复生成
    ``amount`` 主导的等价表达式。原始列仍被保留，仅用于兼容 V1 公式。
    """

    required = {
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
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"构造无量纲 terminal 缺少列: {sorted(missing)}")
    output = frame.sort_values(["stock", "date"]).copy()

    def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        left = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)
        values = np.divide(
            left,
            right,
            out=np.full(len(output), np.nan, dtype=float),
            where=np.isfinite(left) & np.isfinite(right) & (np.abs(right) > 1e-12),
        )
        return pd.Series(values, index=output.index, dtype=float)

    price_range = pd.to_numeric(output["high"], errors="coerce") - pd.to_numeric(
        output["low"], errors="coerce"
    )
    output["gap_open"] = safe_ratio(output["open"], output["preclose"]) - 1.0
    output["high_low_ratio"] = safe_ratio(output["high"], output["low"]) - 1.0
    output["close_position"] = safe_ratio(
        pd.to_numeric(output["close"], errors="coerce")
        - pd.to_numeric(output["low"], errors="coerce"),
        price_range,
    )
    output["body_to_range"] = safe_ratio(
        pd.to_numeric(output["close"], errors="coerce")
        - pd.to_numeric(output["open"], errors="coerce"),
        price_range,
    )

    grouped = output.groupby("stock", sort=False, observed=True)
    for column, windows in (("volume", (5, 20)), ("amount", (5, 20))):
        numeric = pd.to_numeric(output[column], errors="coerce")
        for window in windows:
            rolling_mean = grouped[column].transform(
                lambda values, w=window: pd.to_numeric(values, errors="coerce")
                .rolling(w, min_periods=max(3, w // 2))
                .mean()
            )
            output[f"{column}_rel{window}"] = safe_ratio(numeric, rolling_mean)
    turnover_mean20 = grouped["turnover_pct"].transform(
        lambda values: pd.to_numeric(values, errors="coerce")
        .rolling(20, min_periods=10)
        .mean()
    )
    output["turnover_rel20"] = safe_ratio(output["turnover_pct"], turnover_mean20)
    output["volatility20"] = grouped["return_1d"].transform(
        lambda values: pd.to_numeric(values, errors="coerce")
        .rolling(20, min_periods=10)
        .std(ddof=1)
    )
    derived = [
        "gap_open",
        "high_low_ratio",
        "close_position",
        "body_to_range",
        "volume_rel5",
        "volume_rel20",
        "amount_rel5",
        "amount_rel20",
        "turnover_rel20",
        "volatility20",
    ]
    output[derived] = output[derived].replace([np.inf, -np.inf], np.nan)
    return output.sort_values(["date", "stock"]).reset_index(drop=True)


def _stable_priority(date: pd.Timestamp, stock: str, seed: int) -> int:
    """为 ``(date, stock)`` 生成跨平台稳定的抽样优先级。"""

    payload = f"{pd.Timestamp(date).date()}|{stock}|{seed}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def deterministic_stratified_sample(
    frame: pd.DataFrame,
    *,
    sample_size: int = 500,
    size_buckets: int = 5,
    random_seed: int = 20260921,
) -> pd.Series:
    """逐日按行业×市值层比例抽取固定数量股票。

    抽样仅使用当日已知的行业、市值与股票代码，不查看未来收益。每层先按
    比例分配整数名额，余下名额按稳定哈希补齐，因此重复运行得到相同键。
    返回值与输入索引对齐；历史 warm-up 行可在调用后另行置为 ``False``。
    """

    required = {"date", "stock", "industry_sector", "log_float_market_cap"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"抽样数据缺少列: {sorted(missing)}")
    if sample_size < 1 or size_buckets < 1:
        raise ValueError("sample_size 与 size_buckets 必须为正整数")

    selected = pd.Series(False, index=frame.index, dtype=bool)
    for date, group in frame.groupby("date", sort=True, observed=True):
        eligible = group.loc[
            group["industry_sector"].notna()
            & np.isfinite(pd.to_numeric(group["log_float_market_cap"], errors="coerce"))
        ].copy()
        if len(eligible) < sample_size:
            raise ValueError(f"{pd.Timestamp(date).date()} 可抽样股票不足 {sample_size} 只")

        # 市值分桶用当日截面秩，避免 qcut 在大量相同值时丢桶。
        size_rank = eligible["log_float_market_cap"].rank(method="first", pct=True)
        bucket = np.minimum((size_rank * size_buckets).astype(int), size_buckets - 1)
        eligible["_stratum"] = (
            eligible["industry_sector"].astype(str) + "|" + bucket.astype(str)
        )
        eligible["_priority"] = [
            _stable_priority(date, str(stock), random_seed)
            for stock in eligible["stock"]
        ]

        counts = eligible.groupby("_stratum", observed=True).size()
        quotas = np.floor(sample_size * counts / len(eligible)).astype(int)
        chosen_indices: list[int] = []
        for stratum, quota in quotas.items():
            if quota <= 0:
                continue
            candidates = eligible.loc[eligible["_stratum"].eq(stratum)]
            chosen_indices.extend(
                candidates.nsmallest(int(quota), "_priority").index.tolist()
            )

        remainder = sample_size - len(chosen_indices)
        if remainder > 0:
            remaining = eligible.drop(index=chosen_indices)
            chosen_indices.extend(
                remaining.nsmallest(remainder, "_priority").index.tolist()
            )
        selected.loc[chosen_indices] = True
    return selected


def rank_gaussianize_label_by_date(
    frame: pd.DataFrame,
    *,
    label_column: str = "label_1d",
    minimum_cross_section: int = 50,
) -> pd.Series:
    """在每日完整可用截面上转换 label，供 pooled KSG 使用。"""

    if label_column not in frame:
        raise KeyError(f"缺少 label 列: {label_column}")
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, indices in frame.groupby("date", sort=True, observed=True).indices.items():
        position = np.asarray(indices, dtype=np.int64)
        transformed = rank_gaussianize_matrix(
            frame.iloc[position][label_column].to_numpy(dtype=float),
            min_obs=minimum_cross_section,
        )[:, 0]
        result.iloc[position] = transformed
    return result


def preprocess_candidate_by_date(
    raw_factor: pd.Series,
    frame: pd.DataFrame,
    *,
    mad_multiple: float = 5.0,
    minimum_cross_section: int = 50,
) -> pd.Series:
    """执行候选输出的正式顺序并返回日度 Rank-Gaussian 因子。

    顺序固定为：原始表达式 → MAD 截尾 → 行业/对数流通市值中性化 →
    Rank Gaussianization。处理在完整日截面完成，之后才使用冻结的 500 股键。
    """

    required = {"date", "industry_sector", "log_float_market_cap"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"中性化数据缺少列: {sorted(missing)}")
    if len(raw_factor) != len(frame) or not raw_factor.index.equals(frame.index):
        raise ValueError("raw_factor 必须与 frame 长度和索引完全一致")

    result = pd.Series(np.nan, index=frame.index, dtype=float)
    numeric = pd.to_numeric(raw_factor, errors="coerce")
    score_dates = None
    if "is_score_date" in frame:
        score_dates = set(frame.loc[frame["is_score_date"].astype(bool), "date"])
    for date, indices in frame.groupby("date", sort=True, observed=True).indices.items():
        # warm-up 行只服务于 rolling expression，本身不进入 fitness，没必要
        # 为它们重复做 240 次以上的横截面回归。
        if score_dates is not None and date not in score_dates:
            continue
        position = np.asarray(indices, dtype=np.int64)
        daily = frame.iloc[position]
        _, _, _, rank_gaussian = preprocess_matrix(
            numeric.iloc[position].to_numpy(dtype=float),
            daily["log_float_market_cap"].to_numpy(dtype=float),
            industry=daily["industry_sector"].to_numpy(),
            n_mad=mad_multiple,
            min_obs=minimum_cross_section,
        )
        result.iloc[position] = rank_gaussian[:, 0]
    return result


def candidate_random_state(expression: Expression, base_seed: int) -> int:
    """让同一表达式的 KSG tie jitter 与岛、进程和运行顺序无关。"""

    fingerprint = structural_hash(expression, make_default_registry())
    return int((base_seed + int(fingerprint[:8], 16)) % (2**32 - 1))


def build_semantic_projection(
    observation_count: int,
    *,
    probe_size: int = 1024,
    bit_count: int = 128,
    random_state: int = 20260921,
) -> tuple[np.ndarray, np.ndarray]:
    """建立固定的语义探针与随机超平面。

    语义探针只读取 selection 样本上的候选输出，不读取 label。每个候选都使用
    同一组位置和投影，因此 SimHash 的 Hamming 距离可近似比较两个输出向量的
    夹角；这比只比较表达式字符串更能识别“写法不同、信号相同”的候选。
    """

    if observation_count < 32:
        raise ValueError("semantic probe 至少需要 32 个观测")
    if probe_size < 32 or bit_count < 8 or bit_count % 8:
        raise ValueError("probe_size 至少为 32，bit_count 必须是 8 的正整数倍")
    size = min(int(probe_size), int(observation_count))
    positions = np.linspace(0, observation_count - 1, num=size, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    projection = rng.normal(size=(int(bit_count), size)).astype(np.float32)
    norms = np.linalg.norm(projection, axis=1, keepdims=True)
    projection = np.divide(
        projection,
        norms,
        out=np.zeros_like(projection),
        where=norms > 0,
    )
    return positions, projection


def semantic_simhash(
    values: pd.Series | np.ndarray,
    positions: np.ndarray,
    projection: np.ndarray,
) -> str:
    """把因子的标准化探针输出压缩为方向不敏感的语义代码。

    因子取反只改变交易方向，不产生新的信息，因此代码在 ``x`` 与 ``-x``
    之间取同一规范表示。缺失值在有限样本中心化后填零；若探针有效率不足
    70%，返回空代码并交由既有输出 gate 拒绝或降级处理。
    """

    array = np.asarray(values, dtype=float).reshape(-1)
    probe_positions = np.asarray(positions, dtype=np.int64)
    if probe_positions.ndim != 1 or len(probe_positions) != projection.shape[1]:
        raise ValueError("positions 与 projection 的探针维度不一致")
    if len(array) == 0 or probe_positions.min() < 0 or probe_positions.max() >= len(array):
        raise ValueError("semantic probe 位置超出候选输出范围")
    probe = array[probe_positions].astype(np.float32, copy=True)
    finite = np.isfinite(probe)
    if int(finite.sum()) < max(32, int(np.ceil(0.70 * len(probe)))):
        return ""
    mean = float(np.mean(probe[finite]))
    scale = float(np.std(probe[finite]))
    if not np.isfinite(scale) or scale <= 1e-12:
        return ""
    probe[finite] = (probe[finite] - mean) / scale
    probe[~finite] = 0.0
    bits = np.asarray(projection @ probe >= 0.0, dtype=np.uint8)
    packed = np.packbits(bits).tobytes()
    complement = bytes(value ^ 0xFF for value in packed)
    # x 与 -x 的 bit 串互为补码；取字典序较小者实现方向规范化。
    return min(packed, complement).hex()


def semantic_hamming_distance(first: str, second: str) -> int:
    """计算两个十六进制语义代码的 Hamming 距离；空代码不可比较。"""

    if not first or not second:
        raise ValueError("semantic code 不能为空")
    if len(first) != len(second):
        raise ValueError("semantic code 长度必须一致")
    return (int(first, 16) ^ int(second, 16)).bit_count()


def load_search_panel(path: str) -> pd.DataFrame:
    """加载已经冻结抽样键的 selection panel。"""

    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["stock"] = frame["stock"].astype("string")
    frame = frame.sort_values(["date", "stock"]).reset_index(drop=True)
    required = {
        "date",
        "stock",
        "industry_sector",
        "log_float_market_cap",
        "label_rankgauss",
        "is_fitness_sample",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"large-scale panel 缺少列: {sorted(missing)}")
    return frame


def make_neutralized_evaluator(
    frame: pd.DataFrame,
    *,
    channel: str,
    k: int = 5,
    mad_multiple: float = 5.0,
    minimum_cross_section: int = 50,
    minimum_dates: int = 180,
    output_policy: OutputPolicy | None = None,
    random_seed: int = 20260921,
    semantic_probe_size: int = 1024,
    semantic_projection_bits: int = 128,
) -> Callable[[Expression], RankICFitnessOutcome | MIFitnessOutcome]:
    """建立正式大搜索 evaluator；两个通道共享完全相同的预处理与样本。"""

    if channel not in {"rankic", "mi"}:
        raise ValueError("channel 必须是 'rankic' 或 'mi'")
    registry = make_default_registry()
    context = EvaluationContext(frame)
    executors = make_operator_executors()
    sample_mask = frame["is_fitness_sample"].to_numpy(dtype=bool)
    sample_frame = frame.loc[sample_mask].copy()
    semantic_positions, semantic_projection = build_semantic_projection(
        len(sample_frame),
        probe_size=semantic_probe_size,
        bit_count=semantic_projection_bits,
        random_state=random_seed + 97_531,
    )
    policy = output_policy or OutputPolicy(
        minimum_valid_fraction=0.70,
        minimum_valid_observations=72_000,
        minimum_unique_values=2,
    )

    def evaluator(expression: Expression) -> RankICFitnessOutcome | MIFitnessOutcome:
        started = perf_counter()
        raw = evaluate_expression(expression, context, registry, executors)
        if not isinstance(raw, pd.Series):
            raise TypeError("GP 候选必须输出 PANEL Series")
        processed = preprocess_candidate_by_date(
            raw,
            frame,
            mad_multiple=mad_multiple,
            minimum_cross_section=minimum_cross_section,
        )
        factor = processed.loc[sample_mask]
        semantic_code = semantic_simhash(
            factor.to_numpy(dtype=float),
            semantic_positions,
            semantic_projection,
        )
        if channel == "rankic":
            outcome, _ = evaluate_rank_ic_fitness(
                factor,
                sample_frame,
                label_column="label_rankgauss",
                minimum_cross_section=minimum_cross_section,
                minimum_dates=minimum_dates,
                output_policy=policy,
            )
        else:
            outcome = evaluate_pure_mi_fitness(
                factor,
                sample_frame,
                label_column="label_rankgauss",
                k=k,
                minimum_dates=minimum_dates,
                output_policy=policy,
                random_state=candidate_random_state(expression, random_seed),
            )
        return replace(
            outcome,
            evaluation_seconds=perf_counter() - started,
            semantic_code=semantic_code,
        )

    return evaluator
