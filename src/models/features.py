"""读取冻结因子矩阵并构造七组 LightGBM 对照特征。"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd


def _as_bool(series: pd.Series) -> pd.Series:
    """兼容 CSV 中的布尔值和 True/False 字符串。"""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_feature_matrices(
    data_dir: Path,
    representatives_path: Path,
    *,
    expected_traditional: int,
    expected_gp: int,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    list[str],
    np.ndarray,
    list[str],
    pd.DataFrame,
]:
    """读取共同 key、传统因子和冻结 GP 因子，并检查逐行对齐。"""

    base = pd.read_parquet(data_dir / "traditional_rankgauss" / "base.parquet")
    base["date"] = pd.to_datetime(base["date"])
    if base.duplicated(["date", "stock"]).any():
        raise ValueError("base 的 (date, stock) 必须唯一")

    manifest = pd.read_csv(data_dir / "traditional_manifest.csv")
    traditional_names = manifest["factor"].astype(str).tolist()
    if (
        len(traditional_names) != expected_traditional
        or len(set(traditional_names)) != expected_traditional
    ):
        raise ValueError("传统因子清单数量或唯一性错误")

    arrays: list[np.ndarray] = []
    loaded_names: list[str] = []
    for filename, subset in manifest.groupby("file", sort=True):
        chunk = pd.read_parquet(data_dir / "traditional_rankgauss" / filename)
        if not (
            np.array_equal(pd.to_datetime(chunk["date"]), base["date"])
            and np.array_equal(chunk["stock"].astype(str), base["stock"].astype(str))
        ):
            raise ValueError(f"传统因子分块 {filename} 与 base 未对齐")
        names = subset["factor"].astype(str).tolist()
        arrays.append(
            chunk[[f"{name}__rankgauss" for name in names]].to_numpy(
                dtype=np.float32, copy=True
            )
        )
        loaded_names.extend(names)
        del chunk
        gc.collect()
    traditional = np.column_stack(arrays)
    if loaded_names != traditional_names:
        raise ValueError("传统因子读取顺序与 manifest 不一致")
    np.nan_to_num(traditional, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    gp_frame = pd.read_parquet(data_dir / "gp57_rankgauss_2018_2026.parquet")
    if not (
        np.array_equal(pd.to_datetime(gp_frame["date"]), base["date"])
        and np.array_equal(gp_frame["stock"].astype(str), base["stock"].astype(str))
    ):
        raise ValueError("GP 因子与 base 未对齐")
    representatives = pd.read_csv(representatives_path).sort_values("candidate_id")
    gp_names = representatives["candidate_id"].astype(str).tolist()
    if len(gp_names) != expected_gp or len(set(gp_names)) != expected_gp:
        raise ValueError("GP 因子清单数量或唯一性错误")
    gp = gp_frame[gp_names].to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(gp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return base, traditional, traditional_names, gp, gp_names, representatives


def channel_ids(
    representatives: pd.DataFrame,
    *,
    expected_rankic: int,
    expected_mi: int,
) -> tuple[list[str], list[str]]:
    """从冻结代表表读取 RankIC 与 MI 两个来源通道。"""

    rankic_ids = representatives.loc[
        _as_bool(representatives["source_rankic"]), "candidate_id"
    ].astype(str).tolist()
    mi_ids = representatives.loc[
        _as_bool(representatives["source_mi"]), "candidate_id"
    ].astype(str).tolist()
    if len(rankic_ids) != expected_rankic or len(mi_ids) != expected_mi:
        raise ValueError(
            f"冻结通道计数错误: RankIC={len(rankic_ids)}, MI={len(mi_ids)}"
        )
    return rankic_ids, mi_ids


def build_group_matrix(
    group_name: str,
    traditional: np.ndarray,
    traditional_names: list[str],
    gp: np.ndarray,
    gp_names: list[str],
    *,
    rankic_ids: list[str],
    mi_ids: list[str],
    cmi_core: list[str],
) -> tuple[np.ndarray, list[str]]:
    """构造冻结的七组特征对照，不改变任何因子数值。"""

    gp_index = {name: number for number, name in enumerate(gp_names)}

    def select(names: list[str]) -> tuple[np.ndarray, list[str]]:
        unknown = sorted(set(names).difference(gp_index))
        if unknown:
            raise KeyError(f"未知 GP 因子: {unknown}")
        return (
            gp[:, [gp_index[name] for name in names]],
            [f"gp__{name}" for name in names],
        )

    if group_name == "traditional":
        return traditional, traditional_names
    if group_name == "union_only":
        return gp, [f"gp__{name}" for name in gp_names]
    if group_name == "mi_only":
        return select(mi_ids)

    additions = {
        "traditional_plus_rankic": rankic_ids,
        "traditional_plus_mi": mi_ids,
        "traditional_plus_union": gp_names,
        "traditional_plus_cmi_core": cmi_core,
    }
    if group_name not in additions:
        raise KeyError(f"未知 feature group: {group_name}")
    extra_matrix, extra_names = select(additions[group_name])
    return (
        np.column_stack([traditional, extra_matrix]),
        traditional_names + extra_names,
    )
