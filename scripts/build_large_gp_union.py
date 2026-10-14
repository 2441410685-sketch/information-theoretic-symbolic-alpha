"""合并 RankIC-GP 与 MI-GP 的 Top-50，形成最多 100 个候选的并集。

并集不是第三次 GP，也不在此阶段使用 CMI。结构重复时保留两个来源标记，
后续再在 2023 validation 上统一做稳定性、显著性与增量信息筛选。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "experiments" / "large_scale_gp"


def build_union(rankic_path: Path, mi_path: Path) -> pd.DataFrame:
    """按结构哈希取并集，同时保留来源与各通道排名。"""

    rankic = pd.read_csv(rankic_path)
    mi = pd.read_csv(mi_path)
    required = {"canonical", "structural_hash", "fitness"}
    for name, frame in (("rankic", rankic), ("mi", mi)):
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{name} shortlist 缺少列: {sorted(missing)}")

    rankic = rankic.drop_duplicates("structural_hash").reset_index(drop=True)
    mi = mi.drop_duplicates("structural_hash").reset_index(drop=True)
    rankic_rows = {
        str(row.structural_hash): row for row in rankic.itertuples(index=False)
    }
    mi_rows = {str(row.structural_hash): row for row in mi.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    for fingerprint in sorted(set(rankic_rows) | set(mi_rows)):
        rank_row = rankic_rows.get(fingerprint)
        mi_row = mi_rows.get(fingerprint)
        representative = rank_row if rank_row is not None else mi_row
        rows.append(
            {
                "candidate_id": f"C{len(rows) + 1:03d}",
                "structural_hash": fingerprint,
                "canonical": representative.canonical,
                "source_rankic": rank_row is not None,
                "source_mi": mi_row is not None,
                "rankic_search_rank": (
                    rankic.index[rankic["structural_hash"].astype(str).eq(fingerprint)][0] + 1
                    if rank_row is not None
                    else pd.NA
                ),
                "mi_search_rank": (
                    mi.index[mi["structural_hash"].astype(str).eq(fingerprint)][0] + 1
                    if mi_row is not None
                    else pd.NA
                ),
                "rankic_search_fitness": rank_row.fitness if rank_row is not None else pd.NA,
                "mi_search_fitness": mi_row.fitness if mi_row is not None else pd.NA,
            }
        )
    union = pd.DataFrame(rows)
    if len(union) > len(rankic) + len(mi):
        raise RuntimeError("并集行数不应超过两个候选集合行数之和")
    return union


def main() -> None:
    """默认只检查输入路径；显式 ``--mode build`` 才生成并集。"""

    parser = argparse.ArgumentParser(description="Build RankIC/MI candidate union")
    parser.add_argument("--mode", choices=("plan", "build"), default="plan")
    parser.add_argument(
        "--rankic",
        type=Path,
        default=DEFAULT_ROOT / "rankic" / "shortlist_top50.csv",
    )
    parser.add_argument(
        "--mi",
        type=Path,
        default=DEFAULT_ROOT / "mi" / "shortlist_top50.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "candidate_union_max100.csv",
    )
    arguments = parser.parse_args()
    if arguments.mode == "plan":
        print(f"rankic={arguments.rankic}")
        print(f"mi={arguments.mi}")
        print("PLAN ONLY: 未读取搜索结果，未生成并集。")
        return
    union = build_union(arguments.rankic, arguments.mi)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    union.to_csv(arguments.output, index=False)
    print(f"union candidates={len(union)}, output={arguments.output}")


if __name__ == "__main__":
    main()
