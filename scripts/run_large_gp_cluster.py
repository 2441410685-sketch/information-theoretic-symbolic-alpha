"""在一个 20 核节点上运行一个独立的 RankIC 或 MI 群岛。

默认 ``--mode plan`` 不加载 panel、不计算 fitness。正式运行必须显式指定
``--mode run --channel rankic`` 或 ``--channel mi``。两个 channel 应由外层
shell 并行启动，各使用 8 个 worker，彼此之间不交换个体。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# 每个岛是一个进程；若 BLAS 再开线程会严重超卖 20 核节点。
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.symbolic.expression import ExpressionFactory, make_default_registry  # noqa: E402
from src.symbolic.fitness import invalid_mi_outcome, invalid_rank_ic_outcome  # noqa: E402
from src.symbolic.island_gp import (  # noqa: E402
    IslandGPConfig,
    IslandState,
    advance_island,
    new_island_state,
    ring_migration_plan,
    state_archive_frame,
)
from src.symbolic.large_scale import load_search_panel, make_neutralized_evaluator  # noqa: E402
from src.symbolic.operators import ExpressionLimits, OutputPolicy  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "large_scale_gp.json"
DEFAULT_PANEL = (
    PROJECT_ROOT / "experiments" / "large_scale_gp" / "data" / "selection_panel.parquet"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "large_scale_gp"
DEFAULT_TERMINAL_NAMES = (
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "turnover_pct",
    "return_1d",
)
DEFAULT_OPERATOR_NAMES = (
    "add",
    "subtract",
    "multiply",
    "safe_div",
    "delay",
    "delta",
    "ts_mean",
    "ts_std",
    "ts_min",
    "ts_max",
    "ts_rank",
    "ts_corr",
    "ts_cov",
    "rank",
)


_WORKER_RESOURCES: dict[str, object] = {}


def load_config(path: Path) -> dict[str, object]:
    """读取正式搜索配置。"""

    return json.loads(path.read_text(encoding="utf-8"))


def make_island_config(config: dict[str, object]) -> IslandGPConfig:
    """把 JSON 搜索段转换为强类型配置。"""

    search = config["search"]
    return IslandGPConfig(
        island_count=int(search["islands_per_channel"]),
        population_size=int(search["population_per_island"]),
        generations=int(search["generations"]),
        migration_interval=int(search["migration_interval_generations"]),
        elite_migrants=int(search["elite_migrants"]),
        novelty_migrants=int(search["novelty_migrants"]),
        tournament_size=int(search["tournament_size"]),
        elite_size=int(search["elite_size"]),
        reproduction_probability=float(search["reproduction_probability"]),
        crossover_probability=float(search["crossover_probability"]),
        mutation_probability=float(search["mutation_probability"]),
        initial_max_depth=int(search["initial_max_depth"]),
        mutation_max_depth=int(search["mutation_max_depth"]),
        terminal_probability=float(search["terminal_probability"]),
        windows=tuple(int(value) for value in search["windows"]),
        base_random_seed=int(search["base_random_seed"]),
        semantic_min_hamming=int(search.get("semantic_min_hamming", 0)),
        semantic_retry_limit=int(search.get("semantic_retry_limit", 0)),
    )


def make_limits(config: dict[str, object]) -> ExpressionLimits:
    """建立仅作硬门槛的表达式约束，不把复杂度写进 fitness。"""

    hard = config["hard_constraints"]
    return ExpressionLimits(
        maximum_nodes=int(hard["maximum_nodes"]),
        maximum_depth=int(hard["maximum_depth"]),
        maximum_required_history=int(hard["maximum_required_history"]),
        maximum_weighted_complexity=float(hard["maximum_weighted_complexity"]),
    )


def _initialize_worker(panel_path: str, channel: str, config_path: str) -> None:
    """每个长驻 worker 只加载一次 panel，避免每个迁移轮重复读盘。"""

    config = load_config(Path(config_path))
    frame = load_search_panel(panel_path)
    hard = config["hard_constraints"]
    preprocessing = config["preprocessing"]
    search = config["search"]
    policy = OutputPolicy(
        minimum_valid_fraction=float(hard["minimum_valid_fraction"]),
        minimum_valid_observations=int(hard["minimum_valid_observations"]),
        minimum_unique_values=int(hard["minimum_unique_values"]),
    )
    evaluator = make_neutralized_evaluator(
        frame,
        channel=channel,
        k=int(search["ksg_k"]),
        mad_multiple=float(preprocessing["mad_multiple"]),
        minimum_cross_section=int(preprocessing["minimum_cross_section"]),
        minimum_dates=int(hard["minimum_valid_dates"]),
        output_policy=policy,
        random_seed=int(search["base_random_seed"]),
        semantic_probe_size=int(search.get("semantic_probe_size", 1024)),
        semantic_projection_bits=int(search.get("semantic_projection_bits", 128)),
    )
    registry = make_default_registry()
    _WORKER_RESOURCES.update(
        {
            "factory": ExpressionFactory(registry),
            "evaluator": evaluator,
            "config": make_island_config(config),
            "limits": make_limits(config),
            "invalid_factory": invalid_rank_ic_outcome if channel == "rankic" else invalid_mi_outcome,
            "terminal_names": tuple(config.get("grammar", {}).get("terminal_names", DEFAULT_TERMINAL_NAMES)),
            "operator_names": tuple(config.get("grammar", {}).get("operator_names", DEFAULT_OPERATOR_NAMES)),
        }
    )


def _advance_worker(
    state: IslandState,
    migrants: tuple,
    generation_count: int,
) -> IslandState:
    """进程池任务：推进一个岛到下一个迁移边界。"""

    return advance_island(
        state,
        _WORKER_RESOURCES["factory"],
        _WORKER_RESOURCES["evaluator"],
        _WORKER_RESOURCES["terminal_names"],
        _WORKER_RESOURCES["operator_names"],
        config=_WORKER_RESOURCES["config"],
        limits=_WORKER_RESOURCES["limits"],
        invalid_outcome_factory=_WORKER_RESOURCES["invalid_factory"],
        generations_to_advance=generation_count,
        migrants=migrants,
    )


def save_checkpoint(
    path: Path,
    *,
    states: list[IslandState],
    incoming: dict[int, tuple],
    completed_generations: int,
) -> None:
    """保存可信本项目内部 checkpoint；不要加载外部来源的 pickle。"""

    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(
            {
                "states": states,
                "incoming": incoming,
                "completed_generations": completed_generations,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    temporary.replace(path)


def latest_checkpoint(directory: Path) -> Path | None:
    """返回代数最大的 checkpoint。"""

    candidates = sorted(directory.glob("checkpoint_gen*.pkl"))
    return candidates[-1] if candidates else None


def save_results(
    states: list[IslandState],
    output: Path,
    channel: str,
    config: dict[str, object],
) -> None:
    """保存完整 archive 与 fitness 优先、语义去重的候选 shortlist。"""

    registry = make_default_registry()
    limits = make_limits(config)
    occurrences = pd.concat(
        [pd.DataFrame(state.occurrence_rows) for state in states], ignore_index=True
    )
    summaries = pd.concat(
        [pd.DataFrame(state.generation_rows) for state in states], ignore_index=True
    )
    archives = pd.concat(
        [state_archive_frame(state, registry, limits) for state in states],
        ignore_index=True,
    )
    archives = (
        archives.sort_values(
            ["fitness", "weighted_complexity", "first_generation"],
            ascending=[False, True, True],
        )
        .drop_duplicates("structural_hash", keep="first")
        .reset_index(drop=True)
    )
    channel_dir = output / channel
    channel_dir.mkdir(parents=True, exist_ok=True)
    occurrences.to_parquet(channel_dir / "all_occurrences.parquet", index=False)
    summaries.to_csv(channel_dir / "generation_summary.csv", index=False)
    archives.to_parquet(channel_dir / "all_unique_candidates.parquet", index=False)
    shortlist_count = int(config["shortlist"][f"{channel}_channel"])
    accepted = archives.loc[archives["accepted"].astype(bool)].copy()
    schedule = tuple(
        int(value)
        for value in config["search"].get("semantic_shortlist_hamming_schedule", [0])
    )
    shortlist, threshold_used = select_semantic_shortlist(
        accepted,
        shortlist_count,
        hamming_schedule=schedule,
    )
    if len(shortlist) != shortlist_count:
        raise RuntimeError(
            f"{channel} 只有 {len(shortlist)} 个可保留候选，少于要求的 {shortlist_count}"
        )
    shortlist.to_csv(channel_dir / f"shortlist_top{shortlist_count}.csv", index=False)
    manifest = {
        "channel": channel,
        "islands": int(config["search"]["islands_per_channel"]),
        "population_per_island": int(config["search"]["population_per_island"]),
        "generations": int(config["search"]["generations"]),
        "population_slots": len(occurrences),
        "unique_structures": len(archives),
        "unique_semantic_codes": int(accepted["semantic_code"].replace("", pd.NA).nunique()),
        "shortlist_count": len(shortlist),
        "semantic_shortlist_hamming_used": threshold_used,
        "fitness_soft_penalties": [],
        "cmi_inside_search": False,
        "final_holdout_opened": False,
    }
    (channel_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def select_semantic_shortlist(
    archives: pd.DataFrame,
    count: int,
    *,
    hamming_schedule: tuple[int, ...] = (4, 2, 0),
) -> tuple[pd.DataFrame, int]:
    """按 fitness 贪心保留语义不同候选，必要时逐级放宽近重复门槛。

    Hamming 门槛只决定 shortlist 的覆盖范围，不进入 fitness。128-bit
    SimHash 下，4 bit 大致对应极高的输出余弦相似度，因此首先排除近乎相同
    的信号；若不足目标数量才退到 2/0 bit，而结构去重始终保持。
    """

    if count < 1 or not hamming_schedule:
        raise ValueError("count 必须为正且 hamming_schedule 不能为空")
    ranked = archives.sort_values(
        ["fitness", "weighted_complexity", "first_generation"],
        ascending=[False, True, True],
    ).drop_duplicates("structural_hash", keep="first")
    for threshold in hamming_schedule:
        chosen_indices: list[int] = []
        chosen_codes: list[str] = []
        for index, row in ranked.iterrows():
            code = str(row.get("semantic_code", "") or "")
            if code and any(
                len(code) == len(other)
                and (int(code, 16) ^ int(other, 16)).bit_count() <= threshold
                for other in chosen_codes
            ):
                continue
            chosen_indices.append(index)
            if code:
                chosen_codes.append(code)
            if len(chosen_indices) == count:
                return ranked.loc[chosen_indices].reset_index(drop=True), threshold
    return ranked.head(count).reset_index(drop=True), int(hamming_schedule[-1])


def run_channel(
    config_path: Path,
    panel_path: Path,
    output: Path,
    channel: str,
    workers: int,
    resume: bool,
) -> None:
    """运行一个 channel 的 8 岛环形迁移搜索。"""

    config = load_config(config_path)
    island_config = make_island_config(config)
    channel_dir = output / channel
    checkpoint_dir = channel_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = latest_checkpoint(checkpoint_dir) if resume else None
    if checkpoint is None:
        channel_offset = 0 if channel == "rankic" else 1_000_000
        states = [
            new_island_state(
                island_id,
                island_config.base_random_seed + channel_offset + island_id * 10_007,
            )
            for island_id in range(island_config.island_count)
        ]
        incoming: dict[int, tuple] = {state.island_id: () for state in states}
        completed = 0
    else:
        with checkpoint.open("rb") as handle:
            payload = pickle.load(handle)
        states = payload["states"]
        incoming = payload["incoming"]
        completed = int(payload["completed_generations"])

    registry = make_default_registry()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(str(panel_path.resolve()), channel, str(config_path.resolve())),
    ) as executor:
        while completed < island_config.generations:
            step = min(
                island_config.migration_interval,
                island_config.generations - completed,
            )
            futures = [
                executor.submit(
                    _advance_worker,
                    state,
                    incoming.get(state.island_id, ()),
                    step,
                )
                for state in states
            ]
            states = [future.result() for future in futures]
            completed += step
            if completed < island_config.generations:
                incoming = ring_migration_plan(
                    states,
                    elite_count=island_config.elite_migrants,
                    novelty_count=island_config.novelty_migrants,
                    registry=registry,
                )
            else:
                incoming = {state.island_id: () for state in states}
            save_checkpoint(
                checkpoint_dir / f"checkpoint_gen{completed:03d}.pkl",
                states=states,
                incoming=incoming,
                completed_generations=completed,
            )
            print(f"{channel}: completed generation {completed}/{island_config.generations}")
    save_results(states, output, channel, config)


def main() -> None:
    """默认仅打印资源与搜索计划；正式执行需要明确 ``--mode run``。"""

    parser = argparse.ArgumentParser(description="Run one large-scale GP archipelago")
    parser.add_argument("--mode", choices=("plan", "run"), default="plan")
    parser.add_argument("--channel", choices=("rankic", "mi"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    print(json.dumps({"channel": arguments.channel, **config["cluster"]}, ensure_ascii=False, indent=2))
    if arguments.mode == "plan":
        print("PLAN ONLY: 未加载 panel，未运行任何 GP fitness。")
        return
    if not arguments.panel.exists():
        raise FileNotFoundError(f"缺少 selection panel: {arguments.panel}")
    run_channel(
        arguments.config,
        arguments.panel,
        arguments.output,
        arguments.channel,
        arguments.workers,
        arguments.resume,
    )


if __name__ == "__main__":
    main()
