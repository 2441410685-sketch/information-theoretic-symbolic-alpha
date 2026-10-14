"""可断点续跑的环形岛屿 GP 核心。

一个 channel 对应一个独立群岛。RankIC 与 MI 群岛之间绝不迁移，保证后续
比较回答的是 fitness 差异，而不是两个目标相互污染后的混合搜索。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .expression import (
    Expression,
    ExpressionFactory,
    Literal,
    Terminal,
    canonical_expression,
    structural_hash,
)
from .fitness import (
    CMIFitnessOutcome,
    MIFitnessOutcome,
    RankICFitnessOutcome,
)
from .gp import (
    FitnessEvaluator,
    Individual,
    InvalidOutcomeFactory,
    _audit_row,
    _evaluate_candidate,
    _summarize_generation,
    mutate_expression,
    random_panel_expression,
    subtree_crossover,
    tournament_select,
)
from .operators import ExpressionLimits, check_expression_constraints, weighted_complexity


FitnessOutcome = RankICFitnessOutcome | MIFitnessOutcome | CMIFitnessOutcome


@dataclass(frozen=True, slots=True)
class IslandGPConfig:
    """一个 channel 的群岛拓扑和岛内遗传参数。"""

    island_count: int = 8
    population_size: int = 64
    generations: int = 20
    migration_interval: int = 4
    elite_migrants: int = 1
    novelty_migrants: int = 1
    tournament_size: int = 4
    elite_size: int = 2
    reproduction_probability: float = 0.10
    crossover_probability: float = 0.45
    mutation_probability: float = 0.45
    initial_max_depth: int = 4
    mutation_max_depth: int = 3
    terminal_probability: float = 0.25
    windows: tuple[int, ...] = (2, 3, 5, 10, 20, 40, 60)
    base_random_seed: int = 20260921
    semantic_min_hamming: int = 0
    semantic_retry_limit: int = 0

    def __post_init__(self) -> None:
        if self.island_count < 2:
            raise ValueError("岛屿模型至少需要 2 个岛")
        if self.population_size < 8:
            raise ValueError("每岛 population_size 至少为 8")
        if self.generations < 1 or self.migration_interval < 1:
            raise ValueError("generations 与 migration_interval 必须为正")
        if self.generations % self.migration_interval:
            raise ValueError("generations 必须能被 migration_interval 整除")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size 超出合法范围")
        if not 1 <= self.elite_size < self.population_size:
            raise ValueError("elite_size 超出合法范围")
        migrant_count = self.elite_migrants + self.novelty_migrants
        if migrant_count < 1 or migrant_count >= self.population_size:
            raise ValueError("迁移个体数必须位于 [1, population_size)")
        probability_sum = (
            self.reproduction_probability
            + self.crossover_probability
            + self.mutation_probability
        )
        if not np.isclose(probability_sum, 1.0):
            raise ValueError("reproduction/crossover/mutation 概率之和必须为 1")
        if self.semantic_min_hamming < 0 or self.semantic_retry_limit < 0:
            raise ValueError("semantic_min_hamming 与 semantic_retry_limit 不能为负")


@dataclass(slots=True)
class IslandState:
    """一个岛可序列化的完整状态，用于迁移轮之间保存 checkpoint。"""

    island_id: int
    generation: int
    population: list[Individual]
    fitness_cache: dict[str, FitnessOutcome]
    expression_archive: dict[str, Expression]
    first_generation: dict[str, int]
    occurrence_rows: list[dict[str, object]]
    generation_rows: list[dict[str, object]]
    rng_state: dict[str, object]


def new_island_state(island_id: int, random_seed: int) -> IslandState:
    """建立尚未生成初代种群的空状态。"""

    rng = np.random.default_rng(random_seed)
    return IslandState(
        island_id=island_id,
        generation=0,
        population=[],
        fitness_cache={},
        expression_archive={},
        first_generation={},
        occurrence_rows=[],
        generation_rows=[],
        rng_state=rng.bit_generator.state,
    )


def _expression_tokens(expression: Expression) -> frozenset[str]:
    """提取结构 token，供 novelty migrant 使用。"""

    if isinstance(expression, Terminal):
        return frozenset({f"T:{expression.name}"})
    if isinstance(expression, Literal):
        return frozenset({f"L:{expression.value_kind.value}"})
    tokens = {f"O:{expression.operator}"}
    for argument in expression.arguments:
        tokens.update(_expression_tokens(argument))
    return frozenset(tokens)


def _jaccard_distance(first: frozenset[str], second: frozenset[str]) -> float:
    union = first | second
    return 0.0 if not union else 1.0 - len(first & second) / len(union)


def _semantic_code(individual: Individual) -> str:
    """读取 evaluator 生成的 selection-only SimHash；旧实验 outcome 可为空。"""

    return str(getattr(individual.outcome, "semantic_code", "") or "")


def semantic_hamming_distance(first: str, second: str) -> int:
    """返回语义 SimHash 的 Hamming 距离。"""

    if not first or not second or len(first) != len(second):
        raise ValueError("两个 semantic code 必须非空且长度一致")
    return (int(first, 16) ^ int(second, 16)).bit_count()


def _is_semantically_distinct(
    candidate: Individual,
    references: list[Individual] | tuple[Individual, ...],
    minimum_hamming: int,
) -> bool:
    """只把非常接近的输出视为重复；没有语义码时不误删候选。"""

    code = _semantic_code(candidate)
    if minimum_hamming <= 0 or not code:
        return True
    distances = [
        semantic_hamming_distance(code, other_code)
        for reference in references
        if (other_code := _semantic_code(reference)) and len(other_code) == len(code)
    ]
    return not distances or min(distances) > minimum_hamming


def _diverse_elites(
    population: list[Individual],
    count: int,
    minimum_hamming: int,
) -> list[Individual]:
    """在不改写 fitness 的前提下，优先保留语义不同的高质量精英。"""

    ranked = _valid_sorted(population)
    chosen: list[Individual] = []
    for item in ranked:
        if _is_semantically_distinct(item, chosen, minimum_hamming):
            chosen.append(item)
        if len(chosen) == count:
            return chosen
    # 极端情况下语义库太窄，用后续高 fitness 个体补齐，保证种群规模稳定。
    for item in ranked:
        if item not in chosen:
            chosen.append(item)
        if len(chosen) == count:
            break
    return chosen


def _valid_sorted(population: list[Individual]) -> list[Individual]:
    """按 fitness 降序、复杂度升序排列有效个体。"""

    return sorted(
        (item for item in population if item.outcome.accepted),
        key=lambda item: (item.outcome.fitness, -weighted_complexity(item.expression)),
        reverse=True,
    )


def choose_migrants(
    source: IslandState,
    target: IslandState,
    *,
    elite_count: int,
    novelty_count: int,
    registry,
) -> tuple[Individual, ...]:
    """选择精英和结构新颖个体；已在目标岛出现的结构优先跳过。"""

    ranked = _valid_sorted(source.population)
    if not ranked:
        return ()
    target_hashes = set(target.fitness_cache)
    chosen: list[Individual] = []
    for item in ranked:
        fingerprint = structural_hash(item.expression, registry)
        if fingerprint not in target_hashes:
            chosen.append(item)
        if len(chosen) == elite_count:
            break
    if not chosen:
        chosen.append(ranked[0])

    # novelty 不是随机低质量个体：先按 selection 输出的语义 Hamming 距离，
    # 无语义码时才退回 token Jaccard；fitness 始终只是质量排序，不加惩罚。
    candidate_pool = ranked[: max(2, len(ranked) // 2)]
    while len(chosen) < elite_count + novelty_count:
        chosen_hashes = {structural_hash(item.expression, registry) for item in chosen}
        # 新颖迁移者必须相对“已选迁移者 + 目标岛当前种群”都有差异，
        # 否则只是把另一个结构写法的同一信号搬进目标岛。
        references = [*chosen, *target.population]
        reference_tokens = [_expression_tokens(item.expression) for item in references]
        reference_codes = [code for item in references if (code := _semantic_code(item))]
        candidates = [
            item
            for item in candidate_pool
            if structural_hash(item.expression, registry) not in chosen_hashes
            and structural_hash(item.expression, registry) not in target_hashes
        ]
        if not candidates:
            break
        best = max(
            candidates,
            key=lambda item: (
                min(
                    (
                        semantic_hamming_distance(_semantic_code(item), code)
                        for code in reference_codes
                        if _semantic_code(item)
                        and len(_semantic_code(item)) == len(code)
                    ),
                    default=-1,
                ),
                min(
                    _jaccard_distance(_expression_tokens(item.expression), tokens)
                    for tokens in reference_tokens
                ),
                item.outcome.fitness,
            ),
        )
        chosen.append(best)
    return tuple(chosen)


def ring_migration_plan(
    states: list[IslandState],
    *,
    elite_count: int,
    novelty_count: int,
    registry,
) -> dict[int, tuple[Individual, ...]]:
    """构造 ``island i → island (i+1) mod n`` 的环形迁移计划。"""

    ordered = sorted(states, key=lambda state: state.island_id)
    plan: dict[int, tuple[Individual, ...]] = {}
    for position, source in enumerate(ordered):
        target = ordered[(position + 1) % len(ordered)]
        plan[target.island_id] = choose_migrants(
            source,
            target,
            elite_count=elite_count,
            novelty_count=novelty_count,
            registry=registry,
        )
    return plan


def _inject_migrants(
    state: IslandState,
    migrants: tuple[Individual, ...],
    registry,
) -> None:
    """以迁入个体替换当前最差个体，并复用同通道同样本上的 fitness。"""

    if not migrants or not state.population:
        return
    existing = {structural_hash(item.expression, registry) for item in state.population}
    incoming: list[Individual] = []
    for migrant in migrants:
        fingerprint = structural_hash(migrant.expression, registry)
        if fingerprint in existing:
            continue
        state.fitness_cache[fingerprint] = migrant.outcome
        incoming.append(
            Individual(
                migrant.expression,
                migrant.outcome,
                "migrant",
                (fingerprint,),
                True,
            )
        )
        existing.add(fingerprint)
    if not incoming:
        return
    ranked_worst_first = sorted(
        range(len(state.population)),
        key=lambda index: (
            state.population[index].outcome.accepted,
            state.population[index].outcome.fitness,
        ),
    )
    for index, migrant in zip(ranked_worst_first, incoming):
        state.population[index] = migrant


def advance_island(
    state: IslandState,
    factory: ExpressionFactory,
    evaluator: FitnessEvaluator,
    terminal_names: tuple[str, ...],
    operator_names: tuple[str, ...],
    *,
    config: IslandGPConfig,
    limits: ExpressionLimits,
    invalid_outcome_factory: InvalidOutcomeFactory,
    generations_to_advance: int,
    migrants: tuple[Individual, ...] = (),
) -> IslandState:
    """让一个岛前进若干代；昂贵 evaluator 只对未见结构执行。"""

    if generations_to_advance < 1:
        raise ValueError("generations_to_advance 必须为正")
    if state.generation + generations_to_advance > config.generations:
        raise ValueError("推进后会超过总 generations")
    rng = np.random.default_rng()
    rng.bit_generator.state = state.rng_state

    if not state.population:
        expressions = [
            random_panel_expression(
                rng,
                factory,
                terminal_names,
                operator_names,
                config.windows,
                maximum_depth=config.initial_max_depth,
                terminal_probability=config.terminal_probability,
            )
            for _ in range(config.population_size)
        ]
        state.population = [
            _evaluate_candidate(
                expression,
                "initial",
                (),
                evaluator,
                factory.registry,
                limits,
                state.fitness_cache,
                invalid_outcome_factory,
            )
            for expression in expressions
        ]
    _inject_migrants(state, migrants, factory.registry)

    stop_generation = state.generation + generations_to_advance
    while state.generation < stop_generation:
        generation = state.generation
        for position, individual in enumerate(state.population):
            row = _audit_row(individual, generation, position, factory.registry, limits)
            row["island_id"] = state.island_id
            state.occurrence_rows.append(row)
            fingerprint = str(row["structural_hash"])
            if row["structure_accepted"] and fingerprint not in state.expression_archive:
                state.expression_archive[fingerprint] = individual.expression
                state.first_generation[fingerprint] = generation
        summary = _summarize_generation(state.population, generation, factory.registry)
        summary["island_id"] = state.island_id
        state.generation_rows.append(summary)
        state.generation += 1
        if state.generation == config.generations:
            break

        valid = [item for item in state.population if item.outcome.accepted]
        if not valid:
            raise RuntimeError(f"island {state.island_id} 当前代没有有效个体")
        elites = _diverse_elites(
            valid,
            config.elite_size,
            config.semantic_min_hamming,
        )
        next_population = [
            Individual(
                elite.expression,
                elite.outcome,
                "elite",
                (structural_hash(elite.expression, factory.registry),),
                True,
            )
            for elite in elites
        ]
        semantic_retries = 0
        while len(next_population) < config.population_size:
            draw = rng.random()
            first = tournament_select(
                valid, rng, config.tournament_size, factory.registry
            )
            first_hash = structural_hash(first.expression, factory.registry)
            if draw < config.reproduction_probability:
                child = first.expression
                origin, parents = "reproduction", (first_hash,)
            elif draw < config.reproduction_probability + config.crossover_probability:
                second = tournament_select(valid, rng, config.tournament_size, factory.registry)
                # 配偶语义近乎相同时最多重抽数次，降低同一信号家族自交；
                # 这只改变探索路径，不改变 RankIC/MI fitness 定义。
                for _ in range(config.semantic_retry_limit):
                    if _is_semantically_distinct(
                        second,
                        [first],
                        config.semantic_min_hamming,
                    ):
                        break
                    second = tournament_select(
                        valid, rng, config.tournament_size, factory.registry
                    )
                second_hash = structural_hash(second.expression, factory.registry)
                child = subtree_crossover(
                    first.expression, second.expression, rng, factory.registry
                )
                origin, parents = "crossover", (first_hash, second_hash)
            else:
                child = mutate_expression(
                    first.expression,
                    rng,
                    factory,
                    terminal_names,
                    operator_names,
                    config.windows,
                    mutation_max_depth=config.mutation_max_depth,
                    terminal_probability=config.terminal_probability,
                )
                origin, parents = "mutation", (first_hash,)
            offspring = _evaluate_candidate(
                child,
                origin,
                parents,
                evaluator,
                factory.registry,
                limits,
                state.fitness_cache,
                invalid_outcome_factory,
            )
            if (
                offspring.outcome.accepted
                and not _is_semantically_distinct(
                    offspring,
                    next_population,
                    config.semantic_min_hamming,
                )
                and semantic_retries < config.semantic_retry_limit
            ):
                semantic_retries += 1
                continue
            next_population.append(offspring)
            semantic_retries = 0
        state.population = next_population
    state.rng_state = rng.bit_generator.state
    return state


def state_archive_frame(state: IslandState, registry, limits: ExpressionLimits) -> pd.DataFrame:
    """把一个岛的唯一结构档案转换为可合并表格。"""

    rows: list[dict[str, object]] = []
    for fingerprint, expression in state.expression_archive.items():
        outcome = state.fitness_cache[fingerprint]
        report = check_expression_constraints(expression, registry, limits)
        rows.append(
            {
                "island_id": state.island_id,
                "first_generation": state.first_generation[fingerprint],
                "canonical": canonical_expression(expression, registry),
                "structural_hash": fingerprint,
                "node_count": report.node_count,
                "depth": report.depth,
                "required_history": report.required_history,
                "weighted_complexity": report.weighted_complexity,
                **asdict(outcome),
            }
        )
    return pd.DataFrame(rows)
