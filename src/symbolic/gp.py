"""Day 22：可复现的 RankIC Genetic Programming 基线。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .expression import (
    Call,
    Expression,
    ExpressionFactory,
    ExpressionRegistry,
    Literal,
    Terminal,
    ValueKind,
    canonical_expression,
    expression_kind,
    replace_subtree,
    structural_hash,
)
from .fitness import (
    CMIFitnessOutcome,
    MIFitnessOutcome,
    RankICFitnessOutcome,
    invalid_cmi_outcome,
    invalid_mi_outcome,
    invalid_rank_ic_outcome,
)
from .operators import (
    ExpressionLimits,
    check_expression_constraints,
    weighted_complexity,
)


FitnessOutcome = RankICFitnessOutcome | MIFitnessOutcome | CMIFitnessOutcome
FitnessEvaluator = Callable[[Expression], FitnessOutcome]
InvalidOutcomeFactory = Callable[[str], FitnessOutcome]


@dataclass(frozen=True, slots=True)
class GPConfig:
    """固定一个 GP run 的搜索预算和随机机制。"""

    population_size: int = 24
    generations: int = 5
    tournament_size: int = 3
    elite_size: int = 2
    reproduction_probability: float = 0.10
    crossover_probability: float = 0.45
    mutation_probability: float = 0.45
    initial_max_depth: int = 3
    mutation_max_depth: int = 2
    terminal_probability: float = 0.30
    windows: tuple[int, ...] = (2, 3, 5, 10, 20)
    random_seed: int = 20260917

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size 至少为 4")
        if self.generations < 1:
            raise ValueError("generations 至少为 1")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size 必须位于 [2, population_size]")
        if not 1 <= self.elite_size < self.population_size:
            raise ValueError("elite_size 必须位于 [1, population_size)")
        probability_sum = (
            self.reproduction_probability
            + self.crossover_probability
            + self.mutation_probability
        )
        if not np.isclose(probability_sum, 1.0):
            raise ValueError("reproduction/crossover/mutation 概率之和必须为 1")
        if min(self.initial_max_depth, self.mutation_max_depth) < 1:
            raise ValueError("生成深度必须为正")
        if not 0 <= self.terminal_probability <= 1:
            raise ValueError("terminal_probability 必须位于 [0,1]")
        if not self.windows or any(window < 1 for window in self.windows):
            raise ValueError("windows 必须包含正整数")


@dataclass(frozen=True, slots=True)
class Individual:
    """一个带训练期 fitness 的 GP 个体。"""

    expression: Expression
    outcome: FitnessOutcome
    origin: str
    parent_hashes: tuple[str, ...]
    cached_fitness: bool


@dataclass(slots=True)
class GPSearchResult:
    """一个 seed 的赢家、完整 occurrence audit 和 unique archive。"""

    best_expression: Expression
    best_outcome: FitnessOutcome
    occurrence_audit: pd.DataFrame
    unique_candidates: pd.DataFrame
    generation_summary: pd.DataFrame


def _walk_paths(
    expression: Expression,
    path: tuple[int, ...] = (),
) -> list[tuple[tuple[int, ...], Expression]]:
    rows = [(path, expression)]
    if isinstance(expression, Call):
        for index, argument in enumerate(expression.arguments):
            rows.extend(_walk_paths(argument, path + (index,)))
    return rows


def random_panel_expression(
    rng: np.random.Generator,
    factory: ExpressionFactory,
    terminal_names: tuple[str, ...],
    operator_names: tuple[str, ...],
    windows: tuple[int, ...],
    *,
    maximum_depth: int,
    terminal_probability: float,
) -> Expression:
    """按 registry 类型签名递归生成一个 PANEL 表达式。"""

    if maximum_depth <= 1 or rng.random() < terminal_probability:
        return factory.terminal(str(rng.choice(terminal_names)))
    operator = factory.registry.operators[str(rng.choice(operator_names))]
    arguments: list[Expression] = []
    for kind in operator.input_kinds:
        if kind is ValueKind.PANEL:
            arguments.append(
                random_panel_expression(
                    rng,
                    factory,
                    terminal_names,
                    operator_names,
                    windows,
                    maximum_depth=maximum_depth - 1,
                    terminal_probability=terminal_probability,
                )
            )
        elif kind is ValueKind.WINDOW:
            arguments.append(factory.window(int(rng.choice(windows))))
        elif kind is ValueKind.SCALAR:
            arguments.append(factory.scalar(float(rng.normal())))
        else:
            raise TypeError(f"不支持的 ValueKind: {kind}")
    return factory.call(operator.name, *arguments)


def subtree_crossover(
    first: Expression,
    second: Expression,
    rng: np.random.Generator,
    registry: ExpressionRegistry,
) -> Expression:
    """从两个 parent 中选择输出类型相同的子树并替换。"""

    first_nodes = _walk_paths(first)
    second_nodes = _walk_paths(second)
    second_kinds = {
        expression_kind(node, registry)
        for _, node in second_nodes
    }
    # 两棵 PANEL 表达式的根节点一定兼容；先过滤可配对
    # 类型，避免抽到 WINDOW 子树而另一 parent 无窗口节点。
    compatible_first = [
        item
        for item in first_nodes
        if expression_kind(item[1], registry) in second_kinds
    ]
    first_path, first_subtree = compatible_first[
        int(rng.integers(len(compatible_first)))
    ]
    target_kind = expression_kind(first_subtree, registry)
    compatible = [
        node
        for node in second_nodes
        if expression_kind(node[1], registry) is target_kind
    ]
    _, replacement_tree = compatible[int(rng.integers(len(compatible)))]
    return replace_subtree(first, first_path, replacement_tree, registry)


def mutate_expression(
    expression: Expression,
    rng: np.random.Generator,
    factory: ExpressionFactory,
    terminal_names: tuple[str, ...],
    operator_names: tuple[str, ...],
    windows: tuple[int, ...],
    *,
    mutation_max_depth: int,
    terminal_probability: float,
) -> Expression:
    """选择随机子树，并用相同输出类型的新随机子树替换。"""

    nodes = _walk_paths(expression)
    path, old_subtree = nodes[int(rng.integers(len(nodes)))]
    kind = expression_kind(old_subtree, factory.registry)
    if kind is ValueKind.PANEL:
        replacement_tree = random_panel_expression(
            rng,
            factory,
            terminal_names,
            operator_names,
            windows,
            maximum_depth=mutation_max_depth,
            terminal_probability=terminal_probability,
        )
    elif kind is ValueKind.WINDOW:
        replacement_tree = factory.window(int(rng.choice(windows)))
    elif kind is ValueKind.SCALAR:
        replacement_tree = factory.scalar(float(rng.normal()))
    else:
        raise TypeError(f"不支持的 ValueKind: {kind}")
    return replace_subtree(expression, path, replacement_tree, factory.registry)


def tournament_select(
    population: list[Individual],
    rng: np.random.Generator,
    tournament_size: int,
    registry: ExpressionRegistry,
) -> Individual:
    """有放回抽取 tournament，并按 fitness、复杂度、canonical 决胜。"""

    indices = rng.integers(0, len(population), size=tournament_size)
    contestants = [population[int(index)] for index in indices]
    return max(
        contestants,
        key=lambda individual: (
            individual.outcome.fitness,
            -weighted_complexity(individual.expression),
            canonical_expression(individual.expression, registry),
        ),
    )


def _evaluate_candidate(
    expression: Expression,
    origin: str,
    parent_hashes: tuple[str, ...],
    evaluator: FitnessEvaluator,
    registry: ExpressionRegistry,
    limits: ExpressionLimits,
    fitness_cache: dict[str, FitnessOutcome],
    invalid_outcome_factory: InvalidOutcomeFactory,
) -> Individual:
    report = check_expression_constraints(expression, registry, limits)
    if not report.accepted:
        outcome = invalid_outcome_factory(
            "structure:" + ",".join(report.reasons)
        )
        return Individual(expression, outcome, origin, parent_hashes, False)
    fingerprint = report.structural_hash
    cached = fingerprint in fitness_cache
    if cached:
        outcome = fitness_cache[fingerprint]
    else:
        outcome = evaluator(expression)
        fitness_cache[fingerprint] = outcome
    return Individual(expression, outcome, origin, parent_hashes, cached)


def _audit_row(
    individual: Individual,
    generation: int,
    position: int,
    registry: ExpressionRegistry,
    limits: ExpressionLimits,
) -> dict[str, object]:
    report = check_expression_constraints(individual.expression, registry, limits)
    return {
        "generation": generation,
        "position": position,
        "origin": individual.origin,
        "parent_hashes": ",".join(individual.parent_hashes),
        "canonical": canonical_expression(individual.expression, registry),
        "structural_hash": structural_hash(individual.expression, registry),
        "cached_fitness": individual.cached_fitness,
        "structure_accepted": report.accepted,
        "structure_reasons": ",".join(report.reasons),
        "node_count": report.node_count,
        "depth": report.depth,
        "required_history": report.required_history,
        "weighted_complexity": report.weighted_complexity,
        **asdict(individual.outcome),
    }


def _summarize_generation(
    population: list[Individual],
    generation: int,
    registry: ExpressionRegistry,
) -> dict[str, object]:
    accepted = [individual for individual in population if individual.outcome.accepted]
    semantic_codes = {
        str(getattr(item.outcome, "semantic_code", ""))
        for item in accepted
        if str(getattr(item.outcome, "semantic_code", ""))
    }
    if not accepted:
        return {
            "generation": generation,
            "population_size": len(population),
            "valid_count": 0,
            "distinct_count": len(
                {structural_hash(item.expression, registry) for item in population}
            ),
            "distinct_semantic_count": 0,
            "best_fitness": np.nan,
            "median_fitness": np.nan,
            "mean_fitness": np.nan,
            "best_canonical": "",
            "mean_weighted_complexity": np.nan,
        }
    best = max(
        accepted,
        key=lambda item: (
            item.outcome.fitness,
            -weighted_complexity(item.expression),
        ),
    )
    fitness = np.array([item.outcome.fitness for item in accepted], dtype=float)
    return {
        "generation": generation,
        "population_size": len(population),
        "valid_count": len(accepted),
        "distinct_count": len(
            {structural_hash(item.expression, registry) for item in population}
        ),
        "distinct_semantic_count": len(semantic_codes),
        "best_fitness": float(np.max(fitness)),
        "median_fitness": float(np.median(fitness)),
        "mean_fitness": float(np.mean(fitness)),
        "best_canonical": canonical_expression(best.expression, registry),
        "mean_weighted_complexity": float(
            np.mean([weighted_complexity(item.expression) for item in accepted])
        ),
    }


def _run_gp_search(
    factory: ExpressionFactory,
    evaluator: FitnessEvaluator,
    terminal_names: tuple[str, ...],
    operator_names: tuple[str, ...],
    *,
    config: GPConfig = GPConfig(),
    limits: ExpressionLimits = ExpressionLimits(),
    invalid_outcome_factory: InvalidOutcomeFactory,
) -> GPSearchResult:
    """运行一个 seed 的通用 GP，并完整保存 occurrence 与唯一候选。"""

    missing_terminals = set(terminal_names).difference(factory.registry.terminals)
    missing_operators = set(operator_names).difference(factory.registry.operators)
    if missing_terminals or missing_operators:
        raise KeyError(
            f"registry 缺少 terminals={sorted(missing_terminals)}, "
            f"operators={sorted(missing_operators)}"
        )
    rng = np.random.default_rng(config.random_seed)
    fitness_cache: dict[str, FitnessOutcome] = {}
    expression_archive: dict[str, Expression] = {}
    first_generation: dict[str, int] = {}
    occurrence_rows: list[dict[str, object]] = []
    generation_rows: list[dict[str, object]] = []

    initial = [
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
    population = [
        _evaluate_candidate(
            expression,
            "initial",
            (),
            evaluator,
            factory.registry,
            limits,
            fitness_cache,
            invalid_outcome_factory,
        )
        for expression in initial
    ]

    for generation in range(config.generations):
        for position, individual in enumerate(population):
            row = _audit_row(
                individual,
                generation,
                position,
                factory.registry,
                limits,
            )
            occurrence_rows.append(row)
            fingerprint = row["structural_hash"]
            # archive 保留所有结构合法且确实执行过 fitness gate 的
            # 唯一候选，包括因输出退化而被拒绝的表达式；这能防止
            # 最终只保留“成功候选”导致的选择性汇报。
            if row["structure_accepted"] and fingerprint not in expression_archive:
                expression_archive[str(fingerprint)] = individual.expression
                first_generation[str(fingerprint)] = generation
        generation_rows.append(
            _summarize_generation(population, generation, factory.registry)
        )
        if generation == config.generations - 1:
            break
        valid_population = [item for item in population if item.outcome.accepted]
        if not valid_population:
            raise RuntimeError("当前 generation 没有任何有效个体")
        elites = sorted(
            valid_population,
            key=lambda item: (
                item.outcome.fitness,
                -weighted_complexity(item.expression),
            ),
            reverse=True,
        )[: config.elite_size]
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
        while len(next_population) < config.population_size:
            draw = rng.random()
            first_parent = tournament_select(
                valid_population, rng, config.tournament_size, factory.registry
            )
            first_hash = structural_hash(first_parent.expression, factory.registry)
            if draw < config.reproduction_probability:
                child, origin, parents = (
                    first_parent.expression,
                    "reproduction",
                    (first_hash,),
                )
            elif draw < (
                config.reproduction_probability + config.crossover_probability
            ):
                second_parent = tournament_select(
                    valid_population, rng, config.tournament_size, factory.registry
                )
                second_hash = structural_hash(
                    second_parent.expression, factory.registry
                )
                child = subtree_crossover(
                    first_parent.expression,
                    second_parent.expression,
                    rng,
                    factory.registry,
                )
                origin, parents = "crossover", (first_hash, second_hash)
            else:
                child = mutate_expression(
                    first_parent.expression,
                    rng,
                    factory,
                    terminal_names,
                    operator_names,
                    config.windows,
                    mutation_max_depth=config.mutation_max_depth,
                    terminal_probability=config.terminal_probability,
                )
                origin, parents = "mutation", (first_hash,)
            next_population.append(
                _evaluate_candidate(
                    child,
                    origin,
                    parents,
                    evaluator,
                    factory.registry,
                    limits,
                    fitness_cache,
                    invalid_outcome_factory,
                )
            )
        population = next_population

    accepted_hashes = [
        fingerprint
        for fingerprint in expression_archive
        if fitness_cache[fingerprint].accepted
    ]
    if not accepted_hashes:
        raise RuntimeError("整个搜索没有产生通过 fitness gate 的表达式")
    best_hash = max(
        accepted_hashes,
        key=lambda fingerprint: (
            fitness_cache[fingerprint].fitness,
            -weighted_complexity(expression_archive[fingerprint]),
        ),
    )
    unique_rows: list[dict[str, object]] = []
    for fingerprint, expression in expression_archive.items():
        report = check_expression_constraints(expression, factory.registry, limits)
        unique_rows.append(
            {
                "first_generation": first_generation[fingerprint],
                "canonical": report.canonical,
                "structural_hash": fingerprint,
                "node_count": report.node_count,
                "depth": report.depth,
                "required_history": report.required_history,
                "weighted_complexity": report.weighted_complexity,
                **asdict(fitness_cache[fingerprint]),
            }
        )
    unique = pd.DataFrame(unique_rows).sort_values(
        ["fitness", "weighted_complexity"], ascending=[False, True]
    )
    return GPSearchResult(
        best_expression=expression_archive[best_hash],
        best_outcome=fitness_cache[best_hash],
        occurrence_audit=pd.DataFrame(occurrence_rows),
        unique_candidates=unique.reset_index(drop=True),
        generation_summary=pd.DataFrame(generation_rows),
    )


def run_rank_ic_gp(
    factory: ExpressionFactory,
    evaluator: Callable[[Expression], RankICFitnessOutcome],
    terminal_names: tuple[str, ...],
    operator_names: tuple[str, ...],
    *,
    config: GPConfig = GPConfig(),
    limits: ExpressionLimits = ExpressionLimits(),
) -> GPSearchResult:
    """使用 RankIC outcome 运行 GP，保持 Day 22 的公开接口。"""

    return _run_gp_search(
        factory,
        evaluator,
        terminal_names,
        operator_names,
        config=config,
        limits=limits,
        invalid_outcome_factory=invalid_rank_ic_outcome,
    )


def run_mi_gp(
    factory: ExpressionFactory,
    evaluator: Callable[[Expression], MIFitnessOutcome],
    terminal_names: tuple[str, ...],
    operator_names: tuple[str, ...],
    *,
    config: GPConfig = GPConfig(),
    limits: ExpressionLimits = ExpressionLimits(),
) -> GPSearchResult:
    """使用 KSG MI outcome 运行 GP；是否加惩罚由 evaluator 决定。"""

    return _run_gp_search(
        factory,
        evaluator,
        terminal_names,
        operator_names,
        config=config,
        limits=limits,
        invalid_outcome_factory=invalid_mi_outcome,
    )


def run_cmi_gp(
    factory: ExpressionFactory,
    evaluator: Callable[[Expression], CMIFitnessOutcome],
    terminal_names: tuple[str, ...],
    operator_names: tuple[str, ...],
    *,
    config: GPConfig = GPConfig(),
    limits: ExpressionLimits = ExpressionLimits(),
) -> GPSearchResult:
    """使用 Day 24 的增量 CMI 与三项惩罚 outcome 运行 GP。"""

    return _run_gp_search(
        factory,
        evaluator,
        terminal_names,
        operator_names,
        config=config,
        limits=limits,
        invalid_outcome_factory=invalid_cmi_outcome,
    )
