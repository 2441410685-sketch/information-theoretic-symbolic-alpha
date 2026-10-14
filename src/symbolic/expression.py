"""Day 20：类型安全、时间可审计的符号因子表达式树。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite
from typing import TypeAlias

import numpy as np
import pandas as pd


class ValueKind(str, Enum):
    """表达式节点的值类型，防止窗口参数与面板变量混用。"""

    PANEL = "panel"
    SCALAR = "scalar"
    WINDOW = "window"


@dataclass(frozen=True, slots=True)
class TerminalSpec:
    """声明一个可被表达式读取的当期字段及其历史需求。"""

    name: str
    value_kind: ValueKind = ValueKind.PANEL
    minimum_history: int = 0
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("terminal 名称不能为空")
        if self.minimum_history < 0:
            raise ValueError("terminal 的 minimum_history 不能为负")


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """声明算子的类型签名和时间语义，不在本文件冻结数值实现。"""

    name: str
    input_kinds: tuple[ValueKind, ...]
    output_kind: ValueKind
    time_semantics: str = "pointwise"
    window_argument: int | None = None
    commutative: bool = False
    forward_offset: int = 0

    def __post_init__(self) -> None:
        allowed = {"pointwise", "cross_sectional", "delay", "rolling"}
        if not self.name:
            raise ValueError("operator 名称不能为空")
        if self.time_semantics not in allowed:
            raise ValueError(f"未知时间语义: {self.time_semantics}")
        if self.time_semantics in {"delay", "rolling"}:
            if self.window_argument is None:
                raise ValueError("delay/rolling 算子必须声明 window_argument")
            if not 0 <= self.window_argument < len(self.input_kinds):
                raise ValueError("window_argument 超出参数范围")
            if self.input_kinds[self.window_argument] is not ValueKind.WINDOW:
                raise ValueError("window_argument 必须指向 WINDOW 类型参数")
        elif self.window_argument is not None:
            raise ValueError("非时序窗口算子不应声明 window_argument")
        if self.forward_offset < 0:
            raise ValueError("forward_offset 不能为负")


@dataclass(frozen=True, slots=True)
class Terminal:
    """表达式树叶节点：读取一个已注册的输入字段。"""

    name: str


@dataclass(frozen=True, slots=True)
class Literal:
    """表达式树叶节点：标量常数或正整数历史窗口。"""

    value: int | float
    value_kind: ValueKind


@dataclass(frozen=True, slots=True)
class Call:
    """表达式树内部节点：对若干子表达式调用一个已注册算子。"""

    operator: str
    arguments: tuple[Expression, ...]


Expression: TypeAlias = Terminal | Literal | Call
OperatorExecutor: TypeAlias = Callable[[tuple[object, ...], "EvaluationContext"], object]


class ExpressionRegistry:
    """保存允许的 terminal/operator，并阻止未声明或未来型操作。"""

    def __init__(
        self,
        terminals: tuple[TerminalSpec, ...],
        operators: tuple[OperatorSpec, ...],
        *,
        maximum_window: int = 252,
        allow_future: bool = False,
    ) -> None:
        if maximum_window < 1:
            raise ValueError("maximum_window 必须至少为 1")
        self.terminals = {spec.name: spec for spec in terminals}
        self.operators = {spec.name: spec for spec in operators}
        if len(self.terminals) != len(terminals):
            raise ValueError("terminal 名称必须唯一")
        if len(self.operators) != len(operators):
            raise ValueError("operator 名称必须唯一")
        if not allow_future and any(spec.forward_offset > 0 for spec in operators):
            raise ValueError("当前研究禁止注册任何未来引用算子")
        self.maximum_window = maximum_window
        self.allow_future = allow_future


class ExpressionFactory:
    """只通过注册表创建合法节点，使错误尽量在生成时暴露。"""

    def __init__(self, registry: ExpressionRegistry) -> None:
        self.registry = registry

    def terminal(self, name: str) -> Terminal:
        """创建已注册字段；label 等未注册字段不能成为因子输入。"""

        node = Terminal(name=name)
        validate_expression(node, self.registry)
        return node

    def scalar(self, value: float) -> Literal:
        """创建有限标量常数。"""

        node = Literal(value=float(value), value_kind=ValueKind.SCALAR)
        validate_expression(node, self.registry)
        return node

    def window(self, value: int) -> Literal:
        """创建正整数回看窗口；零和负数均可能引入错误时间语义。"""

        node = Literal(value=value, value_kind=ValueKind.WINDOW)
        validate_expression(node, self.registry)
        return node

    def call(self, operator: str, *arguments: Expression) -> Call:
        """创建算子节点，并立即执行参数数量与类型检查。"""

        node = Call(operator=operator, arguments=tuple(arguments))
        validate_expression(node, self.registry)
        return node


def _literal_kind(node: Literal) -> ValueKind:
    if node.value_kind not in {ValueKind.SCALAR, ValueKind.WINDOW}:
        raise TypeError("literal 只能是 SCALAR 或 WINDOW")
    if isinstance(node.value, bool):
        raise TypeError("布尔值不能作为数值或窗口")
    if node.value_kind is ValueKind.WINDOW:
        if not isinstance(node.value, int):
            raise TypeError("历史窗口必须是整数")
    elif not isinstance(node.value, (int, float)):
        raise TypeError("标量常数必须是数值")
    return node.value_kind


def expression_kind(expression: Expression, registry: ExpressionRegistry) -> ValueKind:
    """递归推断节点输出类型，同时拒绝未知节点。"""

    if isinstance(expression, Terminal):
        try:
            return registry.terminals[expression.name].value_kind
        except KeyError as exc:
            raise KeyError(f"未注册 terminal: {expression.name}") from exc
    if isinstance(expression, Literal):
        return _literal_kind(expression)
    if isinstance(expression, Call):
        try:
            return registry.operators[expression.operator].output_kind
        except KeyError as exc:
            raise KeyError(f"未注册 operator: {expression.operator}") from exc
    raise TypeError(f"未知表达式节点: {type(expression)!r}")


def validate_expression(expression: Expression, registry: ExpressionRegistry) -> ValueKind:
    """验证整棵树的白名单、参数类型、窗口范围和未来引用。"""

    if isinstance(expression, Terminal):
        return expression_kind(expression, registry)
    if isinstance(expression, Literal):
        kind = expression_kind(expression, registry)
        if kind is ValueKind.WINDOW:
            value = int(expression.value)
            if not 1 <= value <= registry.maximum_window:
                raise ValueError(
                    f"历史窗口必须位于 [1, {registry.maximum_window}]，收到 {value}"
                )
        elif not isfinite(float(expression.value)):
            raise ValueError("标量常数必须是有限值")
        return kind
    if not isinstance(expression, Call):
        raise TypeError(f"未知表达式节点: {type(expression)!r}")
    try:
        spec = registry.operators[expression.operator]
    except KeyError as exc:
        raise KeyError(f"未注册 operator: {expression.operator}") from exc
    if spec.forward_offset > 0 and not registry.allow_future:
        raise ValueError(f"算子 {spec.name} 含未来引用，当前研究禁止")
    if len(expression.arguments) != len(spec.input_kinds):
        raise TypeError(
            f"{spec.name} 需要 {len(spec.input_kinds)} 个参数，"
            f"收到 {len(expression.arguments)} 个"
        )
    actual_kinds = tuple(validate_expression(arg, registry) for arg in expression.arguments)
    if actual_kinds != spec.input_kinds:
        expected = tuple(kind.value for kind in spec.input_kinds)
        actual = tuple(kind.value for kind in actual_kinds)
        raise TypeError(f"{spec.name} 参数类型应为 {expected}，收到 {actual}")
    return spec.output_kind


def _format_literal(node: Literal) -> str:
    if node.value_kind is ValueKind.WINDOW:
        return str(int(node.value))
    value = float(node.value)
    if value == 0.0:
        value = 0.0
    return format(value, ".12g")


def canonical_expression(expression: Expression, registry: ExpressionRegistry) -> str:
    """生成跨运行稳定的规范字符串；交换律参数按子串排序。"""

    validate_expression(expression, registry)
    if isinstance(expression, Terminal):
        return expression.name
    if isinstance(expression, Literal):
        return _format_literal(expression)
    spec = registry.operators[expression.operator]
    arguments = [canonical_expression(arg, registry) for arg in expression.arguments]
    if spec.commutative:
        arguments.sort()
    return f"{spec.name}({','.join(arguments)})"


def parse_canonical_expression(
    text: str,
    factory: ExpressionFactory,
) -> Expression:
    """把本项目的 canonical expression 反解析为类型安全表达式树。

    反解析不接受任意 Python 语法，只读取 ``canonical_expression``
    生成的白名单格式。这使历史冻结公式可直接在新的
    fitness 下复评，无需重跑旧 GP 搜索。
    """

    source = text.strip()
    if not source:
        raise ValueError("canonical expression 不能为空")
    position = 0

    def skip_spaces() -> None:
        nonlocal position
        while position < len(source) and source[position].isspace():
            position += 1

    def read_token() -> str:
        nonlocal position
        skip_spaces()
        start = position
        while position < len(source) and source[position] not in "(),":
            position += 1
        token = source[start:position].strip()
        if not token:
            raise ValueError(f"位置 {start} 缺少节点 token")
        return token

    def parse_node(expected_kind: ValueKind) -> Expression:
        nonlocal position
        token = read_token()
        skip_spaces()
        if position < len(source) and source[position] == "(":
            if expected_kind is not ValueKind.PANEL:
                raise TypeError("当前参数不允许嵌套 PANEL operator")
            try:
                spec = factory.registry.operators[token]
            except KeyError as error:
                raise KeyError(f"未注册 operator: {token}") from error
            position += 1
            arguments: list[Expression] = []
            for argument_number, input_kind in enumerate(spec.input_kinds):
                if argument_number:
                    skip_spaces()
                    if position >= len(source) or source[position] != ",":
                        raise ValueError(f"operator {token} 参数之间缺少逗号")
                    position += 1
                arguments.append(parse_node(input_kind))
            skip_spaces()
            if position >= len(source) or source[position] != ")":
                raise ValueError(f"operator {token} 缺少右括号")
            position += 1
            node = factory.call(token, *arguments)
            if spec.output_kind is not expected_kind:
                raise TypeError(f"operator {token} 输出类型不匹配")
            return node
        if expected_kind is ValueKind.PANEL:
            return factory.terminal(token)
        if expected_kind is ValueKind.WINDOW:
            try:
                value = int(token)
            except ValueError as error:
                raise ValueError(f"窗口参数必须是整数: {token}") from error
            if str(value) != token:
                raise ValueError(f"窗口参数必须使用 canonical 整数: {token}")
            return factory.window(value)
        if expected_kind is ValueKind.SCALAR:
            try:
                return factory.scalar(float(token))
            except ValueError as error:
                raise ValueError(f"标量参数必须是有限数值: {token}") from error
        raise TypeError(f"不支持的 ValueKind: {expected_kind}")

    expression = parse_node(ValueKind.PANEL)
    skip_spaces()
    if position != len(source):
        raise ValueError(f"表达式末尾含未解析内容: {source[position:]}")
    # 要求严格 round-trip，避免两种文本隐式表示同一树。
    if canonical_expression(expression, factory.registry) != source:
        raise ValueError("expression 不是规范 canonical 形式")
    return expression


def structural_hash(expression: Expression, registry: ExpressionRegistry) -> str:
    """对 canonical form 做 SHA-256，供重复表达式识别。"""

    canonical = canonical_expression(expression, registry)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _node_count(expression: Expression) -> int:
    if isinstance(expression, Call):
        return 1 + sum(_node_count(arg) for arg in expression.arguments)
    return 1


def _depth(expression: Expression) -> int:
    if isinstance(expression, Call):
        return 1 + max((_depth(arg) for arg in expression.arguments), default=0)
    return 1


def required_history(expression: Expression, registry: ExpressionRegistry) -> int:
    """计算最早必须回看的交易日数，不把当日 t 计作历史一天。"""

    validate_expression(expression, registry)
    if isinstance(expression, Terminal):
        return registry.terminals[expression.name].minimum_history
    if isinstance(expression, Literal):
        return 0
    spec = registry.operators[expression.operator]
    base = max(required_history(arg, registry) for arg in expression.arguments)
    if spec.window_argument is None:
        return base
    window_node = expression.arguments[spec.window_argument]
    assert isinstance(window_node, Literal)
    window = int(window_node.value)
    # delay(x, d) 最早读取 t-d；rolling(x, w) 包含 t，故新增 w-1。
    return base + (window if spec.time_semantics == "delay" else window - 1)


def maximum_forward_offset(expression: Expression, registry: ExpressionRegistry) -> int:
    """审计整棵树最远向未来读取多少期；合法因子必须等于 0。"""

    validate_expression(expression, registry)
    if not isinstance(expression, Call):
        return 0
    spec = registry.operators[expression.operator]
    child_offset = max(
        (maximum_forward_offset(arg, registry) for arg in expression.arguments),
        default=0,
    )
    return max(spec.forward_offset, child_offset)


def required_terminals(expression: Expression) -> tuple[str, ...]:
    """返回表达式依赖的全部输入字段，便于数据与泄漏审计。"""

    if isinstance(expression, Terminal):
        return (expression.name,)
    if isinstance(expression, Literal):
        return ()
    names: set[str] = set()
    for argument in expression.arguments:
        names.update(required_terminals(argument))
    return tuple(sorted(names))


@dataclass(frozen=True, slots=True)
class ExpressionAudit:
    """面试和实验报告所需的表达式结构摘要。"""

    canonical: str
    structural_hash: str
    value_kind: str
    node_count: int
    depth: int
    operator_count: int
    terminal_count: int
    required_history: int
    maximum_forward_offset: int
    terminals: tuple[str, ...]


def audit_expression(expression: Expression, registry: ExpressionRegistry) -> ExpressionAudit:
    """一次性生成复杂度、窗口、输入和未来引用审计。"""

    kind = validate_expression(expression, registry)
    nodes = list(_walk(expression))
    return ExpressionAudit(
        canonical=canonical_expression(expression, registry),
        structural_hash=structural_hash(expression, registry),
        value_kind=kind.value,
        node_count=_node_count(expression),
        depth=_depth(expression),
        operator_count=sum(isinstance(node, Call) for _, node in nodes),
        terminal_count=sum(isinstance(node, Terminal) for _, node in nodes),
        required_history=required_history(expression, registry),
        maximum_forward_offset=maximum_forward_offset(expression, registry),
        terminals=required_terminals(expression),
    )


def _walk(expression: Expression, path: tuple[int, ...] = ()):
    yield path, expression
    if isinstance(expression, Call):
        for index, argument in enumerate(expression.arguments):
            yield from _walk(argument, path + (index,))


@dataclass(frozen=True, slots=True)
class NodeTrace:
    """单个节点的路径、类型和局部窗口记录。"""

    path: str
    node_kind: str
    name: str
    value_kind: str
    required_history: int
    forward_offset: int


def trace_expression(expression: Expression, registry: ExpressionRegistry) -> tuple[NodeTrace, ...]:
    """逐节点展开表达式，确保每个节点的输入窗口均可追踪。"""

    validate_expression(expression, registry)
    rows: list[NodeTrace] = []
    for path, node in _walk(expression):
        if isinstance(node, Terminal):
            node_kind, name = "terminal", node.name
        elif isinstance(node, Literal):
            node_kind, name = "literal", _format_literal(node)
        else:
            node_kind, name = "call", node.operator
        rows.append(
            NodeTrace(
                path="root" if not path else "root." + ".".join(map(str, path)),
                node_kind=node_kind,
                name=name,
                value_kind=expression_kind(node, registry).value,
                required_history=required_history(node, registry),
                forward_offset=maximum_forward_offset(node, registry),
            )
        )
    return tuple(rows)


def subtree_at_path(expression: Expression, path: tuple[int, ...]) -> Expression:
    """按子节点索引路径读取子树，空路径代表根节点。"""

    current = expression
    for index in path:
        if not isinstance(current, Call):
            raise IndexError("路径试图进入叶节点")
        if not 0 <= index < len(current.arguments):
            raise IndexError(f"子节点索引越界: {index}")
        current = current.arguments[index]
    return current


def replace_subtree(
    expression: Expression,
    path: tuple[int, ...],
    replacement: Expression,
    registry: ExpressionRegistry,
) -> Expression:
    """执行一次类型保持的结构替换，这是 Day 20 的最小变异示例。"""

    validate_expression(expression, registry)
    validate_expression(replacement, registry)
    original = subtree_at_path(expression, path)
    if expression_kind(original, registry) is not expression_kind(replacement, registry):
        raise TypeError("replacement 必须与被替换子树具有相同输出类型")

    def rebuild(node: Expression, remaining: tuple[int, ...]) -> Expression:
        if not remaining:
            return replacement
        if not isinstance(node, Call):
            raise IndexError("路径试图进入叶节点")
        index, *tail = remaining
        if not 0 <= index < len(node.arguments):
            raise IndexError(f"子节点索引越界: {index}")
        arguments = list(node.arguments)
        arguments[index] = rebuild(arguments[index], tuple(tail))
        return Call(node.operator, tuple(arguments))

    mutated = rebuild(expression, path)
    validate_expression(mutated, registry)
    return mutated


@dataclass(slots=True)
class EvaluationContext:
    """表达式执行所需的最小 panel 上下文。"""

    frame: pd.DataFrame
    date_column: str = "date"
    stock_column: str = "stock"

    def __post_init__(self) -> None:
        required = {self.date_column, self.stock_column}
        missing = required.difference(self.frame.columns)
        if missing:
            raise KeyError(f"panel 缺少 key: {sorted(missing)}")
        if self.frame.duplicated([self.date_column, self.stock_column]).any():
            raise ValueError("panel 的 (date, stock) 必须唯一")


def evaluate_expression(
    expression: Expression,
    context: EvaluationContext,
    registry: ExpressionRegistry,
    executors: Mapping[str, OperatorExecutor],
) -> pd.Series | int | float:
    """递归执行表达式；具体算子数值政策由 Day 21 executor 提供。"""

    validate_expression(expression, registry)
    cache: dict[Expression, object] = {}

    def evaluate(node: Expression) -> object:
        if node in cache:
            return cache[node]
        if isinstance(node, Terminal):
            if node.name not in context.frame.columns:
                raise KeyError(f"panel 缺少 terminal 列: {node.name}")
            result: object = pd.to_numeric(context.frame[node.name], errors="raise")
        elif isinstance(node, Literal):
            result = int(node.value) if node.value_kind is ValueKind.WINDOW else float(node.value)
        else:
            try:
                executor = executors[node.operator]
            except KeyError as exc:
                raise KeyError(f"缺少 operator executor: {node.operator}") from exc
            arguments = tuple(evaluate(argument) for argument in node.arguments)
            result = executor(arguments, context)
        if expression_kind(node, registry) is ValueKind.PANEL:
            if not isinstance(result, pd.Series):
                raise TypeError("PANEL 节点 executor 必须返回 pandas.Series")
            if not result.index.equals(context.frame.index):
                raise ValueError("PANEL 节点输出必须与输入 frame 索引完全对齐")
            numeric = pd.to_numeric(result, errors="raise")
            if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValueError("表达式结果含 inf；应使用受保护算子")
            result = numeric
        cache[node] = result
        return result

    output = evaluate(expression)
    return output  # type: ignore[return-value]


def make_default_registry() -> ExpressionRegistry:
    """建立 Day 20 的符号语言；这里只冻结签名，数值语义留给 Day 21。"""

    terminal_names = (
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "turnover_pct",
        "return_1d",
        # V2 只在生成器中使用下面的无量纲终端；保留上面的 V1 terminal，
        # 使已经冻结的历史表达式仍可被解析和复评。
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
    )
    terminals = tuple(
        TerminalSpec(name, description="截至因子日收盘已知的基础行情字段")
        for name in terminal_names
    )
    panel = ValueKind.PANEL
    window = ValueKind.WINDOW
    operators = (
        OperatorSpec("add", (panel, panel), panel, commutative=True),
        OperatorSpec("subtract", (panel, panel), panel),
        OperatorSpec("multiply", (panel, panel), panel, commutative=True),
        OperatorSpec("safe_div", (panel, panel), panel),
        OperatorSpec("delay", (panel, window), panel, "delay", 1),
        OperatorSpec("delta", (panel, window), panel, "delay", 1),
        OperatorSpec("ts_mean", (panel, window), panel, "rolling", 1),
        OperatorSpec("ts_std", (panel, window), panel, "rolling", 1),
        OperatorSpec("ts_min", (panel, window), panel, "rolling", 1),
        OperatorSpec("ts_max", (panel, window), panel, "rolling", 1),
        OperatorSpec("ts_rank", (panel, window), panel, "rolling", 1),
        OperatorSpec("ts_corr", (panel, panel, window), panel, "rolling", 2),
        OperatorSpec("ts_cov", (panel, panel, window), panel, "rolling", 2),
        OperatorSpec("rank", (panel,), panel, "cross_sectional"),
    )
    return ExpressionRegistry(terminals, operators, maximum_window=252)
