"""符号表达式去重、类型与无未来函数测试。"""

from __future__ import annotations

import pytest

from src.symbolic import (
    ExpressionFactory,
    audit_expression,
    canonical_expression,
    make_default_registry,
    structural_hash,
)


def test_commutative_form_is_deduplicated() -> None:
    """交换加法输入不能伪装成新的候选因子。"""

    registry = make_default_registry()
    factory = ExpressionFactory(registry)
    first = factory.call("add", factory.terminal("close"), factory.terminal("open"))
    second = factory.call("add", factory.terminal("open"), factory.terminal("close"))
    assert canonical_expression(first, registry) == canonical_expression(second, registry)
    assert structural_hash(first, registry) == structural_hash(second, registry)


def test_label_cannot_be_a_terminal() -> None:
    """表达式引擎不能把未来收益标签当成输入。"""

    factory = ExpressionFactory(make_default_registry())
    with pytest.raises(KeyError):
        factory.terminal("label_1d")


def test_nested_expression_has_no_forward_offset() -> None:
    """滚动与延迟表达式必须机械证明只读取当前及历史数据。"""

    registry = make_default_registry()
    factory = ExpressionFactory(registry)
    expression = factory.call(
        "rank",
        factory.call(
            "ts_mean",
            factory.call("delta", factory.terminal("close"), factory.window(2)),
            factory.window(5),
        ),
    )
    audit = audit_expression(expression, registry)
    assert audit.maximum_forward_offset == 0
    assert audit.required_history == 6
