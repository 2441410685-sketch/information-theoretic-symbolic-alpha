"""Top-K 组合构建与 A 股状态化执行。"""

from .execution_constraints import (
    ExecutionResult,
    add_open_execution_flags,
    simulate_topk_open_execution,
    summarize_execution_backtest,
)
__all__ = [
    "ExecutionResult",
    "add_open_execution_flags",
    "simulate_topk_open_execution",
    "summarize_execution_backtest",
]
