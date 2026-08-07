"""Safe arithmetic evaluator exposed as an agent tool.

Rejects the expression outright if it contains any token associated with
code execution (`__`, `import`, `exec`, `eval`, `os`, `sys`, `subprocess`,
`open`) before ever touching it, then evaluates it with `numexpr` rather
than `eval()` so arbitrary code execution isn't possible even if the
agent forwards untrusted user input.
"""

from __future__ import annotations

import re

import numexpr
from langchain_core.tools import tool

_BLOCKED_PATTERN = re.compile(
    r"__|\bimport\b|\bexec\b|\beval\b|\bos\b|\bsys\b|\bsubprocess\b|\bopen\b",
    re.IGNORECASE,
)


class CalculatorError(ValueError):
    pass


@tool
def calculator(expression: str) -> str:
    """Evaluate a numeric arithmetic expression (e.g. "2 * (3 + 4) / 5", "sqrt(16)") and return the result as a string."""
    if _BLOCKED_PATTERN.search(expression):
        raise CalculatorError(f"expression contains a disallowed token: {expression!r}")

    try:
        result = numexpr.evaluate(expression)
    except Exception as exc:
        raise CalculatorError(f"failed to evaluate {expression!r}: {exc}") from exc

    return str(result.item() if hasattr(result, "item") else result)
