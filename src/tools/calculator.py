"""Safe arithmetic evaluator exposed as an agent tool.

Uses Python's `ast` module to walk a whitelist of numeric node types
instead of `eval()`, so arbitrary code execution isn't possible even
if the agent forwards untrusted user input.
"""

from __future__ import annotations

import ast
import math
import operator

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e}


class CalculatorError(ValueError):
    pass


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError(f"unsupported constant: {node.value!r}")

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise CalculatorError("only whitelisted math functions are allowed")
        args = [_eval_node(arg) for arg in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise CalculatorError(f"unknown identifier: {node.id}")

    raise CalculatorError(f"unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> float:
    """Evaluate a numeric expression safely. Raises CalculatorError on
    anything outside the arithmetic/math-function whitelist.
    """

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"invalid syntax: {exc}") from exc

    return _eval_node(tree.body)


TOOL_SPEC = {
    "name": "calculator",
    "description": "Evaluate a numeric arithmetic expression (supports + - * / ** % // and sqrt/log/trig functions).",
    "parameters": {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "The expression to evaluate"}},
        "required": ["expression"],
    },
}
