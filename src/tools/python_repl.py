"""Sandboxed Python REPL tool.

Executes in a subprocess with a hard 5s timeout and a restricted
builtins set. Imports are allowlisted (pandas, numpy, math, plotly,
json) via a static AST check before the code ever runs, so `os`, `sys`,
`subprocess`, and anything else not on the allowlist are rejected
up front. This is a best-effort sandbox suitable for a trusted-agent /
internal-tool context, not a hard security boundary for untrusted code.
"""

from __future__ import annotations

import ast
import builtins
import multiprocessing
import queue

from langchain_core.tools import tool

TIMEOUT_SECONDS = 5

_ALLOWED_MODULES = {"pandas", "numpy", "math", "plotly", "json"}

_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in ("abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len",
                 "list", "map", "max", "min", "print", "range", "round", "set", "sorted",
                 "str", "sum", "tuple", "zip", "__import__")
}


class PythonREPLError(RuntimeError):
    pass


def _check_ast_safety(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            names = [module] if module else [alias.name for alias in node.names]
            for name in names:
                top = name.split(".")[0] if name else None
                if top not in _ALLOWED_MODULES:
                    raise PythonREPLError(
                        f"import of '{name}' is not allowed in the sandbox "
                        f"(allowed: {', '.join(sorted(_ALLOWED_MODULES))})"
                    )


def _worker(code: str, result_queue: "multiprocessing.Queue") -> None:
    import io
    import contextlib

    stdout = io.StringIO()
    local_scope: dict = {}
    try:
        with contextlib.redirect_stdout(stdout):
            exec(compile(code, "<repl>", "exec"), {"__builtins__": _SAFE_BUILTINS}, local_scope)
        result_queue.put(("ok", stdout.getvalue()))
    except Exception as exc:  # noqa: BLE001 - deliberately broad, forwarded to caller
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_python(code: str, timeout_seconds: int = TIMEOUT_SECONDS) -> str:
    _check_ast_safety(code)

    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue()
    process = ctx.Process(target=_worker, args=(code, result_queue))
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        raise PythonREPLError(f"execution exceeded {timeout_seconds}s timeout")

    try:
        status, payload = result_queue.get_nowait()
    except queue.Empty as exc:
        raise PythonREPLError("no output produced (process may have crashed)") from exc

    if status == "error":
        raise PythonREPLError(payload)
    return payload


@tool
def python_repl(code: str) -> str:
    """Execute a short Python snippet in a sandboxed subprocess (5s timeout) and return captured stdout. pandas, numpy, math, plotly, and json may be imported; os, sys, and subprocess are blocked."""
    return _run_python(code)
