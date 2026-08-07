"""Sandboxed Python REPL tool.

Executes in a subprocess with a hard timeout and a restricted builtins
set, and blocks obviously dangerous imports (os, sys, subprocess, socket,
shutil) via a static AST check before ever running the code. This is a
best-effort sandbox suitable for a trusted-agent / internal-tool context,
not a hard security boundary for untrusted code.
"""

from __future__ import annotations

import ast
import multiprocessing
import queue

TIMEOUT_SECONDS = 5

_BLOCKED_MODULES = {"os", "sys", "subprocess", "socket", "shutil", "ctypes", "pathlib"}

_SAFE_BUILTINS = {
    name: getattr(__builtins__, name) if hasattr(__builtins__, name) else None
    for name in ("abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len",
                 "list", "map", "max", "min", "print", "range", "round", "set", "sorted",
                 "str", "sum", "tuple", "zip")
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
                if name and name.split(".")[0] in _BLOCKED_MODULES:
                    raise PythonREPLError(f"import of '{name}' is not allowed in the sandbox")


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


def run_python(code: str, timeout_seconds: int = TIMEOUT_SECONDS) -> str:
    """Run `code` in an isolated subprocess and return captured stdout.

    Raises PythonREPLError on a blocked import, non-zero exit, or timeout.
    """

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


TOOL_SPEC = {
    "name": "python_repl",
    "description": "Execute a short Python snippet in a sandboxed subprocess and return stdout.",
    "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python code to execute"}},
        "required": ["code"],
    },
}
