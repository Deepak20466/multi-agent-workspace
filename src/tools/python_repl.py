"""Sandboxed Python REPL tool.

Executes in a subprocess with a hard 5s timeout and a restricted
builtins set. Imports are allowlisted (pandas, numpy, math, plotly,
json) via a static AST check before the code ever runs, so `os`, `sys`,
`subprocess`, and anything else not on the allowlist are rejected up
front. This is a best-effort sandbox suitable for a trusted-agent /
internal-tool context, not a hard security boundary for untrusted code.

Defense-in-depth layers (see `_check_ast_safety` / `_safe_import` /
`_harden_module` below), because a single check is easy to route around:

1. Static AST check, before anything executes: rejects non-allowlisted
   `import` statements, any bare reference to a dangerous builtin name
   (`eval`, `exec`, `open`, `getattr`, ...) or dangerous module name
   (`os`, `sys`, `subprocess`, ...) *anywhere* in the source -- not just
   at `import` statements, since `pandas.os` or a bare `os` reference
   reads the same module without ever writing `import os` -- and any
   dunder attribute/name access (`.__class__`, `.__globals__`,
   `__builtins__`, ...), which closes the classic "walk the live object
   graph to find an already-imported dangerous module" escape that a
   pure import-statement check can't see.
2. A restricted `__builtins__` dict for the actual `exec()` call, with
   `__import__` replaced by an allowlist-enforcing wrapper rather than
   removed outright -- it has to stay callable for ordinary `import`
   statements to work at all under a custom builtins dict, but calling
   it directly (`__import__("os")`) is exactly how the sandbox was
   previously escaped, so the wrapper re-checks the allowlist itself
   instead of trusting the AST pass alone.
3. Runtime hardening of whatever the allowlisted modules themselves
   expose: pandas/numpy ship real file- and network-capable functions
   (`read_csv`, `to_sql`, `np.load`, ...) that a pure import allowlist
   can't stop, since "pandas" itself is intentionally allowed. Those
   specific entry points are stubbed out on the imported module/class
   objects before the sandboxed code ever gets a reference to them.
"""

from __future__ import annotations

import ast
import builtins
import multiprocessing
import queue
import re

from langchain_core.tools import tool

TIMEOUT_SECONDS = 5

_ALLOWED_MODULES = {"pandas", "numpy", "math", "plotly", "json"}

# Names that must never resolve to anything inside the sandbox, whether
# referenced bare (`ast.Name`) or via attribute access on some other
# object (`ast.Attribute`, e.g. `pandas.os` or `some_obj.system`).
_BLOCKED_NAMES = {
    # exec/eval-family and introspection builtins that can reconstruct
    # access to anything else once called
    "eval", "exec", "compile", "open", "input", "getattr", "setattr",
    "delattr", "vars", "globals", "locals", "breakpoint", "help",
    "exit", "quit", "memoryview", "__import__", "__builtins__",
    "__loader__", "__build_class__",
    # modules explicitly called out as forbidden, plus direct equivalents
    # (process/env/fs/network access, or ways back to the above)
    "os", "sys", "subprocess", "pathlib", "shutil", "socket", "ctypes",
    "importlib", "builtins", "multiprocessing", "threading", "signal",
    "pty", "platform", "tempfile", "gc", "code", "codeop", "pickle",
    "marshal", "ftplib", "smtplib", "http", "urllib", "webbrowser",
    "sqlite3", "resource", "fcntl", "mmap", "asyncio", "inspect",
}

_DUNDER_RE = re.compile(r"^__.+__$")


class PythonREPLError(RuntimeError):
    pass


def _check_ast_safety(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise PythonREPLError(f"syntax error: {exc}") from exc

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
        elif isinstance(node, ast.Name):
            if node.id in _BLOCKED_NAMES or _DUNDER_RE.match(node.id):
                raise PythonREPLError(f"use of '{node.id}' is not allowed in the sandbox")
        elif isinstance(node, ast.Attribute):
            if node.attr in _BLOCKED_NAMES or _DUNDER_RE.match(node.attr):
                raise PythonREPLError(f"access to '.{node.attr}' is not allowed in the sandbox")
            if node.attr == "format":
                # `"{0.__class__}".format(x)` reaches dunder attributes
                # through the format mini-language's own string parsing,
                # entirely invisible to the Attribute/Name checks above
                # since the dangerous name is just string content, not a
                # Python attribute-access AST node.
                raise PythonREPLError("'.format(' calls are not allowed in the sandbox")
        elif isinstance(node, ast.ClassDef):
            raise PythonREPLError("class definitions are not allowed in the sandbox")


def _blocked_stub(name: str):
    def _stub(*_args, **_kwargs):
        raise PythonREPLError(f"'{name}' is disabled in the sandbox (no file/network I/O)")

    return _stub


_PANDAS_BLOCKED_FUNCS = {
    "read_csv", "read_excel", "read_html", "read_json", "read_pickle",
    "read_sql", "read_sql_query", "read_sql_table", "read_hdf",
    "read_parquet", "read_feather", "read_orc", "read_stata", "read_sas",
    "read_spss", "read_table", "read_fwf", "read_clipboard", "read_gbq",
    "ExcelWriter", "HDFStore",
}
_PANDAS_BLOCKED_METHODS = {
    "to_csv", "to_excel", "to_pickle", "to_sql", "to_hdf", "to_parquet",
    "to_feather", "to_clipboard", "to_gbq",
}
_NUMPY_BLOCKED_FUNCS = {
    "load", "save", "savez", "savez_compressed", "fromfile", "tofile",
    "memmap", "genfromtxt", "loadtxt", "savetxt", "DataSource",
}
_PLOTLY_FIGURE_BLOCKED_METHODS = {"write_image", "write_html", "write_json"}

_hardened_module_ids: set[int] = set()


def _harden_module(top_name: str, module) -> None:
    """Stub out file-/network-capable entry points on an already-imported
    allowlisted module, in place, for the lifetime of this one-shot
    worker subprocess (a fresh interpreter per call, so this never
    leaks into the parent process or other invocations).
    """

    if id(module) in _hardened_module_ids:
        return
    _hardened_module_ids.add(id(module))

    if top_name == "pandas":
        for fname in _PANDAS_BLOCKED_FUNCS:
            if hasattr(module, fname):
                try:
                    setattr(module, fname, _blocked_stub(fname))
                except (AttributeError, TypeError):
                    pass
        for cls_name in ("DataFrame", "Series"):
            cls = getattr(module, cls_name, None)
            if cls is None:
                continue
            for mname in _PANDAS_BLOCKED_METHODS:
                if hasattr(cls, mname):
                    try:
                        setattr(cls, mname, _blocked_stub(mname))
                    except (AttributeError, TypeError):
                        pass
    elif top_name == "numpy":
        for fname in _NUMPY_BLOCKED_FUNCS:
            if hasattr(module, fname):
                try:
                    setattr(module, fname, _blocked_stub(fname))
                except (AttributeError, TypeError):
                    pass
    elif top_name == "plotly":
        fig_cls = getattr(getattr(module, "graph_objects", None), "Figure", None)
        if fig_cls is not None:
            for mname in _PLOTLY_FIGURE_BLOCKED_METHODS:
                if hasattr(fig_cls, mname):
                    try:
                        setattr(fig_cls, mname, _blocked_stub(mname))
                    except (AttributeError, TypeError):
                        pass


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Replacement for the `__import__` builtin exposed to sandboxed code.

    `import`/`from ... import ...` statements compiled under a custom
    `__builtins__` dict still need a real, callable `__import__` to work
    at all -- so it can't simply be removed the way `eval`/`exec`/`open`
    are. Instead it's replaced with this wrapper, which re-enforces the
    same allowlist as `_check_ast_safety` at call time. This is what
    closes the previously-verified escape: `__import__("os")` used to
    reach the real `builtins.__import__` directly (bypassing the AST
    check, which only looked at `Import`/`ImportFrom` statement nodes,
    not calls to the `__import__` function), and from there
    `subprocess.run(..., shell=True)` gave arbitrary shell execution.
    """

    top = name.split(".")[0] if name else None
    if top not in _ALLOWED_MODULES:
        raise PythonREPLError(
            f"import of '{name}' is not allowed in the sandbox "
            f"(allowed: {', '.join(sorted(_ALLOWED_MODULES))})"
        )
    module = builtins.__import__(name, globals, locals, fromlist, level)
    _harden_module(top, module)
    return module


_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in ("abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len",
                 "list", "map", "max", "min", "print", "range", "round", "set", "sorted",
                 "str", "sum", "tuple", "zip")
}
_SAFE_BUILTINS["__import__"] = _safe_import


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
