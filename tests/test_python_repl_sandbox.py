"""Regression tests for the Python REPL sandbox hardening.

`test_repl_sandbox_escape_via_dunder_import_is_blocked` and
`test_repl_sandbox_escape_subprocess_shell_execution_is_blocked` pin down
the exact escape that was previously verified against this tool: calling
the `__import__` builtin directly (rather than writing an `import`
statement) reached the real `os`/`subprocess` modules, bypassing the
AST check that only inspected `Import`/`ImportFrom` statement nodes, and
from there `subprocess.run(..., shell=True)` gave arbitrary shell
command execution. The remaining tests cover equivalent bypass
techniques (object-graph traversal via dunder attributes, `getattr`
indirection, format-string dunder access, module-attribute reach-through
like `pandas.os`) plus confirming legitimate pandas/numpy/math/json
calculations -- the tool's actual purpose -- still work.
"""

import pytest

from src.tools.python_repl import PythonREPLError, python_repl


def _run(code: str) -> str:
    return python_repl.invoke({"code": code})


def _blocked(code: str) -> str:
    with pytest.raises(PythonREPLError) as exc_info:
        _run(code)
    return str(exc_info.value)


# --- legitimate calculations must keep working -----------------------------


def test_benign_arithmetic_executes():
    assert _run("print(2 + 2)") == "4\n"


def test_benign_math_module_executes():
    assert _run("import math\nprint(math.sqrt(16))") == "4.0\n"


def test_benign_json_module_executes():
    assert _run('import json\nprint(json.dumps({"a": 1}))') == '{"a": 1}\n'


def test_benign_pandas_calculation_executes():
    code = 'import pandas as pd\nprint(pd.DataFrame({"a": [1, 2, 3]}).sum().sum())'
    assert _run(code) == "6\n"


def test_benign_numpy_calculation_executes():
    code = "import numpy as np\nprint(np.array([1, 2, 3]).sum())"
    assert _run(code) == "6\n"


# --- the previously verified escape and its exact variants -----------------


def test_repl_sandbox_escape_via_dunder_import_is_blocked():
    """The exact escape verified in the security audit: `__import__("os")`
    used to reach the real `os` module directly, since the old AST check
    only looked for `import`/`from` *statements*, not calls to the
    `__import__` *function*.
    """

    message = _blocked('osmod = __import__("os")\nprint(osmod.listdir("."))')
    assert "__import__" in message


def test_repl_sandbox_escape_subprocess_shell_execution_is_blocked():
    """The full escalation verified in the audit: from `__import__`, reach
    `subprocess.run(..., shell=True)` for arbitrary shell command
    execution (`whoami` in the original finding).
    """

    code = (
        'sp = __import__("subprocess")\n'
        'out = sp.run(["whoami"], capture_output=True, text=True, shell=True)\n'
        "print(out.stdout)"
    )
    message = _blocked(code)
    assert "__import__" in message


def test_bare_dunder_import_reference_is_blocked():
    """Even referencing (not calling) `__import__` must be rejected --
    otherwise it could be stashed in a variable and invoked indirectly.
    """

    assert "__import__" in _blocked("x = __import__\nprint(x)")


def test_plain_os_import_statement_is_still_blocked():
    _blocked("import os\nprint(os.getcwd())")


def test_plain_subprocess_import_statement_is_still_blocked():
    _blocked('import subprocess\nsubprocess.run(["whoami"])')


def test_importlib_import_module_is_blocked():
    _blocked('import importlib\nimportlib.import_module("os")')


# --- equivalent escape mechanisms -------------------------------------------


def test_object_graph_traversal_via_dunder_attributes_is_blocked():
    """Classic sandbox-escape technique: walk the live class graph via
    `__class__`/`__bases__`/`__subclasses__` to find something dangerous
    already loaded in the process, entirely without an `import`.
    """

    message = _blocked("print(().__class__.__bases__[0].__subclasses__())")
    assert "__" in message


def test_getattr_indirection_to_dunder_is_blocked():
    _blocked("print(getattr(1, '__class__'))")


def test_format_string_dunder_bypass_is_blocked():
    """`"{0.__class__}".format(x)` reaches dunder attributes through the
    format mini-language's own string parsing, invisible to a check that
    only looks for literal `.` attribute-access AST nodes.
    """

    _blocked('print("{0.__class__}".format(1))')


def test_eval_is_blocked():
    _blocked('print(eval("1+1"))')


def test_exec_is_blocked():
    _blocked('exec("print(1)")')


def test_open_is_blocked():
    _blocked("open('C:/Windows/win.ini').read()")


def test_dunder_builtins_name_is_blocked():
    _blocked("print(__builtins__)")


def test_class_definition_is_blocked():
    _blocked("class Foo:\n    pass\nprint(Foo)")


def test_module_attribute_reach_through_is_blocked():
    """`pandas.os` (or any `<allowed-module>.os`) reads the same `os`
    module without ever writing `import os` -- must be blocked the same
    as a direct `os` reference.
    """

    _blocked("import pandas as pd\nprint(pd.os)")


# --- allowed modules' own file/network entry points -------------------------


def test_pandas_read_csv_is_disabled_even_though_pandas_is_allowed():
    """`pandas` itself is allowlisted for legitimate dataframe work, but
    its file/network-capable functions must still be unreachable.
    """

    message = _blocked('import pandas as pd\npd.read_csv("C:/Windows/win.ini")')
    assert "read_csv" in message


def test_pandas_dataframe_to_csv_is_disabled():
    message = _blocked('import pandas as pd\npd.DataFrame({"a": [1]}).to_csv("C:/temp/x.csv")')
    assert "to_csv" in message


def test_numpy_load_is_disabled():
    message = _blocked('import numpy as np\nnp.load("x.npy")')
    assert "load" in message


# --- unrelated existing behavior --------------------------------------------


def test_disallowed_module_import_error_message_still_lists_allowed_modules():
    """Not a new check -- confirms the original allowlist error message
    (used by callers/agents surfacing this to a user) is unchanged.
    """

    message = _blocked("import requests")
    assert "allowed" in message
    assert "pandas" in message


def test_timeout_is_still_enforced():
    from src.tools.python_repl import _run_python

    with pytest.raises(PythonREPLError, match="timeout"):
        _run_python("while True:\n    pass", timeout_seconds=1)
