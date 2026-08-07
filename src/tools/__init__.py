"""Tool registry + MCP server exposing calculator, python_repl, and
send_email as MCP tools so any MCP-compatible client (Claude Desktop,
other agent frameworks) can call them, not just the in-process agents.
"""

from __future__ import annotations

from .calculator import calculate, CalculatorError, TOOL_SPEC as CALCULATOR_SPEC
from .python_repl import run_python, PythonREPLError, TOOL_SPEC as PYTHON_REPL_SPEC
from .send_email import send_email, EmailSendError, TOOL_SPEC as SEND_EMAIL_SPEC

TOOL_REGISTRY = {
    "calculator": {"spec": CALCULATOR_SPEC, "fn": lambda args: calculate(args["expression"])},
    "python_repl": {"spec": PYTHON_REPL_SPEC, "fn": lambda args: run_python(args["code"])},
    "send_email": {
        "spec": SEND_EMAIL_SPEC,
        "fn": lambda args: send_email(args["to"], args["subject"], args["body"]),
    },
}

__all__ = [
    "calculate",
    "CalculatorError",
    "run_python",
    "PythonREPLError",
    "send_email",
    "EmailSendError",
    "TOOL_REGISTRY",
    "build_mcp_server",
]


def build_mcp_server():
    """Construct an MCP `Server` exposing TOOL_REGISTRY as MCP tools.

    Imported lazily so `import src.tools` doesn't hard-require the `mcp`
    package for callers that only need the plain Python functions.
    """

    from mcp.server import Server
    from mcp.types import Tool, TextContent

    server = Server("multi-agent-workspace-tools")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(name=name, description=entry["spec"]["description"], inputSchema=entry["spec"]["parameters"])
            for name, entry in TOOL_REGISTRY.items()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name not in TOOL_REGISTRY:
            raise ValueError(f"unknown tool: {name}")
        result = TOOL_REGISTRY[name]["fn"](arguments)
        return [TextContent(type="text", text=str(result))]

    return server
