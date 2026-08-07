"""Tool registry exposing calculator, python_repl, and send_email as
LangChain-compatible tools, plus an MCP server so any MCP-compatible
client (Claude Desktop, other agent frameworks) can call them too.
"""

from __future__ import annotations

from .calculator import calculator, CalculatorError
from .python_repl import python_repl, PythonREPLError
from .send_email import send_email, EmailSendError

ALL_TOOLS = [calculator, python_repl, send_email]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}

__all__ = [
    "calculator",
    "CalculatorError",
    "python_repl",
    "PythonREPLError",
    "send_email",
    "EmailSendError",
    "ALL_TOOLS",
    "TOOL_MAP",
    "build_mcp_server",
]


def build_mcp_server():
    """Construct an MCP `Server` exposing ALL_TOOLS as MCP tools.

    Imported lazily so `import src.tools` doesn't hard-require the `mcp`
    package for callers that only need the plain tool objects.
    """

    from mcp.server import Server
    from mcp.types import Tool, TextContent

    server = Server("multi-agent-workspace-tools")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(name=t.name, description=t.description, inputSchema=t.get_input_schema().model_json_schema())
            for t in ALL_TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name not in TOOL_MAP:
            raise ValueError(f"unknown tool: {name}")
        result = TOOL_MAP[name].invoke(arguments)
        return [TextContent(type="text", text=str(result))]

    return server
