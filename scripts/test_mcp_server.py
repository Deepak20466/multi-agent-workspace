"""Local smoke test for src/mcp_server.py: spawns it as a subprocess over
stdio (the same transport Claude Desktop uses) and calls each of the four
tools once, printing the result.

Usage: python scripts/test_mcp_server.py
"""

from __future__ import annotations

import asyncio
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "src.mcp_server"])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Registered tools:", [t.name for t in tools.tools])
            print()

            calls = [
                ("search_docs", {"query": "What is the refund policy?"}),
                ("query_sql", {"question": "How many rows are in the orders table?"}),
                ("extract_document", {"file_path": "data/sample_documents/refund_policy.txt", "query": "What is the refund window?"}),
                ("web_search", {"query": "current weather in Paris"}),
            ]

            for name, args in calls:
                print(f"--- {name}({args}) ---")
                result = await session.call_tool(name, args)
                for block in result.content:
                    if block.type == "text":
                        print(block.text)
                print()


if __name__ == "__main__":
    asyncio.run(main())
