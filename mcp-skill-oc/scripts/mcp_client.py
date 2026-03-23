#!/sandbox/test_mcp_client/.venv/bin/python3
"""
MCP Memory Agent client — thin wrapper around the MCP server.

Usage:
  python3 mcp_client.py <tool_name> <query> [user_id]

Tools:
  memory_agent          — Query the memory agent (conversational, remembers context)
  restart_memory_agent  — Restart/reset the memory agent for a user
  fetch_memory_items    — Fetch stored memory items for a user

Exit codes:
  0  success (output on stdout)
  1  error   (message on stderr)
"""

import asyncio
import json
import sys
import os
import warnings

# Suppress all deprecation warnings (fastmcp emits them from deep in contextlib)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*streamable_http_client.*")

# Allow overriding the MCP server URL via environment variable
MCP_SERVER_URL = os.environ.get(
    "MCP_MEMORY_SERVER_URL",
    "http://host.openshell.internal:9000/mcp"
)

VALID_TOOLS = {"memory_agent", "restart_memory_agent", "fetch_memory_items"}


async def call_mcp(tool_name: str, query: str, user_id: str) -> str:
    """Connect to the MCP server and call the specified tool."""
    from fastmcp import Client

    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool(
            tool_name,
            {"query": query, "user_id": user_id},
        )

    # Extract text from first content block
    if hasattr(result, "content") and result.content:
        text = result.content[0].text
    elif isinstance(result, dict):
        text = result.get("text", json.dumps(result))
    else:
        text = str(result)

    return text


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks that some models emit."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def main():
    if len(sys.argv) < 3:
        print("Usage: mcp_client.py <tool_name> <query> [user_id]", file=sys.stderr)
        sys.exit(1)

    tool_name = sys.argv[1]
    query = sys.argv[2]
    user_id = sys.argv[3] if len(sys.argv) > 3 else "openclaw"

    if tool_name not in VALID_TOOLS:
        print(f"Error: unknown tool '{tool_name}'. Valid: {', '.join(sorted(VALID_TOOLS))}", file=sys.stderr)
        sys.exit(1)

    try:
        output = asyncio.run(call_mcp(tool_name, query, user_id))
        # Clean up chain-of-thought leakage
        output = strip_think_tags(output)
        print(output)
    except Exception as e:
        print(f"Error calling MCP server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
