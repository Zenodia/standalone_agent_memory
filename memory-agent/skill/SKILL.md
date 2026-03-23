# MCP Memory Agent

Connect to an MCP-based memory agent server to have conversations with persistent memory per user. The server stores facts, preferences, and context across interactions.

## Trigger

Use this skill when:
- The user asks you to query, talk to, or interact with the **memory agent** or **MCP agent**
- The user wants to store or recall information via the external memory service
- The user asks to reset/restart the memory agent
- The user asks to fetch stored memory items

Keywords: memory agent, mcp agent, mcp memory, remember this (when referring to external memory service)

## Prerequisites

- Uses the Python venv at `/sandbox/test_mcp_client/.venv` (has `fastmcp` pre-installed)
- MCP server reachable at `http://host.openshell.internal:9000/mcp` (override via `MCP_MEMORY_SERVER_URL` env var)

## Tools Available

The MCP server exposes three tools:

| Tool | Purpose | Parameters |
|------|---------|------------|
| `memory_agent` | Conversational query — the agent remembers prior context per user | `query`, `user_id` |
| `restart_memory_agent` | Reset the agent's conversation state for a user | `query` (reason), `user_id` |
| `fetch_memory_items` | Retrieve stored memory items for a user | `query` (filter), `user_id` |

## Usage

Run the client script from this skill's `scripts/` directory:

```bash
# Query the memory agent
python3 scripts/mcp_client.py memory_agent "What do you know about me?" ruth

# Fetch stored memories
python3 scripts/mcp_client.py fetch_memory_items "all" ruth

# Restart/reset the agent
python3 scripts/mcp_client.py restart_memory_agent "fresh start" ruth
```

### Parameters

- **arg 1** — tool name: `memory_agent`, `restart_memory_agent`, or `fetch_memory_items`
- **arg 2** — query string (the message or filter)
- **arg 3** — user_id (optional, defaults to `openclaw`)

### Environment

- `MCP_MEMORY_SERVER_URL` — override the server endpoint (default: `http://host.openshell.internal:9000/mcp`)

## Agent Instructions

When using this skill:

1. **Choose the right tool:**
   - For conversation/questions → `memory_agent`
   - To see what's stored → `fetch_memory_items`
   - To reset state → `restart_memory_agent`

2. **Pick a consistent user_id** — this is how the server tracks memory per user. Use the human's name or a stable identifier.

3. **Strip noise** — the script automatically removes `<think>` blocks from responses. Present the clean output to the user.

4. **Timeout** — allow up to 30 seconds for the MCP call; the memory agent may do RAG lookups.

5. **Error handling** — if the script exits non-zero, report the error and check if the MCP server is reachable.

## Example Workflow

```
User: "Ask the memory agent what it knows about Ruth"

Agent steps:
1. exec: python3 <skill_dir>/scripts/mcp_client.py memory_agent "What do you know about Ruth?" ruth
2. Read stdout
3. Present the response to the user
```
