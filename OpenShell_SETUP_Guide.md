# MCP Server + OpenShell Sandbox Setup Guide

## Overview

This guide documents how to run a FastMCP server on the host machine and access it from within an OpenShell sandbox. OpenShell exposes the host via the special hostname `host.openshell.internal`, so no SSH tunnelling is required.

---

## Architecture

```
Sandbox
  └─ python test_mcp_client.py
       └─ connects to host.openshell.internal:9000
            └─ OpenShell proxy (10.200.0.1:3128) → host 172.17.0.1:9000
                 └─ memory_mcp_server.py (FastMCP, port 9000) on host
                      └─ NVIDIA NIM API (nvidia/llama-3.3-nemotron-super-49b-v1.5)
```

---

## Step 1 — Sandbox Policy

The sandbox needs a custom policy file that enables:
- pip installs (pypi endpoints)
- GitHub access
- Claude Code endpoints
- Outbound access to `host.openshell.internal:9000` for the MCP server

The policy file is at: `~/standalone_agent_memory/sandbox_policy.yaml`

Apply it to an existing sandbox:
```bash
openshell policy set <sandbox-name> --policy ~/standalone_agent_memory/sandbox_policy.yaml --wait
```

Or create a new sandbox with it:
```bash
openshell sandbox create --policy ~/standalone_agent_memory/sandbox_policy.yaml --name <sandbox-name>
```

### Key lessons about the policy

- **Do NOT use `protocol: rest` + `enforcement: enforce` for plain HTTP (non-TLS) endpoints** — the OpenShell gateway will intercept and return 403.
- **Private IPs (RFC1918) are blocked entirely** by the OpenShell gateway regardless of the hostname in the policy. Even `host.openshell.internal` resolves to a private IP (`172.17.0.1`) and will be blocked unless you explicitly add `allowed_ips` to the endpoint entry.
- **`allowed_ips` is required** for `host.openshell.internal` because the proxy resolves the hostname and sees an internal address. The correct entry is:
  ```yaml
  - host: host.openshell.internal
    port: 9000
    allowed_ips:
      - 172.17.0.1
  ```
- The proxy log message that indicates this is missing: `FORWARD blocked: internal IP without allowed_ips dst_host=host.openshell.internal`

### SSH notes (OpenShell-specific)

OpenShell uses `russh` as the SSH server inside sandboxes. **It does not support SSH remote port forwarding (`-R`)**. Do not attempt `ssh -R` tunnels — they will silently fail with:
```
Warning: remote port forwarding failed for listen port <N>
```
Use `host.openshell.internal` with `allowed_ips` instead.

---

## Step 2 — MCP Server

Start the server on the host:
```bash
cd ~/standalone_agent_memory
source .venv/bin/activate
python memory_mcp_server.py
```

The server listens on `0.0.0.0:9000/mcp`.

Verify it is up from the host:
```bash
curl -X POST http://0.0.0.0:9000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
```

---

## Step 3 — MCP Client

The client (`test_mcp_client.py`) connects to `host.openshell.internal:9000`, which the OpenShell proxy forwards to the host machine.

```python
MCP_SERVER_URL = "http://host.openshell.internal:9000/mcp"
```

Run from inside the sandbox:
```bash
python test_mcp_client.py
```

---

## Startup Checklist (each session)

On the **host**:
```bash
# Start the MCP server
cd ~/standalone_agent_memory && source .venv/bin/activate
python memory_mcp_server.py &
```

Then from the **sandbox**:
```bash
python test_mcp_client.py
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `403 Forbidden` from `10.x.x.x` | OpenShell blocks private IPs | Use `host.openshell.internal` with `allowed_ips` |
| `403 Forbidden` from `host.openshell.internal` | Missing `allowed_ips` in policy | Add `allowed_ips: [172.17.0.1]` to the endpoint entry and re-apply policy |
| `FORWARD blocked: internal IP without allowed_ips` in logs | Same as above | Same fix |
| `403 Forbidden` from policy endpoint | `protocol: rest` + `enforcement: enforce` on plain HTTP | Remove protocol/enforcement fields from that endpoint |
| `ssh -R` tunnel fails silently | `russh` in OpenShell sandboxes does not support remote port forwarding | Use `host.openshell.internal` approach instead |
| `Invalid json output: <think>` ToolError | LLM returns `<think>` tags before JSON in extraction chain | Fixed in `MemoryManager.py` — ensure `StrOutputParser` + think-tag stripping precedes `JsonOutputParser` |
| `[404] Function not found` from NVIDIA | Model not available for your account | Change model in `memory_mcp_server.py` to `nvidia/llama-3.3-nemotron-super-49b-v1.5` |
| `<think>` tags in final response | Thinking model output not stripped | `strip_think_tags()` in `memory_mcp_server.py` handles this — ensure server is restarted after code changes |
