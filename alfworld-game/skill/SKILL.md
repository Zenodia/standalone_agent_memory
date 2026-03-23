# ALFWorld Game

Play ALFWorld text-adventure games via an MCP server. An LLM on the server chooses actions while this client drives the game loop, logs everything, and lets you play step-by-step or in bursts.

## Trigger

Use this skill when:
- The user wants to play, start, continue, or interact with an **ALFWorld** game
- The user mentions **text adventure**, **alfworld**, or **game agent**
- The user asks to see game history or current game state

Keywords: alfworld, text adventure, game, play game, game agent, alfworld game

## Prerequisites

- Python venv at `/sandbox/test_mcp_client/.venv` (has `fastmcp` pre-installed)
- MCP server reachable at `http://host.openshell.internal:9000/mcp` (override via `ALFWORLD_MCP_URL` env var)

## MCP Tools Used

| Tool | Purpose |
|------|---------|
| `reset_env` | Reset the environment and start a new game episode |
| `step_env` | Execute an action in the current game |
| `get_current_state` | Get the current observation, task, score, and available commands |
| `llm_choose_action` | Ask the server-side LLM to pick the next action |

## Commands

Run the client script from this skill's `scripts/` directory:

```bash
# Start a new game and play 1 step (default)
python3 scripts/alfworld_client.py play

# Start a new game and play 10 steps
python3 scripts/alfworld_client.py play --steps 10

# Continue the current game for 5 more steps
python3 scripts/alfworld_client.py continue --steps 5

# Just check the current game state (no actions)
python3 scripts/alfworld_client.py state

# Reset the environment without playing
python3 scripts/alfworld_client.py reset
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--steps N` | Number of steps to take per run | `1` |
| `--url URL` | Override MCP server URL | `http://host.openshell.internal:9000/mcp` |

## Game History

Every game session is logged to **`game_history.md`** inside this skill's root directory (`skills/alfworld-game/game_history.md`). The log includes:
- Game start timestamps and task descriptions
- Each step: LLM-chosen action, observation, score, done status
- Final result: success/failure/paused summary

Read `game_history.md` to review past games or show the user what happened.

## Agent Instructions

1. **New game:** Use `play` to reset and start fresh. Default is 1 step at a time so the user can follow along.
2. **Step-by-step:** Use `play --steps 1` or `continue --steps 1` to advance one step, show the user what happened, and ask if they want to continue.
3. **Batch play:** Use `--steps N` for larger bursts when the user wants faster runs.
4. **Resume:** Use `continue` to pick up where a paused game left off.
5. **Show history:** Read `game_history.md` from this skill directory to display past game logs.
6. **Timeout:** Allow up to 60 seconds per run — the LLM decision + game step can take time.

## Example Workflow

```
User: "Let's play an ALFWorld game"

Agent steps:
1. exec: <skill_dir>/scripts/alfworld_client.py play --steps 1
2. Show the task, observation, and LLM's chosen action
3. Ask: "Want to continue? I can take more steps."

User: "Yeah run 5 more steps"

4. exec: <skill_dir>/scripts/alfworld_client.py continue --steps 5
5. Summarize progress, show score, indicate if task was solved
```
