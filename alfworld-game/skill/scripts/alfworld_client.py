#!/sandbox/test_mcp_client/.venv/bin/python3
"""
ALFWorld MCP Game Client
-------------------------
Plays the ALFWorld text-adventure game through an MCP server.
An LLM on the server chooses actions; this client drives the game loop
and logs every step to game_history.md inside the skill directory.

Usage:
  python3 alfworld_client.py [command] [options]

Commands:
  play          Start a new game (reset) and run MAX_STEPS steps (default)
  continue      Resume the current game for MAX_STEPS more steps
  state         Print the current game state (no actions taken)
  reset         Reset the environment and print the new game state

Options:
  --steps N     Override MAX_STEPS (default: 1)
  --url URL     Override MCP server URL

Exit codes:
  0  success
  1  error
"""

import asyncio
import json
import sys
import os
import warnings
from datetime import datetime
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*streamable_http_client.*")

from fastmcp import Client

# ── Configuration ─────────────────────────────────────────────────────────────

ALFWORLD_MCP_URL = os.environ.get(
    "ALFWORLD_MCP_URL",
    "http://host.openshell.internal:9000/mcp",
)

SKILL_DIR = Path(__file__).resolve().parent.parent
HISTORY_FILE = SKILL_DIR / "game_history.md"

DEFAULT_MAX_STEPS = 1


# ── Markdown helpers ──────────────────────────────────────────────────────────

def _md_new_game(f, task: str, obs: str, banner: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"\n---\n\n## {banner} — {timestamp}\n\n")
    f.write(f"**Task:** {task}\n\n")
    f.write(f"**Initial Observation:**\n> {obs}\n\n")


def _md_resume(f, step: int) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"\n### ↩ Resumed at step {step} — {timestamp}\n\n")


def _md_step(f, step: int, action: str, obs: str, score: float, done: bool) -> None:
    f.write(f"### Step {step:02d}\n")
    f.write(f"**Action:** `{action}`\n\n")
    f.write(f"**Observation:** {obs}\n\n")
    f.write(f"**Score:** {score} | **Done:** {done}\n\n")


def _md_summary(f, done: bool, score: float, total_steps: int, max_steps: int) -> None:
    f.write("### Result\n")
    if done and score > 0:
        f.write(f"✅ **SUCCESS** — completed in {total_steps} steps. Final score: {score}\n\n")
    elif done:
        f.write(f"❌ **Episode ended** at step {total_steps}. Final score: {score}\n\n")
    else:
        f.write(f"⏸ **Paused** after {max_steps} steps (global step {total_steps}). "
                f"Run `continue` to keep going.\n\n")


# ── History enrichment ────────────────────────────────────────────────────────

def _build_enriched_history(
    history: list[dict],
    task: str,
    current_obs: str,
    admissible: list[str],
    step: int,
) -> list[dict]:
    """
    Build an enriched history list that includes:
    1. A summary of steps taken so far
    2. Trajectory suggestions for the best next actions
    3. The raw recent history (last 5 steps to keep context window small)
    """
    enriched: list[dict] = []

    # ── Step summary ──
    if history:
        actions_taken = [h["action"] for h in history]
        unique_locations = set()
        items_found = []
        items_held = []
        cleaned_items = []

        for h in history:
            obs_lower = h["observation"].lower()
            act_lower = h["action"].lower()

            # Track locations visited
            if act_lower.startswith("go to "):
                unique_locations.add(act_lower.replace("go to ", ""))

            # Track items found
            if "you see a " in obs_lower or "you see an " in obs_lower:
                for token in ["you see a ", "you see an "]:
                    if token in obs_lower:
                        items_part = obs_lower.split(token, 1)[-1].split(".")[0]
                        items_found.append(items_part.strip())

            # Track items taken
            if act_lower.startswith("take "):
                items_held.append(act_lower.replace("take ", "").split(" from ")[0])

            # Track cleaning
            if act_lower.startswith("clean "):
                cleaned_items.append(act_lower.replace("clean ", "").split(" with ")[0])

        summary = f"STEP SUMMARY (step {step}, {len(history)} actions so far):\n"
        summary += f"  Task: {task}\n"
        summary += f"  Actions taken: {', '.join(actions_taken[-10:])}\n"
        summary += f"  Locations visited: {', '.join(sorted(unique_locations)) if unique_locations else 'none'}\n"
        if items_held:
            summary += f"  Items picked up: {', '.join(items_held)}\n"
        if cleaned_items:
            summary += f"  Items cleaned: {', '.join(cleaned_items)}\n"

        enriched.append({
            "action": "[SUMMARY]",
            "observation": summary,
        })

    # ── Trajectory suggestions ──
    task_lower = task.lower()
    suggestions = _suggest_trajectory(task_lower, history, current_obs, admissible)
    if suggestions:
        enriched.append({
            "action": "[TRAJECTORY HINT]",
            "observation": suggestions,
        })

    # ── Recent raw history (last 5 steps for detail) ──
    recent = history[-5:] if len(history) > 5 else history
    enriched.extend(recent)

    return enriched


def _suggest_trajectory(
    task: str, history: list[dict], current_obs: str, admissible: list[str]
) -> str:
    """Generate smart trajectory suggestions based on task type and game state."""
    obs_lower = current_obs.lower()
    actions_done = [h["action"].lower() for h in history]

    hints: list[str] = []

    # Detect task type and suggest accordingly
    if "put" in task and "clean" in task:
        # Clean-and-put task (e.g. "put a clean fork in drawer")
        has_item = any(a.startswith("take ") for a in actions_done)
        has_cleaned = any(a.startswith("clean ") for a in actions_done)
        has_gone_to_sink = any("sinkbasin" in a for a in actions_done)

        if not has_item:
            hints.append("PRIORITY: Find and pick up the target item first. Use 'take <item> from <location>'.")
            # Find relevant take/go actions
            for cmd in admissible:
                if cmd.startswith("take "):
                    hints.append(f"  Suggested: {cmd}")
        elif not has_gone_to_sink and not has_cleaned:
            hints.append("PRIORITY: Go to sinkbasin to clean the item. Use 'go to sinkbasin 1'.")
        elif has_gone_to_sink and not has_cleaned:
            hints.append("PRIORITY: Clean the item now. Use 'clean <item> with sinkbasin 1'.")
            for cmd in admissible:
                if cmd.startswith("clean "):
                    hints.append(f"  Suggested: {cmd}")
        elif has_cleaned:
            hints.append("PRIORITY: Item is clean! Now put it in the target location.")
            hints.append("Use 'put <item> in/on <location>'. Do NOT keep exploring.")
            for cmd in admissible:
                if cmd.startswith("put "):
                    hints.append(f"  Suggested: {cmd}")

    elif "put" in task and "hot" in task:
        # Heat-and-put task
        has_item = any(a.startswith("take ") for a in actions_done)
        has_heated = any("microwave" in a or "stoveburner" in a for a in actions_done)

        if not has_item:
            hints.append("PRIORITY: Find and pick up the target item.")
        elif not has_heated:
            hints.append("PRIORITY: Go to microwave or stoveburner to heat the item.")
        else:
            hints.append("PRIORITY: Item is heated! Put it in the target location now.")
            for cmd in admissible:
                if cmd.startswith("put "):
                    hints.append(f"  Suggested: {cmd}")

    elif "put" in task and "cool" in task:
        # Cool-and-put task
        has_item = any(a.startswith("take ") for a in actions_done)
        has_cooled = any("fridge" in a for a in actions_done)

        if not has_item:
            hints.append("PRIORITY: Find and pick up the target item.")
        elif not has_cooled:
            hints.append("PRIORITY: Go to fridge to cool the item.")
        else:
            hints.append("PRIORITY: Item is cooled! Put it in the target location now.")
            for cmd in admissible:
                if cmd.startswith("put "):
                    hints.append(f"  Suggested: {cmd}")

    elif "put" in task:
        # Simple put task
        has_item = any(a.startswith("take ") for a in actions_done)
        if not has_item:
            hints.append("PRIORITY: Find and pick up the target item first.")
        else:
            hints.append("PRIORITY: You have the item. Put it in the target location now.")
            for cmd in admissible:
                if cmd.startswith("put "):
                    hints.append(f"  Suggested: {cmd}")

    elif "examine" in task or "look" in task:
        hints.append("PRIORITY: Navigate to the target object and examine/look at it.")

    elif "find" in task:
        hints.append("PRIORITY: Search locations systematically. Check countertops, tables, drawers, shelves.")

    # Anti-loop detection
    if len(actions_done) >= 4:
        last_4 = actions_done[-4:]
        if len(set(last_4)) <= 2:
            hints.append("WARNING: You seem to be repeating actions! Try a DIFFERENT approach.")
            hints.append("Pick a 'put' or 'use' action instead of navigating.")

    if len(actions_done) >= 6:
        last_6 = actions_done[-6:]
        go_actions = [a for a in last_6 if a.startswith("go to")]
        if len(go_actions) >= 4:
            hints.append("WARNING: Too much navigation! If you have the item, USE it or PUT it now.")
            for cmd in admissible:
                if cmd.startswith("put ") or cmd.startswith("use ") or cmd.startswith("clean "):
                    hints.append(f"  Try instead: {cmd}")

    return "\n".join(hints) if hints else ""


# ── Commands ──────────────────────────────────────────────────────────────────

async def cmd_state() -> None:
    """Print the current game state without taking any action."""
    async with Client(ALFWORLD_MCP_URL) as client:
        raw = await client.call_tool("get_current_state", {})
        state = json.loads(raw.content[0].text)
        print(json.dumps(state, indent=2))


async def cmd_reset() -> None:
    """Reset the environment and print the fresh state."""
    async with Client(ALFWORLD_MCP_URL) as client:
        raw = await client.call_tool("reset_env", {})
        state = json.loads(raw.content[0].text)
        print(json.dumps(state, indent=2))


async def play_game(reset: bool = True, max_steps: int = DEFAULT_MAX_STEPS) -> None:
    """Play or continue an ALFWorld game."""
    history: list[dict] = []

    async with Client(ALFWORLD_MCP_URL) as client:
        # Decide whether to reset or resume
        if reset:
            raw = await client.call_tool("reset_env", {})
            state = json.loads(raw.content[0].text)
            banner = "GAME START (fresh reset)"
        else:
            raw = await client.call_tool("get_current_state", {})
            state = json.loads(raw.content[0].text)
            if not state.get("observation"):
                raw = await client.call_tool("reset_env", {})
                state = json.loads(raw.content[0].text)
                banner = "GAME START (auto-reset: no active game)"
            else:
                banner = f"RESUMING from step {state['step']}"

        task = state["task"]
        obs = state["observation"]
        admissible = state["admissible_commands"]
        score = state.get("score", 0.0)
        done = state.get("done", False)
        server_step = state.get("step", 0)

        # Print game header
        print(f"{'=' * 60}")
        print(banner)
        print(f"{'=' * 60}")
        print(f"Task: {task}")
        print(f"Observation: {obs}")
        print(f"Available actions ({len(admissible)}): {admissible[:5]}{'…' if len(admissible) > 5 else ''}")
        print()

        if done:
            print("Episode already finished. Use 'play' to start a new game.")
            return

        with open(HISTORY_FILE, "a", encoding="utf-8") as md:
            if "GAME START" in banner:
                _md_new_game(md, task, obs, banner)
            else:
                _md_resume(md, server_step)

            i = 0
            for i in range(1, max_steps + 1):
                display_step = server_step + i

                # Build enriched history with summary and trajectory hints
                enriched_history = _build_enriched_history(
                    history, task, obs, admissible, display_step
                )

                # LLM decides action via MCP
                raw = await client.call_tool(
                    "llm_choose_action",
                    {
                        "task": task,
                        "observation": obs,
                        "admissible_commands": admissible,
                        "history": enriched_history,
                    },
                )
                action = raw.content[0].text.strip()
                print(f"[Step {display_step:02d}] LLM chose: » {action} «")

                # Execute action
                raw = await client.call_tool("step_env", {"action": action})
                result = json.loads(raw.content[0].text)

                if "error" in result:
                    print(f"[ERROR] {result['error']}")
                    md.write(f"> ⚠️ Error: {result['error']}\n\n")
                    break

                obs = result["observation"]
                score = result["score"]
                done = result["done"]
                admissible = result.get("admissible_commands", [])

                print(f"  Obs: {obs[:200]}{'…' if len(obs) > 200 else ''}")
                print(f"  Score: {score}  |  Done: {done}")
                print()

                _md_step(md, display_step, action, obs, score, done)
                history.append({"action": action, "observation": obs})

                if done:
                    break

            total_steps = server_step + (i if i > 0 else 0)
            _md_summary(md, done, score, total_steps, max_steps)

        print(f"{'=' * 60}")
        if done and score > 0:
            print(f"SUCCESS! Task completed at step {total_steps}. Final score: {score}")
        elif done:
            print(f"Episode ended at step {total_steps}. Final score: {score}")
        else:
            print(f"Paused after {max_steps} steps (global step {total_steps}). Run 'continue' to keep going.")
        print(f"{'=' * 60}")
        print(f"History: {HISTORY_FILE}")


def main():
    args = sys.argv[1:]

    # Parse options
    max_steps = DEFAULT_MAX_STEPS
    url_override = None
    positional = []

    i = 0
    while i < len(args):
        if args[i] == "--steps" and i + 1 < len(args):
            max_steps = int(args[i + 1])
            i += 2
        elif args[i] == "--url" and i + 1 < len(args):
            url_override = args[i + 1]
            i += 2
        else:
            positional.append(args[i])
            i += 1

    if url_override:
        global ALFWORLD_MCP_URL
        ALFWORLD_MCP_URL = url_override

    command = positional[0] if positional else "play"

    if command == "play":
        asyncio.run(play_game(reset=True, max_steps=max_steps))
    elif command == "continue":
        asyncio.run(play_game(reset=False, max_steps=max_steps))
    elif command == "state":
        asyncio.run(cmd_state())
    elif command == "reset":
        asyncio.run(cmd_reset())
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Commands: play, continue, state, reset", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
