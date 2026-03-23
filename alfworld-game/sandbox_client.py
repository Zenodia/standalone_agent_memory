"""
llm_play_alfworld_mcp_client.py
--------------------------------
MCP client that uses the LLM from memory_mcp_server.py
(nvidia/llama-3.3-nemotron-super-49b-v1.5) to play the ALFWorld text game
through the alfworld_env_mcp_server.py MCP server.

Usage
-----
    # 1. In one terminal, start the environment server:
    #       python alfworld_env_mcp_server.py
    #
    # 2. In another terminal, run this client:
    #       python llm_play_alfworld_mcp_client.py

The client connects to http://localhost:9001/mcp, resets a game,
then lets the LLM pick actions until the task is done or MAX_STEPS is reached.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from colorama import Fore, init
from fastmcp import Client

init(autoreset=True)

# ── Configuration ─────────────────────────────────────────────────────────────

ALFWORLD_MCP_URL = "http://host.openshell.internal:9000/mcp"
MAX_STEPS = 1
#HISTORY_FILE = Path(__file__).resolve().parent / "game_history.md"
open_claw_skill_path = Path("/sandbox/.openclaw/workspace/skills/")
HISTORY_FILE = open_claw_skill_path / "game_history_sandbox.md"


# ── Markdown helpers ──────────────────────────────────────────────────────────
def _md_new_game(f, task: str, obs: str, banner: str) -> None:
    """Write the header for a new game section."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"\n---\n\n## {banner} — {timestamp}\n\n")
    f.write(f"**Task:** {task}\n\n")
    f.write(f"**Initial Observation:**\n> {obs}\n\n")


def _md_resume(f, step: int) -> None:
    """Write a resume marker."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"\n### ↩ Resumed at step {step} — {timestamp}\n\n")


def _md_step(f, step: int, action: str, obs: str, score: float, done: bool) -> None:
    """Write a single step entry."""
    f.write(f"### Step {step:02d}\n")
    f.write(f"**Action:** `{action}`\n\n")
    f.write(f"**Observation:** {obs}\n\n")
    f.write(f"**Score:** {score} | **Done:** {done}\n\n")


def _md_summary(f, done: bool, score: float, total_steps: int) -> None:
    """Write the final result."""
    f.write("### Result\n")
    if done and score > 0:
        f.write(f"✅ **SUCCESS** — completed in {total_steps} steps. Final score: {score}\n\n")
    elif done:
        f.write(f"❌ **Episode ended** at step {total_steps}. Final score: {score}\n\n")
    else:
        f.write(f"⏸ **Paused** after {MAX_STEPS} steps (global step {total_steps}). "
                f"Call `play_game()` to continue.\n\n")


# ── Main game loop ────────────────────────────────────────────────────────────
async def play_game(reset: bool = False, history: list[dict] | None = None) -> None:
    """
    Play (or continue) an ALFWorld game via the MCP server.

    Args:
        reset:   True  → call reset_env and start a brand-new episode.
                 False → resume from the server's current state; auto-resets
                         only if no game has been started yet.
        history: Prior (action, observation) pairs to include as context for
                 the LLM.  Pass the list returned by a previous call to
                 play_game() to carry context across runs.  Defaults to [].
    """
    if history is None:
        history = []

    print(Fore.CYAN + f"\nConnecting to ALFWorld MCP server at {ALFWORLD_MCP_URL} …\n")

    async with Client(ALFWORLD_MCP_URL) as client:
        # ── Decide whether to reset or resume ────────────────────────────────
        if reset:
            raw = await client.call_tool("reset_env", {})
            state = json.loads(raw.content[0].text)
            banner = "GAME START (fresh reset)"
        else:
            raw = await client.call_tool("get_current_state", {})
            state = json.loads(raw.content[0].text)
            # Auto-reset when the env has never been initialised
            if not state.get("observation"):
                raw = await client.call_tool("reset_env", {})
                state = json.loads(raw.content[0].text)
                banner = "GAME START (auto-reset: no active game on server)"
            else:
                banner = f"RESUMING from step {state['step']}"

        task = state["task"]
        obs = state["observation"]
        admissible = state["admissible_commands"]
        score = state.get("score", 0.0)
        done = state.get("done", False)
        server_step = state.get("step", 0)   # offset so display step is always global
        current_step = server_step           # env step count; passed to llm_choose_action each turn

        print(Fore.GREEN + "=" * 70)
        print(Fore.GREEN + banner)
        print(Fore.GREEN + "=" * 70)
        print(Fore.WHITE + obs)
        print(Fore.YELLOW + f"\nTask: {task}")
        print(Fore.CYAN + f"Available actions ({len(admissible)}): {admissible[:5]}{'…' if len(admissible) > 5 else ''}")
        print()

        if done:
            print(Fore.YELLOW + "Episode is already finished on the server. Pass reset=True to start a new game.")
            return

        with open(HISTORY_FILE, "a", encoding="utf-8") as md:
            # ── Write header block once per play_game() call ─────────────────
            if "GAME START" in banner:
                _md_new_game(md, task, obs, banner)
            else:
                _md_resume(md, server_step)

            for i in range(1, MAX_STEPS + 1):
                display_step = server_step + i   # global step number across all play_game() calls

                # ── LLM decides via MCP tool on the server ───────────────────
                raw = await client.call_tool(
                    "llm_choose_action",
                    {
                        "task": task,
                        "observation": obs,
                        "admissible_commands": admissible,
                        "history": history,
                        "score": score,
                        "done": done,
                        "step": current_step,
                    },
                )
                action = raw.content[0].text.strip()
                print(Fore.MAGENTA + f"[Step {display_step:02d}] LLM chose: » {action} «")

                # ── Execute action ────────────────────────────────────────────
                raw = await client.call_tool("step_env", {"action": action})
                result = json.loads(raw.content[0].text)

                if "error" in result:
                    print(Fore.RED + f"[ERROR] {result['error']}")
                    md.write(f"> ⚠️ Error: {result['error']}\n\n")
                    break

                obs = result["observation"]
                score = result["score"]
                done = result["done"]
                current_step = result.get("step", current_step)
                admissible = result.get("admissible_commands", [])

                print(Fore.WHITE + f"         Obs: {obs[:200]}{'…' if len(obs) > 200 else ''}")
                print(Fore.BLUE + f"         Score: {score}  |  Done: {done}")
                print()

                _md_step(md, display_step, action, obs, score, done)
                history.append({"action": action, "observation": obs})

                if done:
                    break

            # ── Final summary ─────────────────────────────────────────────────
            total_steps = server_step + i
            _md_summary(md, done, score, total_steps)

        print(Fore.GREEN + "=" * 70)
        if done and score > 0:
            print(Fore.GREEN + f"SUCCESS! Task completed at step {total_steps}. Final score: {score}")
        elif done:
            print(Fore.RED + f"Episode ended at step {total_steps}. Final score: {score}")
        else:
            print(Fore.YELLOW + f"Paused after {MAX_STEPS} steps (global step {total_steps}). Call play_game() to continue.")
        print(Fore.GREEN + "=" * 70)
        print(Fore.CYAN + f"History written to: {HISTORY_FILE}")

        return history   # caller can pass this back on the next call to preserve LLM context


if __name__ == "__main__":
    # Example: run MAX_STEPS steps, then continue for another MAX_STEPS steps.
    # First call resets; subsequent calls resume automatically.
    history = asyncio.run(play_game(reset=True))
    # history = asyncio.run(play_game(history=history))   # uncomment to continue
