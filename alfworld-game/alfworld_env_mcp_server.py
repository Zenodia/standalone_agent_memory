"""
alfworld_env_mcp_server.py
--------------------------
FastMCP server that exposes the ALFWorld text-based game environment as MCP tools.

Tools
-----
reset_env()                  – Start / restart a game episode.
step_env(action)             – Execute an action and return the result.
get_admissible_commands()    – Return the currently valid action strings.
get_current_state()          – Return a summary of the current game state.

Run
---
    python alfworld_env_mcp_server.py          # listens on 0.0.0.0:9001/mcp
"""

import asyncio
import json
import os
import re
from pathlib import Path

import yaml
from colorama import Fore
from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

# ── ALFWorld environment ─────────────────────────────────────────────────────
_DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "base_config.yaml"

# ── LLM (same model as memory_mcp_server.py) ─────────────────────────────────
llm = ChatNVIDIA(model="nvidia/llama-3.3-nemotron-super-49b-v1.5")

def _build_system_prompt(
    task: str,
    observation: str,
    admissible_commands: list[str],
    score: float,
    done: bool,
    step: int,
) -> str:
    """Per-step system message: task, state, admissible actions, score, and completion flag."""
    status_line = (
        "Yes — the episode has ended (finished / terminal state)."
        if done
        else "No — the episode is still in progress (not finished yet)."
    )
    if admissible_commands:
        actions_block = "\n".join(f"{i + 1}. {cmd}" for i, cmd in enumerate(admissible_commands))
    else:
        actions_block = "(none — there are no valid actions, likely because the episode has ended.)"

    return f"""You are an expert agent playing a text-based household task game (ALFWorld/TextWorld).
Your goal is to complete the assigned task for this episode as efficiently as possible.

## Current episode (updates every step)

**Task (goal — specific to this game / environment):**
{task or "(not found in observation; infer from context if needed)"}

**Current observation:**
{observation}

**Score thus far (cumulative reward):**
{score}

**Is the game done (finished / completed / terminal)?**
{status_line}

**Environment step count (steps taken so far in this episode):**
{step}

**Admissible actions for this step** (the list changes after every action; copy text exactly):
{actions_block}

## How to reply

- Reply with ONLY the exact action text, copied verbatim from the admissible list above.
- Do NOT include the number, any explanation, punctuation, or extra words.
- Think step-by-step about which action best progresses toward the task goal."""


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks produced by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()



def _load_config(config_path: str | None = None) -> dict:
    path = config_path or str(_DEFAULT_CONFIG)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ALFWorld config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


# Global env state
_env = None
_current_obs: str = ""
_current_score: float = 0.0
_current_done: bool = False
_admissible_commands: list[str] = []
_task_desc: str = ""
_step_count: int = 0


def _init_env() -> None:
    """Lazily initialise the ALFWorld environment (done once per server start)."""
    global _env
    if _env is not None:
        return
    from alfworld.agents.environment import get_environment

    config = _load_config()
    config["env"]["type"] = "AlfredTWEnv"
    raw_env = get_environment("AlfredTWEnv")(config, train_eval="train")
    _env = raw_env.init_env(batch_size=1)
    print("[alfworld_env_mcp_server] Environment initialised.")


def _extract_task(obs: str) -> str:
    for line in obs.splitlines():
        if line.strip().lower().startswith("your task is"):
            return line.strip()
    return ""


# ── FastMCP app ───────────────────────────────────────────────────────────────
mcp = FastMCP("AlfWorldEnvMCP")


@mcp.tool()
async def reset_env() -> str:
    """
    Reset the ALFWorld environment and start a new game episode.

    Returns a JSON string with keys:
        observation  (str)  – The initial text observation.
        admissible_commands (list[str]) – Valid actions at this state.
        task         (str)  – The task the agent must complete.
        step         (int)  – Current step counter (0 after reset).
    """
    global _current_obs, _current_score, _current_done, _admissible_commands
    global _task_desc, _step_count

    _init_env()
    obs, info = _env.reset()

    _current_obs = obs[0]
    _current_score = 0.0
    _current_done = False
    _admissible_commands = list(info["admissible_commands"][0])
    _task_desc = _extract_task(_current_obs)
    _step_count = 0

    result = {
        "observation": _current_obs,
        "admissible_commands": _admissible_commands,
        "task": _task_desc,
        "step": _step_count,
    }
    return json.dumps(result)

@mcp.tool()
def llm_choose_action(
    task: str,
    observation: str,
    admissible_commands: list[str],
    history: list[dict],
    score: float = 0.0,
    done: bool = False,
    step: int = 0,
) -> str:
    """
    Ask the LLM to pick the next best action.

    task, observation, admissible_commands, score, done, and step should reflect the
    current environment state before taking the next action.

    history  – list of {"action": str, "observation": str} dicts from past steps.
    """
    history_text = ""
    if history:
        lines = []
        for i, h in enumerate(history[-5:], 1):   # last 5 steps for context
            lines.append(f"  Step {i}: [{h['action']}] → {h['observation'][:120]}...")
        history_text = "Recent history (last 5 steps):\n" + "\n".join(lines) + "\n\n"

    if not admissible_commands:
        print(
            Fore.YELLOW
            + "[WARN] llm_choose_action: empty admissible_commands; returning empty string."
        )
        return ""

    system_prompt = _build_system_prompt(
        task=task,
        observation=observation,
        admissible_commands=admissible_commands,
        score=score,
        done=done,
        step=step,
    )
    human_msg = (
        history_text
        + "Choose the next action. Reply with the exact admissible action text only."
    )

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_msg)]
    )
    raw = _strip_think(response.content.strip())

    # Exact match
    if raw in admissible_commands:
        return raw
    # Case-insensitive match
    raw_lower = raw.lower()
    for cmd in admissible_commands:
        if cmd.lower() == raw_lower:
            return cmd
    # Substring match
    for cmd in admissible_commands:
        if cmd.lower() in raw_lower or raw_lower in cmd.lower():
            return cmd
    # Fallback
    print(Fore.YELLOW + f"[WARN] LLM replied '{raw}' – not in admissible list; using first option.")
    return admissible_commands[0]

@mcp.tool()
async def step_env(action: str) -> str:
    """
    Execute an action in the ALFWorld environment.

    Args:
        action (str): One of the currently admissible action strings.

    Returns a JSON string with keys:
        observation          (str)   – Text feedback from the environment.
        score                (float) – Cumulative reward.
        done                 (bool)  – True if the episode has ended.
        admissible_commands  (list[str]) – Valid actions at the new state.
        step                 (int)   – Number of steps taken so far.
        action_taken         (str)   – The action that was executed.
    """
    global _current_obs, _current_score, _current_done, _admissible_commands
    global _step_count

    if _env is None:
        return json.dumps({"error": "Environment not initialised. Call reset_env first."})

    if _current_done:
        return json.dumps({
            "error": "Episode is already done. Call reset_env to start a new game.",
            "done": True,
        })

    obs, scores, dones, infos = _env.step([action])

    _current_obs = obs[0]
    _current_score = float(scores[0])
    _current_done = bool(dones[0])
    _admissible_commands = list(infos["admissible_commands"][0]) if not _current_done else []
    _step_count += 1

    result = {
        "observation": _current_obs,
        "score": _current_score,
        "done": _current_done,
        "admissible_commands": _admissible_commands,
        "step": _step_count,
        "action_taken": action,
    }
    return json.dumps(result)


@mcp.tool()
async def get_admissible_commands() -> str:
    """
    Return the list of currently valid action strings.

    Returns a JSON string with keys:
        admissible_commands (list[str])
        step                (int)
    """
    return json.dumps({
        "admissible_commands": _admissible_commands,
        "step": _step_count,
    })


@mcp.tool()
async def get_current_state() -> str:
    """
    Return a snapshot of the current game state.

    Returns a JSON string with keys:
        task                 (str)
        observation          (str)
        score                (float)
        done                 (bool)
        step                 (int)
        admissible_commands  (list[str])
    """
    return json.dumps({
        "task": _task_desc,
        "observation": _current_obs,
        "score": _current_score,
        "done": _current_done,
        "step": _step_count,
        "admissible_commands": _admissible_commands,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(
        mcp.run(
            transport="streamable-http",
            host="0.0.0.0",
            port=9000,
            path="/mcp",
            log_level="info",
        )
    )
