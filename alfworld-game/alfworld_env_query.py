import argparse
import os
import re
from pathlib import Path

import yaml
from alfworld.agents.environment import get_environment
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "base_config.yaml"
print("DEFAULT CONFIG location : {}".format(_DEFAULT_CONFIG))


def load_alfworld_config():
    """Same behavior as alfworld.agents.modules.generic.load_config, but config_file is optional."""
    parser = argparse.ArgumentParser(
        description="ALFWorld TextWorld smoke loop. Set ALFWORLD_DATA and run alfworld-download first."
    )
    parser.add_argument(
        "config_file",
        nargs="?",
        default=str(_DEFAULT_CONFIG),
        help="path to ALFWorld yaml (default: ./configs/base_config.yaml next to this script)",
    )
    parser.add_argument(
        "-p",
        "--params",
        nargs="+",
        metavar="my.setting=value",
        default=[],
        help="override config entries, e.g. -p general.use_cuda=False",
    )
    args = parser.parse_args()
    if not os.path.isfile(args.config_file):
        raise SystemExit(
            f"Config not found: {args.config_file}\n"
            f"Pass a yaml path, or add {_DEFAULT_CONFIG} (e.g. copy from alfworld/configs/base_config.yaml)."
        )
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    for param in args.params:
        fqn_key, value = param.split("=", 1)
        entry_to_change = config
        keys = fqn_key.split(".")
        for k in keys[:-1]:
            entry_to_change = entry_to_change[k]
        entry_to_change[keys[-1]] = value
    return config


# ── LLM setup (same model as memory_mcp_server.py) ──────────────────────────
llm = ChatNVIDIA(model="nvidia/llama-3.3-nemotron-super-49b-v1.5")

SYSTEM_PROMPT = """\
You are an expert agent playing a text-based household task game (ALFWorld/TextWorld).
Your goal is to complete the given task as efficiently as possible.
You will be given the current observation and a numbered list of admissible actions.
Reply with ONLY the exact action text (copied verbatim from the list), nothing else.
Do not add any explanation, punctuation, or extra words."""


def llm_choose_action(task: str, observation: str, admissible_commands: list[str]) -> str:
    """Ask the LLM to pick the best next action from admissible_commands."""
    numbered = "\n".join(f"{i+1}. {cmd}" for i, cmd in enumerate(admissible_commands))
    human_msg = (
        f"Task: {task}\n\n"
        f"Current observation:\n{observation}\n\n"
        f"Admissible actions:\n{numbered}\n\n"
        "Which action should I take next? Reply with the exact action text only."
    )
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human_msg)])
    raw = response.content.strip()
    # Strip <think>...</think> tags (some reasoning models include them)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # If the LLM returned something that matches one of the admissible commands, use it.
    # Otherwise fall back to the closest match (case-insensitive).
    if raw in admissible_commands:
        return raw
    raw_lower = raw.lower()
    for cmd in admissible_commands:
        if cmd.lower() == raw_lower:
            return cmd
    # Partial match fallback: pick the command whose text is contained in the response
    for cmd in admissible_commands:
        if cmd.lower() in raw_lower:
            return cmd
    # Last resort: first admissible command
    print(f"[WARN] LLM output '{raw}' not in admissible commands; defaulting to first option.")
    return admissible_commands[0]


# ── ALFWorld setup ────────────────────────────────────────────────────────────
config = load_alfworld_config()
config["env"]["type"] = "AlfredTWEnv"
env_type = config["env"]["type"]

env = get_environment(env_type)(config, train_eval="train")
env = env.init_env(batch_size=1)

# interact
obs, info = env.reset()
print("###################### ALFWORLD ENVIRONMENT #################################")
print("Info: {}".format(info))
print("\n\n")

print("Observation: {}".format(obs[0]))

# Extract task description from the initial observation
task_desc = ""
for line in obs[0].splitlines():
    if line.strip().lower().startswith("your task is"):
        task_desc = line.strip()
        break

MAX_STEPS = 20
interaction_count = 0
while interaction_count < MAX_STEPS:
    admissible_commands = list(info["admissible_commands"])
    chosen_action = llm_choose_action(task_desc, obs[0], admissible_commands[0])

    # step
    obs, scores, dones, infos = env.step([chosen_action])
    info = infos
    interaction_count += 1
    print("Action: {}, Obs: {}".format(chosen_action, obs[0]))
    print("Observation: {}".format(obs[0]))
    print("Scores: {}".format(scores[0]))
    print("Dones: {}".format(dones[0]))
    print("########################################################")
    print("\n\n")

    if dones[0]:
        print("Task completed!" if scores[0] > 0 else "Game over.")
        break
    