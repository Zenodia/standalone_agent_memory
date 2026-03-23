# standalone_agent_memory
Plug-and-playable standalone agentic memory with minimal code 

## features 
this is a minimalisitc code which support extraction of custom memory from user conversation with the following features.

- filter per user by user_id as identification , note we can also extend the filtering to other supplied info
- automatic summary when conversation turns exceed 3 turns in multi-turns conversations
- keep track of the user-assistant conversations in multi-turns
- giving a conversational memory, this standalone agentic memory module will automatically create memory items and archiv into a vectorstore which is InMemoryVectorStore from langchain 
- the memory item creation is using a strong instruction following LLM from build.nvidia.com , tested these 2 "nvidia/llama-3.1-nemotron-51b-instruct","meta/llama-3.1-405b-instruct"
- a runnable chain which operates on the user conversation automatically
- a custom mcp server which allow easy integration for any agentic framework 
- a custom mcp client script to test the connection and serve as sample snippet code for easier integration 
- incorporate streaming possbility ( not yet supported in the custom MCP server & client)

## environment build
make sure you have Python 3.12 , I am using python 3.12.11
#### using conda environment with Anaconda-Navigator
find appropriate python packages from the env.yml file , if you are using anaconda , you can simple create a new environment with the following command 
in your anaconda terminal , create the environment using the below command
``` 
conda env create --name <a_friendly_environment_name> -f env. yml 
```

then activate the environment using the below command 
```
conda activate <a_friendly_environment_name>
```
#### alternatively , you can do pip install -r requirements.txt 
```
pip install -r requirements.txt
```


### set environment variables 
```
export NVIDIA_API_KEY="your NVIDIA API KEY"
export stream="Yes"
export llm_model="meta/llama-3.1-405b-instruct"
export embed_model="nvidia/nv-embedqa-mistral-7b-v2"
```
or
create an environment file called .env 
```
NVIDIA_API_KEY="your NVIDIA API KEY"
stream="Yes"
llm_model="meta/llama-3.1-405b-instruct"
embed_model="nvidia/nv-embedqa-mistral-7b-v2"
```
and then do 
```
source .env
```

---

## ALFWorld Game (second implementation)

A text-based household task game where a server-side LLM decides every action.
See `alfworld-game/` for all related files.

### Install ALFWorld

ALFWorld is installed automatically via `requirements.txt` (text-only, no visual/THOR deps):

```bash
pip install alfworld==0.4.2
```

Or explicitly if you only need this part:

```bash
pip install alfworld         # latest
pip install alfworld==0.4.2  # pinned version used here
```

### Download the game data

Run the one-time download command after installation (downloads ~1 GB to `~/.cache/alfworld`):

```bash
alfworld-download
```

Then set the environment variable so ALFWorld can find the data:

```bash
export ALFWORLD_DATA=~/.cache/alfworld
```

Add it to your `.env` file to make it permanent:

```
ALFWORLD_DATA=~/.cache/alfworld
```

### Run the ALFWorld MCP server

```bash
cd alfworld-game
python alfworld_env_mcp_server.py
```

The server starts on `0.0.0.0:9000/mcp` and exposes tools:
`reset_env`, `step_env`, `get_current_state`, `get_admissible_commands`, `llm_choose_action`

### Play via the host client

```bash
cd alfworld-game
python host_client.py          # resume (or auto-reset if no game is active)
```

Pass `reset=True` in the script to start a fresh game.
Game history is written to `alfworld-game/game_history.md` automatically.

### Play directly without MCP

```bash
cd alfworld-game
python alfworld_env_query.py   # runs the LLM-driven loop directly, no server needed
```

---

## steps to run — pick ONE implementation

> **Each implementation has its own MCP server. Run only one at a time** — they both listen on port `9000` by default.

---

### Option A — Memory Agent (`memory-agent/`)

#### step 1 : start the Memory Agent MCP server

```bash
# Linux / macOS
python memory-agent/memory_mcp_server.py

# Windows
python .\memory-agent\memory_mcp_server.py
```

You should see something similar to the below:

```
existing NVIDIA_API_KEY in the environment  nvapi-K


╭─ FastMCP 2.0 ──────────────────────────────────────────────────────────────╮
│                                                                            │
│        _ __ ___ ______           __  __  _____________    ____    ____     │
│       _ __ ___ / ____/___ ______/ /_/  |/  / ____/ __ \  |___ \  / __ \    │
│      _ __ ___ / /_  / __ `/ ___/ __/ /|_/ / /   / /_/ /  ___/ / / / / /    │
│     _ __ ___ / __/ / /_/ (__  ) /_/ /  / / /___/ ____/  /  __/_/ /_/ /     │
│    _ __ ___ /_/    \__,_/____/\__/_/  /_/\____/_/      /_____(_)____/      │
│                                                                            │
│    🖥️  Server name:     MemoryMCPTools                                      │
│    📦 Transport:       Streamable-HTTP                                     │
│    🔗 Server URL:      http://0.0.0.0:9000/mcp                             │
│                                                                            │
╰────────────────────────────────────────────────────────────────────────────╯
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit)
```

#### step 2 : run the host client

```bash
# Linux / macOS
python memory-agent/host_client.py

# Windows
python .\memory-agent\host_client.py
```

You should see something similar to the below:

```
Tool: name='memory_agent' ...
Respond from memory enabled agent:
That's quite an interesting introduction, Babe the talking pig! I'm excited to meet
you and your feathered friend, Rob the chicken. What kind of adventures do you two
like to have on the farm?
```

---

### Option B — ALFWorld Game (`alfworld-game/`)

> **Prerequisites:** complete the [ALFWorld install + data download](#alfworld-game-second-implementation) steps above first.

#### step 1 : start the ALFWorld MCP server

```bash
# Linux / macOS
python alfworld-game/alfworld_env_mcp_server.py

# Windows
python .\alfworld-game\alfworld_env_mcp_server.py
```

The server scans and loads the game files on startup (may take ~10 seconds):

```
[alfworld_env_mcp_server] Environment initialised.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit)
```

#### step 2 : run the host client

```bash
# Linux / macOS
python alfworld-game/host_client.py

# Windows
python .\alfworld-game\host_client.py
```

The LLM will start choosing actions and the game log will be written to `alfworld-game/game_history.md`.

To play directly without the MCP server (single-process, no client needed):

```bash
python alfworld-game/alfworld_env_query.py
```




