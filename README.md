# standalone_agent_memory
plug-and-playable standalone agentic memory with minimal code 

## features 
this is a minimalisitc code which support extraction of custom memory from user conversation with the following features.

- giving a conversational memory, this standalone agentic memory module will automatically create memory items and archiv into a vectorstore which is InMemoryVectorStore from langchain 
- the memory item creation is using a strong instruction following LLM from build.nvidia.com , tested these 2 "nvidia/llama-3.1-nemotron-51b-instruct","meta/llama-3.1-405b-instruct"
- a runnable chain which operates on the user conversation automatically
- a custom mcp server which allow easy integration for any agentic framework 
- a custom mcp client script to test the connection and serve as sample snippet code for easier integration 


## environment build
find appropriate python packages from the env.yml file , if you are using anaconda , you can simple create a new environment with the following command 
in your anaconda terminal , create the environment using the below command
``` 
conda env create --name <a_friendly_environment_name> -f env. yml 
```

then activate the environment using the below command 
```
conda activate <a_friendly_environment_name>
```
### set environment variables 
```
export NVIDIA_API_KEY="your NVIDIA API KEY"
```
or
create an environment file called .env 
```
NVIDIA_API_KEY="your NVIDIA API KEY"
```
and then do 
```
source .env
```

## steps to run this minimal example

### step 1 : run the mcp server, due to the modification we will not be using fastmcp and directly spin up the server as 
    
note: I am using vscode on windows with anaconda environment build described above
```python 
python .\memory_mcp_server.py
```
you should see something similar to the below 

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
│                                                                            │
│                                                                            │
│    🖥️  Server name:     MemoryMCPTools                                      │
│    📦 Transport:       Streamable-HTTP                                     │
│    🔗 Server URL:      http://127.0.0.1:4200/mcp                           │
│                                                                            │
│    📚 Docs:            https://gofastmcp.com                               │
│    🚀 Deploy:          https://fastmcp.cloud                               │
│                                                                            │
│    🏎️  FastMCP version: 2.11.3                                              │
│    🤝 MCP version:     1.13.1                                              │
│                                                                            │
╰────────────────────────────────────────────────────────────────────────────╯
[09/08/25 16:06:22] INFO     Starting MCP server 'MemoryMCPTools' with transport 'streamable-http' on        server.py:1522                                                                                       
                             http://127.0.0.1:4200/mcp                                                                                                                                                            
C:\Users\zcharpy\AppData\Local\anaconda3\envs\py312\Lib\site-packages\websockets\legacy\__init__.py:6: DeprecationWarning: websockets.legacy is deprecated; see https://websockets.readthedocs.io/en/stable/howto/upgrade.html for upgrade instructions
  warnings.warn(  # deprecated in 14.0 - 2024-11-09
C:\Users\zcharpy\AppData\Local\anaconda3\envs\py312\Lib\site-packages\uvicorn\protocols\websockets\websockets_impl.py:16: DeprecationWarning: websockets.server.WebSocketServerProtocol is deprecated
  from websockets.server import WebSocketServerProtocol
INFO:     Started server process [37080]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:4200 (Press CTRL+C to quit)
```

### step 2 : test the client with a pre-made query

```python
python .\test_mcp_client.py
```

you should see something similar to the below 
```
Tool: name='memory_agent' title=None description="An Agent with memory enabled, can memorize the past conversation and respond accordingly.\nArgs:\n    query (str): The input user query\n    user_id (str): the current user's id\nReturns:\n    str: output response to the user " inputSchema={'properties': {'query': {'title': 'Query', 'type': 'string'}, 'user_id': {'title': 'User Id', 'type': 'string'}}, 'required': ['query', 'user_id'], 'type': 'object'} outputSchema={'properties': {'result': {'title': 'Result', 'type': 'string'}}, 'required': ['result'], 'title': '_WrappedResult', 'type': 'object', 'x-fastmcp-wrap-result': True} annotations=None meta={'_fastmcp': {'tags': []}}
Respond from memory enabled agent:
That's quite an interesting introduction, Babe the talking pig! I'm excited to meet you and your feathered friend, Rob the chicken. What kind of adventures do you two like to have on the farm?
```




