
import json
from typing import List, Literal, Optional
import tiktoken
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
import uuid
from MemoryManager import MemoryHandler
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from colorama import Fore
import os
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from colorama import Fore
import random
from dotenv import load_dotenv
import nest_asyncio, asyncio
nest_asyncio.apply()



choices=["nvidia/llama-3.1-nemotron-51b-instruct","meta/llama-3.1-405b-instruct"]

if os.environ["NVIDIA_API_KEY"]:
   print("existing NVIDIA_API_KEY in the environment ", os.environ["NVIDIA_API_KEY"][:7])
else:
    load_dotenv()

if os.environ.get('llm_model') == None:
    llm_model=random.choice(choices)
    
if os.environ.get("embed_model")==None:
    embed_model="nvidia/nv-embedqa-mistral-7b-v2"


llm = ChatNVIDIA(model=llm_model)
embed = NVIDIAEmbeddings(model=embed_model,truncate="NONE",)
use_streaming = False
global memory_manager
memory_manager=MemoryHandler(llm,embed,use_streaming )
## loading memory class 

async def mem_routing_function(inputs):
    query=inputs["input"]
    memory_manager.current_input=query
    config=inputs["config"]
    output=await memory_manager.memory_routing(query, config)    
    return output


async def create_memory_items(inputs):
    query=inputs["input"]
    memory_manager.current_input=query
    memory_items = await memory_manager.query_to_memory_items(query=query)
    return memory_items

runnable_parallel_1 = RunnableLambda(mem_routing_function)
runnable_parallel_2 = RunnableLambda(create_memory_items)
    

async def execute_memory_operations(inputs):
    mem_ops=inputs["mem_ops"]
    query=memory_manager.current_input
    memory_items_for_saving=inputs["mem_items"]["facts"]
    if 'save_memory' in mem_ops.lower():        
        memories, ids= memory_manager.save_recall_memory(memory_items_for_saving, memory_manager.config)
        output = ids
    elif "update_memory" in mem_ops.lower():
        print("not implemented error")
        memories, ids = memory_manager.save_recall_memory(memory_items_for_saving, memory_manager.config)
        output = ids
    elif "no operation":
        output=llm.invoke(query).content 
    return output

memory_ops_chain = RunnablePassthrough() | {  # this dict is coerced to a RunnableParallel
    "mem_ops": runnable_parallel_1,
    "mem_items": runnable_parallel_2
    } | execute_memory_operations

