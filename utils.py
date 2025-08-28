
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
## loading memory class 
memory_manager=MemoryHandler(llm,embed)


thread_id=0
   

async def mem_routing_function(inputs):
    query=inputs["input"]
    config=inputs["config"]
    output=await memory_manager.memory_routing(query, config)
    inputs["mem_ops"]=output
    print(Fore.GREEN+"chosen_mem_ops=", output,'\n', Fore.RESET)
    return inputs

async def create_memory_items(inputs):
    query=inputs["input"]
    memory_items = await memory_manager.query_to_memory_items(query=query)
    inputs["memory_items"]=memory_items
    return inputs

async def execute_memory_operations(inputs):
    mem_ops=inputs["mem_ops"]  
    query=inputs["input"]
    if "search_memory" in mem_ops.lower():
        out= await memory_manager.memory_retriever_chain.ainvoke(query)
        output=out.content
        #print(Fore.CYAN+ "integrating response and recall memory items = \n ", output, Fore.RESET)
        memory_items_d= await create_memory_items(inputs)
        memory_items = memory_items_d["memory_items"]["facts"]
        assert type(memory_items)==list 
        #print(Fore.RED +">>>>>>>>>>>>>>>>>>>>> memory_items<<<<<<<<<<<<<<<<<<<<<< \n " , memory_items, Fore.RESET)
        memories, ids= memory_manager.save_recall_memory(memory_items, memory_manager.config)        
    else:        
        user_id=inputs["config"]["configurable"]["user_id"]
        memory_manager.user_id=user_id
        out= await memory_manager.memory_retriever_chain.ainvoke(query)
        output=out.content
        #print(Fore.YELLOW + "no memory operation needed continue to respond ", output, Fore.RESET)    
    return output

