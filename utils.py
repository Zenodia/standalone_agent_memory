from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from  datetime import datetime
import ast
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
import os
import json
from typing import List, Literal, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import get_buffer_string
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
import uuid
import re
from colorama import Fore
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from MemoryManager import MemoryHandler
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

if os.getenv("stream") :
    stream_flag = os.getenv("stream")
    if stream_flag.lower()=="yes":
        use_streaming = True
    else:
        use_streaming = False
    print("using streaming : ", os.getenv("stream"))
else:
    use_streaming = False
    print("no environment variable set")


llm = ChatNVIDIA(model=llm_model)
embed = NVIDIAEmbeddings(model=embed_model,truncate="NONE",)



## loading memory class 
class MemoryOps:
    """
    Implementing Memory Handler into langchain runnable chain for simplicity
    """  
    def __init__(self, llm: ChatNVIDIA , embed: NVIDIAEmbeddings, use_streaming : bool):
        """
            Initialize the Memory Handler to handle agentic memory operations
        """        
        self.llm = llm
        self.embed=embed
        self.use_streaming = use_streaming
        self.memory_manager=MemoryHandler(llm,embed,use_streaming )
        self.recall_vector_store = InMemoryVectorStore(self.embed)
        self.retriever = self.recall_vector_store.as_retriever()
        self.runnable_parallel_1 = RunnableLambda(self.mem_routing_function)
        self.runnable_parallel_2 = RunnableLambda(self.create_memory_items)
        
        self.memory_ops_chain = RunnablePassthrough() | {  # this dict is coerced to a RunnableParallel
        "mem_ops": self.runnable_parallel_1,
        "mem_items": self.runnable_parallel_2
        } | self.execute_memory_operations

    async def mem_routing_function(self, inputs):
        query=inputs["input"]
        self.memory_manager.current_input=query
        config=inputs["config"]
        output=await self.memory_manager.memory_routing(query, config)    
        return output


    async def create_memory_items(self, inputs):
        query=inputs["input"]
        self.memory_manager.current_input=query
        memory_items = await self.memory_manager.query_to_memory_items(query=query)
        return memory_items

    async def execute_memory_operations(self,inputs):
        mem_ops=inputs["mem_ops"]
        query=self.memory_manager.current_input
        memory_items_for_saving=inputs["mem_items"]["facts"]
        if 'save_memory' in mem_ops.lower():        
            memories, ids= self.memory_manager.save_recall_memory(memory_items_for_saving, self.memory_manager.config)
            output = ids
        elif "update_memory" in mem_ops.lower():
            print("not implemented error")
            memories, ids = self.memory_manager.save_recall_memory(memory_items_for_saving, self.memory_manager.config)
            output = ids
        elif "no operation":
            output=llm.invoke(query).content 
        return output

    
