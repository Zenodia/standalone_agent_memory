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
from langchain_core.prompts import ChatPromptTemplate
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
        self.retriever = self.recall_vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5},
        )

        self.runnable_parallel_1_routing_func = RunnableLambda(self.mem_routing_function)
        self.runnable_parallel_2_create_memory = RunnableLambda(self.create_memory_items)
        
        self.config=None
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are assistant with ability to memorize conversations from the user. You should always answer user query based on the following context:\n<Documents>\n{context}\n</Documents>. \
                    Be polite and helpful.",
                ),
                ("user", "{input}"),
            ]
        )
        
        self.retriever_chain = (
            {"context": self.recall_memory, "input": RunnablePassthrough()}
            | prompt
            | self.llm 
        )
        self.memory_ops_chain = RunnablePassthrough() | {  # this dict is coerced to a RunnableParallel
        "mem_ops": self.runnable_parallel_1_routing_func,
        "mem_items": self.runnable_parallel_2_create_memory,        
        } | self.execute_memory_operations

    async def mem_routing_function(self, inputs):
        query=inputs["input"]
        self.memory_manager.current_input=query
        self.config=inputs["config"]
        output=await self.memory_manager.memory_routing(query, self.config)    
        
        return output


    async def create_memory_items(self, inputs):
        query=inputs["input"]
        self.memory_manager.current_input=query
        self.config=inputs["config"]
        memory_items = await self.memory_manager.query_to_memory_items(query=query)
        docs = self.memory_manager.save_recall_memory(memory_items, config=self.config)
        print(Fore.CYAN + "creating memory items =", memory_items, Fore.RESET)
        return docs
    
    async def recall_memory(self, inputs):
        #print(Fore.MAGENTA + "recall memory inputs=\n", inputs, Fore.RESET)
        query=self.memory_manager.current_input
        self.memory_manager.current_input=query
        memory_items = self.memory_manager.search_recall_memories(query, config=self.config)
        print(Fore.MAGENTA + "recall memory items=\n", memory_items, Fore.RESET)
        return memory_items

    async def execute_memory_operations(self,inputs):
        mem_ops=inputs["mem_ops"]
        print(Fore.BLUE +"executing memory operation = ", mem_ops, Fore.RESET)
        query=self.memory_manager.current_input
        
        if 'search_memory' in mem_ops.lower():        
            output = await self.retriever_chain.ainvoke(query)
            output = output.content
            
        elif "no operation":
            output=self.llm.invoke(query).content 
        return output

    
