from fastmcp import FastMCP
from dotenv import load_dotenv
from langchain_core.runnables import  RunnablePassthrough
import os
import nest_asyncio, asyncio
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from utils import MemoryOps
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
import os
from colorama import Fore

llm= ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
embed = NVIDIAEmbeddings(model="nvidia/nv-embedqa-mistral-7b-v2",truncate="NONE",)


llm= ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
embed = NVIDIAEmbeddings(model="nvidia/nv-embedqa-mistral-7b-v2",truncate="NONE",)
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

memory_ops=MemoryOps(llm,embed,use_streaming )
mcp = FastMCP("MemoryMCPTools")

@mcp.tool()     
async def memory_agent(query:str, user_id:str ) -> str :
    """ An Agent with memory enabled, can memorize the past conversation and respond accordingly.
    Args:
        query (str): The input user query
        user_id (str): the current user's id
    Returns:
        str: output response to the user 
    """

    thread_id=0
    user_id="babe"
    config = {"configurable": {"user_id": user_id, "thread_id": str(thread_id)}}
    query = "hi, my name is Babe, I am a pig and I can talk, my best friend is a chicken named Rob."
    output = await memory_ops.memory_ops_chain.ainvoke(input={"input":query, "config":config})
    
    output= output.replace("search_memory","")
    print(Fore.YELLOW + "output from custom mcp server = \n", output)                    
    return output
"""
mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=4200,
        log_level="debug",
    )
"""
if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=4200,
        path='/mcp',
        log_level="debug",
    ))

