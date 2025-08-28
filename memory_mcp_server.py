from fastmcp import FastMCP
from dotenv import load_dotenv

from langchain_core.runnables import  RunnablePassthrough
import os

import nest_asyncio, asyncio
from utils import mem_routing_function, execute_memory_operations

sequence = RunnablePassthrough()  | mem_routing_function  | execute_memory_operations

 
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
    #thread_id=increment_by_1(thread_id)
    config = {"configurable": {"user_id": user_id, "thread_id": str(thread_id)}}
    output=""
    output= await sequence.ainvoke(input={"input":"hi, my name is Babe, I am a pig and I can talk, my best friend is a chicken named Rob.", "config":config})
    output= output.replace("search_memory","")
    if '{' in output:
        index=output.index('{')
        output=output[:index]    
        print(output)
                     
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
        log_level="debug",
    ))

