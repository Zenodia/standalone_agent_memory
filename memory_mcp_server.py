from fastmcp import FastMCP
from dotenv import load_dotenv

from langchain_core.runnables import  RunnablePassthrough
import os

import nest_asyncio, asyncio
from utils import memory_ops_chain
 
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
    thread_id=0
    config = {"configurable": {"user_id": user_id, "thread_id": str(thread_id)}}
    output=""
    output= await memory_ops_chain.ainvoke(input={"input":query, "config":config})
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
        path='/mcp',
        log_level="debug",
    ))

