import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.tools import Tool
from colorama import Fore
async def main(query, user_id):
    client = Client(transport=StreamableHttpTransport("http://127.0.0.1:4200/mcp"))  # use /mcp path
    async with client:
        tools: list[Tool] = await client.list_tools()
        for tool in tools:
            print(f"Tool: {tool}")
        input= "hi, my name is Babe, I am a pig and I can talk, my best friend is a chicken named Rob." #"I had a fight with Rob, he ruined my birthday, he is no longer my best friend !"
        result = await client.call_tool(
            "memory_agent",
            {
                "query": query ,
                "user_id": user_id
            }
        )
    output=result.content[0].text # mcp response to text , which a list with TextContent in the list, access the text via attribute 
    ## example below 
    ### CallToolResult(content=[TextContent(type='text', text="That's quite an interesting introduction, Babe the talking pig! I'm excited to meet you and your feathered friend, Rob the chicken. What kind of adventures do you two like to have on the farm?", annotations=None, meta=None)], structured_content={'result': "That's quite an interesting introduction, Babe the talking pig! I'm excited to meet you and your feathered friend, Rob the chicken. What kind of adventures do you two like to have on the farm?"}, data="That's quite an interesting introduction, Babe the talking pig! I'm excited to meet you and your feathered friend, Rob the chicken. What kind of adventures do you two like to have on the farm?", is_error=False)
    
    print(Fore.CYAN + "inside mcp client , the respond from memory enabled agent:\n", output, Fore.RESET)
    return output
query=input("Enter your query:\n") 
user_id="user_1"
output = asyncio.run(main(query, user_id))
print("\n\n\n")
#print("output from main ", output)

