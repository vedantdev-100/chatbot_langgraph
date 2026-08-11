from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
import asyncio

load_dotenv()  # Load environment variables from .env file

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# MCP client for local FastMCP server
client = MultiServerMCPClient(
    {
        # "arith": {
        #     "transport": "stdio",
        #     # "command": "C:\\Users\\iveda\\AppData\\Local\\Programs\\Python\\Python311",          
        #     # "args": ["C:\\Users\\iveda\\OneDrive\\Desktop\\mcp_server_local\\main.py"],
        #     # "command": "python",          
        #     "command": r"C:\Users\iveda\OneDrive\Desktop\CHATBOT\venv\Scripts\python.exe",          
        #     "args": [ r"C:\Users\iveda\OneDrive\Desktop\mcp_server_local\main.py" ],
        # },
        "arith": {
            "transport": "stdio",
            "command": r"C:\Users\iveda\AppData\Local\Programs\Python\Python311\Scripts\uv.exe",          
            "args": [ "run", 
                     "--directory", 
                     r"C:\Users\iveda\OneDrive\Desktop\mcp_server_local", 
                     "python", 
                     "main.py" ],
        },
        # "expense": {
        #     "transport": "streamable_http",  # if this fails, try "sse"
        #     "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        # }
    }
)


# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def build_graph():

    tools = await client.get_tools()

    print(tools)

    llm_with_tools = llm.bind_tools(tools)

    # nodes
    async def chat_node(state: ChatState):

        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}

    tool_node = ToolNode(tools)
# defining graph and nodes
    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # defining graph connections
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()

    return chatbot

async def main():

    chatbot = await build_graph()

    # running the graph
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="Give me all my expenses for the month of Nov from 1 Nov to 30 Nov")]})

    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())