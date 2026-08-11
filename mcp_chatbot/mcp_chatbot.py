from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio
import os
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

### 1. LLM ###
llm = ChatGroq(
    # model="llama-3.3-70b-versatile",
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

### 2. State ###
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


### 3. MCP client — local stdio server + remote FastMCP Cloud server ###
mcp_client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": r"C:\Users\Pro-3\Desktop\local_mcp_lgin\.venv\Scripts\python.exe",
            "args": [
                r"C:\Users\Pro-3\Desktop\local_mcp_lgin\main.py",
            ],
        },
        "expense": {
            "transport": "streamable_http",  # if this fails, try "sse"
            "url": "https://provincial-plum-turtle.fastmcp.app/mcp",
            "headers": {
                "Authorization": f"Bearer {os.getenv('FASTMCP_API_KEY')}",
            },
        }
    }
)


### 4. Sanitize tool outputs — Groq rejects empty/malformed tool content ###
def sanitize_tool_messages(state: ChatState):
    """Ensure every ToolMessage has non-empty string content (Groq requires this)."""
    fixed = []
    changed = False
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            content = msg.content
            if not content or (isinstance(content, list) and len(content) == 0):
                msg = msg.model_copy(update={"content": "No result returned."})
                changed = True
            elif isinstance(content, list):
                text = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
                if not text.strip():
                    text = "No result returned."
                msg = msg.model_copy(update={"content": text})
                changed = True
        fixed.append(msg)
    return {"messages": fixed} if changed else {}


### 5. Build the graph ###
async def build_graph():
    tools = await mcp_client.get_tools()
    print(f"Loaded {len(tools)} MCP tools:", [t.name for t in tools])

    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)
    graph.add_node("sanitize", sanitize_tool_messages)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "sanitize")
    graph.add_edge("sanitize", "chat_node")

    chatbot = graph.compile()
    return chatbot


### 6. Run ###
async def main():
    chatbot = await build_graph()

    # result = await chatbot.ainvoke({"messages": [HumanMessage(content="Add an expense of rupees 9678 for pertol expense for vehicle")]})
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="Get list of all the expenses saved in the database")]})
    print("Messages: ", result['messages'])
    # print("Result: ", result)

    print("Messages: ", result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())