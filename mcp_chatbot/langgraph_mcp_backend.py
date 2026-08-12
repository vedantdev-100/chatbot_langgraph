from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
import aiosqlite
import requests
import asyncio
import threading
import json
import ormsgpack

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer, EMPTY_BYTES

load_dotenv()


def _patch_jsonplus_serializer_compat():
    if hasattr(JsonPlusSerializer, "dumps") and hasattr(JsonPlusSerializer, "loads"):
        return

    def dumps(self, obj):
        type_, data = self.dumps_typed(obj)
        if type_ in {"null", "bytes", "bytearray", "msgpack", "pickle"}:
            return data
        if type_ == "json":
            return data
        raise NotImplementedError(f"JsonPlusSerializer.dumps does not support type {type_}")

    def loads(self, data):
        if data is None or data == EMPTY_BYTES:
            return None
        try:
            return ormsgpack.unpackb(
                data,
                ext_hook=self._unpack_ext_hook,
                option=ormsgpack.OPT_NON_STR_KEYS,
            )
        except ormsgpack.MsgpackDecodeError:
            return json.loads(data.decode("utf-8"), object_hook=self._reviver)

    JsonPlusSerializer.dumps = dumps
    JsonPlusSerializer.loads = loads


_patch_jsonplus_serializer_compat()

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# -------------------
# 1. LLM
# -------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()


client = MultiServerMCPClient(
    {
        #laptop
        "arith": {
            "transport": "stdio",
            "command": r"C:\Users\iveda\OneDrive\Desktop\CHATBOT\venv\Scripts\python.exe",
            "args": [
                r"C:\Users\iveda\OneDrive\Desktop\mcp_server_local\main.py",
            ],
        },
        # pc
        # "arith": {
        #     "transport": "stdio",
        #     "command": r"C:\Users\Pro-3\Desktop\local_mcp_lgin\.venv\Scripts\python.exe",
        #     "args": [
        #         r"C:\Users\Pro-3\Desktop\local_mcp_lgin\main.py",
        #     ],
        # },
        "expense": {
            "transport": "streamable_http",  # if this fails, try "sse"
            "url": "https://provincial-plum-turtle.fastmcp.app/mcp",
            "headers": {
                "Authorization": f"Bearer {os.getenv('FASTMCP_API_KEY')}",
            },
        }
        }
)


def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []


mcp_tools = load_mcp_tools()

tools = [search_tool, get_stock_price, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------

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


async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer
# -------------------


async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_node("sanitize", sanitize_tool_messages)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "sanitize")
    graph.add_edge("sanitize", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())