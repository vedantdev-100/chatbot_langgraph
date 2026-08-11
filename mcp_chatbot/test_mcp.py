import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": r"C:\Users\iveda\AppData\Local\Programs\Python\Python311\Scripts\uv.exe",
            "args": [
                "run",
                "--directory",
                r"C:\Users\iveda\OneDrive\Desktop\CHATBOT\mcp_chatbot\test_mcp.py",
                "python",
                "main.py",
            ],
        }
    }
)


async def main():

    print("Connecting to MCP server...")

    tools = await client.get_tools()

    print("Connected successfully!")

    for tool in tools:
        print(f"Tool: {tool.name}")


if __name__ == "__main__":
    asyncio.run(main())
