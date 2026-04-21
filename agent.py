"""
AI Agent — uses Claude + MCP to interact with Pinecone and MySQL via mcp_server.py.

Architecture:
    User input
        ↓
    agent.py  (Anthropic SDK tool-use loop)
        ↓  stdio
    mcp_server.py
        ↓          ↓
    Pinecone    MySQL

Usage:
    python agent.py
"""
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack

import anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

SYSTEM_PROMPT = """You are a helpful AI assistant with access to two databases:

1. **Pinecone** (vector / semantic memory) — store_memory, search_memory, delete_memory, pinecone_stats
2. **MySQL** (relational / structured data) — mysql_query, mysql_execute, mysql_list_tables, mysql_describe_table

Guidelines:
- Use `store_memory` to persist any fact, note, or piece of knowledge the user wants saved.
- Use `search_memory` when the user asks to recall, find, or look up something by meaning.
- Use MySQL tools for structured queries: counts, filters, joins, inserts into specific tables.
- Prefer Pinecone for semantic/fuzzy retrieval; prefer MySQL for exact/structured retrieval.
- Always confirm what was stored, found, or changed.
- When uncertain which database to query, try both and synthesise the results.
"""

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "mcp_server.py")


class MCPAgent:
    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-6"
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.tools: list[dict] = []
        self.history: list[dict] = []

    # ── Connection ──────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Launch mcp_server.py as a subprocess and connect via stdio."""
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[SERVER_SCRIPT],
            env=None,
        )
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        await self.session.initialize()

        tools_resp = await self.session.list_tools()
        self.tools = [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            }
            for t in tools_resp.tools
        ]
        names = [t["name"] for t in self.tools]
        print(f"Connected to MCP server. Tools: {names}\n")

    async def close(self) -> None:
        await self.exit_stack.aclose()

    # ── Tool execution ───────────────────────────────────────────────────────

    async def _call_tool(self, name: str, tool_input: dict) -> str:
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
        result = await self.session.call_tool(name, tool_input)
        return result.content[0].text if result.content else ""

    # ── Agentic loop ─────────────────────────────────────────────────────────

    async def chat(self, user_message: str) -> str:
        """Send a user message and run the tool-use loop until a final reply."""
        self.history.append({"role": "user", "content": user_message})

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=self.tools,
                messages=self.history,
            )

            # Separate text blocks from tool-use blocks
            text_parts = [b.text for b in response.content if hasattr(b, "text")]
            tool_uses  = [b for b in response.content if b.type == "tool_use"]

            # No tool calls → final answer
            if response.stop_reason == "end_turn" or not tool_uses:
                self.history.append({"role": "assistant", "content": response.content})
                return " ".join(text_parts) if text_parts else "(no response)"

            # Append assistant turn (includes tool_use blocks)
            self.history.append({"role": "assistant", "content": response.content})

            # Execute every tool call and collect results
            tool_results = []
            for tu in tool_uses:
                preview = json.dumps(tu.input)[:100]
                print(f"  [tool] {tu.name}({preview}{'...' if len(json.dumps(tu.input)) > 100 else ''})")
                result_text = await self._call_tool(tu.name, tu.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result_text,
                    }
                )

            self.history.append({"role": "user", "content": tool_results})

    # ── Interactive REPL ─────────────────────────────────────────────────────

    async def run(self) -> None:
        print("=" * 55)
        print("  AI Agent  ·  Pinecone + MySQL  ·  powered by Claude")
        print("=" * 55)
        print("Commands: 'clear' → reset history  |  'quit' → exit\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                self.history.clear()
                print("Conversation history cleared.\n")
                continue

            print("Agent: ", end="", flush=True)
            reply = await self.chat(user_input)
            print(reply, "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    agent = MCPAgent()
    try:
        await agent.connect()
        await agent.run()
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
