# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "agent-client-protocol>=0.11.1",
# ]
# ///
import asyncio
import os
from typing import Any

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.interfaces import Client


class SimpleClient(Client):
    async def request_permission(
        self, session_id, tool_call, options, **kwargs: Any
    ):
        # Auto-approve everything.
        return {"outcome": {"outcome": "approved"}}

    async def session_update(self, session_id, update, **kwargs):
        # Print streamed text updates if your agent emits them.
        print(update)


async def main():
    async with spawn_agent_process(
        SimpleClient(),
        "./target/release/isanagent",
        "acp",
        env=os.environ,
    ) as (conn, _proc):
        res = await conn.initialize(protocol_version=PROTOCOL_VERSION)
        print("initialized:", res)


        session = await conn.new_session(
            cwd=".",
            mcp_servers=[],
        )
        print("session:", session)

        response = await conn.prompt(
            session_id=session.session_id,
            prompt=[text_block("Tell me a joke.")],
        )

        print(response)


if __name__ == "__main__":
    asyncio.run(main())