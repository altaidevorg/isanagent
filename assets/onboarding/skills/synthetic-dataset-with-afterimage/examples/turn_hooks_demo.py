"""Demonstrate :class:`~afterimage.conversation_turn_hooks.ConversationTurnHooks` with :meth:`~afterimage.conversation_generator.ConversationGenerator.go`.

Hooks fire around each correspondent ``ask`` and respondent ``answer``. The same
generator also uses ``go()`` internally during :meth:`~afterimage.conversation_generator.ConversationGenerator.generate`,
so ``turn_hooks`` apply there as well.

Requires ``GEMINI_API_KEY``. Run from the repository root::

    uv run examples/turn_hooks_demo.py

Or::

    python examples/turn_hooks_demo.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from afterimage import (
    ConversationGenerator,
    ConversationTurnContext,
    ConversationTurnHooks,
    JSONLStorage,
)
from afterimage.types import ConversationEntry


def _short(text: str, limit: int = 100) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3] + "..."


class LoggingTurnHooks(ConversationTurnHooks):
    """Print each hook with context — replace with RAG injection, metrics, tracing, etc."""

    def _prefix(self, ctx: ConversationTurnContext) -> str:
        return (
            f"planned_turns={ctx.planned_turns} "
            f"assistant_msgs_so_far={ctx.respondent_turns_completed} "
            f"entries_in_ctx={len(ctx.conversation)}"
        )

    async def before_correspondent_completion(
        self, ctx: ConversationTurnContext, correspondent_input: str
    ) -> None:
        print(f"\n--- before_correspondent --- {self._prefix(ctx)}")
        print(f"    correspondent_input: {_short(correspondent_input)}")

    async def after_correspondent_completion(
        self, ctx: ConversationTurnContext, user_message: str
    ) -> None:
        print(f"--- after_correspondent --- {self._prefix(ctx)}")
        print(f"    user_message: {_short(user_message)}")

    async def before_respondent_completion(
        self, ctx: ConversationTurnContext, user_message: str
    ) -> None:
        print(f"--- before_respondent --- {self._prefix(ctx)}")
        print(f"    user_message: {_short(user_message)}")

    async def after_respondent_completion(
        self, ctx: ConversationTurnContext, entry: ConversationEntry
    ) -> None:
        print(f"--- after_respondent --- {self._prefix(ctx)}")
        content = entry.content or ""
        print(f"    assistant: {_short(content)}")


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY to run this example.")
        return

    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = JSONLStorage(conversations_path=out_dir / "turn_hooks_demo.jsonl")

    respondent_prompt = (
        "You are a helpful assistant for a small neighborhood bookstore. "
        "Answer briefly and warmly."
    )
    correspondent_prompt = (
        "You are a customer chatting with the store. "
        "Ask natural follow-up questions; output only what the customer would type, "
        "one message at a time, no role labels or meta."
    )

    hooks = LoggingTurnHooks()
    gen = ConversationGenerator(
        respondent_prompt=respondent_prompt,
        api_key=api_key,
        correspondent_prompt=correspondent_prompt,
        model_name="gemini-2.5-flash",
        storage=storage,
        turn_hooks=hooks,
    )

    print("Running go(turns=2) — watch hook order: correspondent → respondent per turn.\n")
    conversation = await gen.go(turns=2)

    print("\n=== final transcript ===")
    for i, entry in enumerate(conversation):
        role = entry.role.value
        print(f"{i + 1}. [{role}] {_short(entry.content or '', 200)}")


if __name__ == "__main__":
    asyncio.run(main())
