"""
Conversation context management with rolling summaries.

This is the core cost-optimization mechanism:
- After each Q&A turn, the summary answer is appended to a rolling context
- When context exceeds a character threshold, it is compressed via the LLM
- Follow-up questions use this compact summary (~200 tokens) instead of
  full conversation history (~2000+ tokens), saving 80-90% in prompt tokens

MongoDB schema for chats:
{
    "chat_id": "uuid",
    "rolling_context": "compressed context string",
    "turn_count": int,
    "total_tokens_saved": int,
    "full_history_tokens": int,  # what full history would have cost
    "created_at": "ISO timestamp",
    "updated_at": "ISO timestamp"
}
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, Dict
from app.database import get_db
from app.llm_service import llm_service
import logging

logger = logging.getLogger(__name__)

# Compress the rolling summary when it exceeds this character count
CONTEXT_COMPRESSION_THRESHOLD = 1500


class ContextManager:
    """CRUD operations and compression for chat context."""

    async def create_chat(self) -> str:
        """Create a new chat and return its UUID."""
        db = get_db()
        chat_id = str(uuid.uuid4())

        record = {
            "chat_id": chat_id,
            "rolling_context": "",
            "turn_count": 0,
            "total_tokens_saved": 0,
            "full_history_tokens": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        await db.chats.insert_one(record)
        logger.info(f"Created chat {chat_id[:8]}...")
        return chat_id

    async def get_chat(self, chat_id: str) -> Optional[Dict]:
        """Retrieve a chat record by ID."""
        db = get_db()
        return await db.chats.find_one(
            {"chat_id": chat_id}, {"_id": 0}
        )

    async def get_rolling_context(
        self, chat_id: str
    ) -> Optional[str]:
        """
        Get the rolling context for a chat.

        Returns None if the chat doesn't exist or has no context yet.
        """
        chat = await self.get_chat(chat_id)
        if chat and chat.get("rolling_context"):
            return chat["rolling_context"]
        return None

    async def update_context(
        self,
        chat_id: str,
        question: str,
        summary_answer: str,
        full_answer: str,
    ) -> int:
        """
        Append the new Q&A turn to the rolling context and compress if needed.

        The key insight: we store only Q + summary_answer (not full_answer)
        in the rolling context, dramatically reducing tokens for follow-ups.

        Args:
            chat_id: The chat to update.
            question: The user's question.
            summary_answer: The concise summary (stored in context).
            full_answer: The full answer (used only for token savings calc).

        Returns:
            Estimated tokens saved by using summary context vs full history.
        """
        db = get_db()
        chat = await self.get_chat(chat_id)

        if not chat:
            raise ValueError(f"Chat {chat_id} not found")

        # Build new context entry using summary (not full answer)
        new_entry = f"Q: {question}\nA: {summary_answer}"

        # Append to existing rolling context
        current_context = chat.get("rolling_context", "")
        if current_context:
            updated_context = f"{current_context}\n\n{new_entry}"
        else:
            updated_context = new_entry

        # Compress if the rolling context is getting too long
        if len(updated_context) > CONTEXT_COMPRESSION_THRESHOLD:
            logger.info(
                f"Context exceeded {CONTEXT_COMPRESSION_THRESHOLD} chars, "
                f"compressing..."
            )
            updated_context = await llm_service.compress_context(
                updated_context
            )

        # ── Calculate token savings ────────────────────────────────
        # Track what full history would have cost vs what summary costs
        full_history_tokens = chat.get("full_history_tokens", 0)
        new_full_history_tokens = full_history_tokens + (
            llm_service.estimate_tokens(f"Q: {question}\nA: {full_answer}")
        )
        context_tokens = llm_service.estimate_tokens(updated_context)
        tokens_saved = max(0, new_full_history_tokens - context_tokens)

        # ── Persist to MongoDB ─────────────────────────────────────
        await db.chats.update_one(
            {"chat_id": chat_id},
            {
                "$set": {
                    "rolling_context": updated_context,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "full_history_tokens": new_full_history_tokens,
                    "total_tokens_saved": tokens_saved,
                },
                "$inc": {"turn_count": 1},
            },
        )

        turn = chat["turn_count"] + 1
        logger.info(
            f"Context updated: chat={chat_id[:8]}... "
            f"turn={turn} tokens_saved={tokens_saved}"
        )

        return tokens_saved


context_manager = ContextManager()
