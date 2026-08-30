"""
RAG Pipeline — the main orchestrator that ties everything together.

Flow for each question:
  1. Create or load chat
  2. Load rolling context from MongoDB (if follow-up)
  3. Intent routing (document question vs. conversation)
  4. Retrieve relevant chunks via hybrid search (RBAC + date filtered)
  5. ABSTENTION GATE #1: if retrieval returns nothing → hardcoded abstain
  6. Stream full answer from LLM (buffered for groundedness check)
  7. ABSTENTION GATE #2: if response has no overlap with context → abstain
  8. Yield answer tokens (or abstention message) to the SSE client
  9. Generate summary and update rolling context in MongoDB

Abstention events are always logged to the `retrieval_gaps` collection
so the compliance team can run periodic content-gap reports.
"""

import re
import json
import logging
import os
import time
from typing import Optional

try:
    from pyinstrument import Profiler
except ImportError:  # pragma: no cover
    Profiler = None  # type: ignore

from app.retriever import retriever
from app.llm_service import llm_service
from app.context_manager import context_manager
from app.gap_logger import log_retrieval_gap, REASON_NO_CANDIDATES, REASON_UNGROUNDED
from app.models import Source

logger = logging.getLogger(__name__)

# ── Abstention constants ──────────────────────────────────────────────────────

# The single canonical message shown to the user on any abstention.
# It is HARDCODED here — the LLM never generates or modifies this string.
ABSTENTION_MESSAGE = (
    "I don't have sufficient information in approved sources to answer this. "
    "This may mean no policy document covers this topic, or you may not have "
    "clearance to view relevant documents."
)

# Sentinel string the LLM is instructed to emit when it cannot ground an answer.
# We check for its presence in the buffered response as a secondary signal.
LLM_ABSTAIN_SENTINEL = "NATWEST_ABSTAIN:"

# Minimum fraction of unique 4+ character chunk tokens that must appear in the
# LLM response for it to be considered grounded. 0.10 = 10%.
GROUNDEDNESS_THRESHOLD = 0.10


# ── Groundedness check ────────────────────────────────────────────────────────

def _extract_tokens(text: str) -> set:
    """Extract unique lowercase alphabetic tokens of length >= 4."""
    return set(re.findall(r'\b[a-zA-Z]{4,}\b', text.lower()))


def check_groundedness(response: str, retrieved_chunks: list) -> bool:
    """
    Return True if the LLM response is sufficiently grounded in the
    retrieved context.

    Two signals are combined:
      1. Lexical overlap: fraction of unique chunk tokens that also appear
         in the response must meet GROUNDEDNESS_THRESHOLD.
      2. Sentinel check: if the LLM emitted NATWEST_ABSTAIN it is treating
         the context as insufficient — we honour that judgment.

    A response that fails either check is considered ungrounded.
    """
    # If the LLM itself flagged abstention, honour it immediately.
    if LLM_ABSTAIN_SENTINEL in response:
        logger.info("🔍 Groundedness: LLM emitted abstain sentinel → ungrounded")
        return False

    if not retrieved_chunks:
        return False

    chunk_text = " ".join(c.get("text", "") for c in retrieved_chunks)
    chunk_tokens = _extract_tokens(chunk_text)

    if not chunk_tokens:
        return True     # nothing to check against; give benefit of doubt

    response_tokens = _extract_tokens(response)
    overlap = len(chunk_tokens & response_tokens) / len(chunk_tokens)

    is_grounded = overlap >= GROUNDEDNESS_THRESHOLD
    logger.info(
        f"🔍 Groundedness: overlap={overlap:.3f} "
        f"(threshold={GROUNDEDNESS_THRESHOLD}) → {'grounded' if is_grounded else 'UNGROUNDED'}"
    )
    return is_grounded


# ── RAG Pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """Orchestrates the full RAG flow from question to answer."""

    async def ask_stream(
        self,
        question: str,
        chat_id: Optional[str] = None,
        user_role: str = "Admin",
        user_department: Optional[str] = None,
    ):
        """
        Process a user question and stream the response via SSE.

        Two hard abstention gates protect against hallucination:
          Gate 1 — No candidates returned after retrieval + RRF + threshold.
          Gate 2 — LLM response has insufficient overlap with retrieved context.

        Both gates log to `retrieval_gaps` for compliance review.
        """
        USE_PYINSTRUMENT = Profiler is not None
        start_time = time.time()

        if USE_PYINSTRUMENT:
            profiler = Profiler(async_mode="enabled")
            profiler.start()

        # ── 1. Handle chat ──────────────────────────────────────────────
        if chat_id:
            chat = await context_manager.get_chat(chat_id)
            if not chat:
                logger.warning(f"Chat {chat_id} not found, creating new")
                chat_id = await context_manager.create_chat()
                turn_count = 1
            else:
                turn_count = chat["turn_count"] + 1
        else:
            chat_id = await context_manager.create_chat()
            turn_count = 1

        # ── 2. Load rolling previous context ───────────────────────────
        previous_context = await context_manager.get_rolling_context(chat_id)

        # ── 3. Intent routing ───────────────────────────────────────────
        intent = await llm_service.classify_intent(question)

        # ── 4. Retrieve relevant chunks (document questions only) ───────
        if intent == "question about uploaded documents":
            retrieved_chunks = await retriever.search(
                question,
                user_role=user_role,
                user_department=user_department,
            )

            # ── ABSTENTION GATE 1: Empty retrieval ──────────────────────
            # This fires when:
            #   • No documents are indexed yet.
            #   • No chunk cleared the RBAC + date + similarity filters.
            #   • The topic is not covered by any ingested document.
            # The LLM is NEVER called in this path.
            if not retrieved_chunks:
                logger.warning(
                    f"🚫 [Abstention/Gate1] No candidates — role={user_role} "
                    f"dept={user_department} query='{question[:60]}'"
                )
                await log_retrieval_gap(
                    query_text=question,
                    user_role=user_role,
                    department=user_department or "Unknown",
                    reason=REASON_NO_CANDIDATES,
                )
                _no_candidate_meta = json.dumps({
                    "type": "metadata",
                    "chat_id": chat_id,
                    "sources": [],
                    "turn_number": turn_count,
                })
                yield f"data: {_no_candidate_meta}\n\n"
                _abstain_chunk = json.dumps({"type": "chunk", "content": ABSTENTION_MESSAGE})
                yield f"data: {_abstain_chunk}\n\n"
                await context_manager.update_context(
                    chat_id, question, ABSTENTION_MESSAGE, ABSTENTION_MESSAGE
                )
                return
        else:
            logger.info("Router: conversational intent — bypassing document search")
            retrieved_chunks = []

        # ── 5. Build source citations ───────────────────────────────────
        sources = [
            Source(
                text=chunk["text"][:300] + "…" if len(chunk["text"]) > 300 else chunk["text"],
                document_name=chunk["document_name"],
                page_number=chunk.get("page_number"),
                relevance_score=round(chunk["score"], 4),
                effective_date=chunk.get("effective_date", ""),
                doc_status=chunk.get("doc_status", "Active"),
            )
            for chunk in retrieved_chunks
        ]

        # ── 6. Yield metadata first ─────────────────────────────────────
        _meta_payload = json.dumps({
            "type":        "metadata",
            "chat_id":     chat_id,
            "sources":     [s.model_dump() for s in sources],
            "turn_number": turn_count,
        })
        yield f"data: {_meta_payload}\n\n"

        # ── 7. Buffer full LLM response (needed for groundedness check) ─
        # We collect all tokens before yielding any to the client so that
        # we can intercept an ungrounded response before it reaches the user.
        full_answer = ""
        first_token_time: Optional[float] = None

        async for token in llm_service.stream_answer(question, retrieved_chunks, previous_context):
            if first_token_time is None:
                first_token_time = time.time()
                logger.info(f"⏱️ TTFT: {first_token_time - start_time:.3f}s")
            full_answer += token

        # ── ABSTENTION GATE 2: Groundedness check ──────────────────────
        # Only applies when we had context to check against.
        if retrieved_chunks and not check_groundedness(full_answer, retrieved_chunks):
            logger.warning(
                f"🚫 [Abstention/Gate2] Ungrounded response — role={user_role} "
                f"dept={user_department} query='{question[:60]}'"
            )
            await log_retrieval_gap(
                query_text=question,
                user_role=user_role,
                department=user_department or "Unknown",
                reason=REASON_UNGROUNDED,
            )
            # Replace LLM output with the hardcoded abstention message.
            full_answer = ABSTENTION_MESSAGE

        # ── 8. Stream the (validated) answer to the client ─────────────
        # Yield in ~80-character chunks to preserve a token-stream feel.
        CHUNK_SIZE = 80
        for i in range(0, len(full_answer), CHUNK_SIZE):
            _chunk_payload = json.dumps({"type": "chunk", "content": full_answer[i:i + CHUNK_SIZE]})
            yield f"data: {_chunk_payload}\n\n"

        # ── 9. Summary + context update ─────────────────────────────────
        if not retrieved_chunks:
            summary_answer = full_answer
        else:
            summary_answer = await llm_service.generate_summary_only(full_answer)

        await context_manager.update_context(
            chat_id=chat_id,
            question=question,
            summary_answer=summary_answer,
            full_answer=full_answer,
        )

        total_time = time.time() - start_time
        logger.info(f"✓ Done | chat_id={chat_id} | total={total_time:.3f}s")

        if USE_PYINSTRUMENT:
            profiler.stop()
            try:
                html = profiler.output_html()
                os.makedirs("profiles", exist_ok=True)
                with open("profiles/streaming_profile.html", "w", encoding="utf-8") as f:
                    f.write(html)
            except Exception:
                pass  # Never let profiler I/O crash the pipeline


rag_pipeline = RAGPipeline()
