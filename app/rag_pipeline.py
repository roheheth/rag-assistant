"""
RAG Pipeline — the main orchestrator that ties everything together.

Flow for each question:
  1. Create or load chat
  2. Load rolling context from MongoDB (if follow-up)
  3. Intent routing (document question vs. conversation)
  4. Retrieve relevant chunks via hybrid search (RBAC + date filtered)
  5. ABSTENTION GATE #1: if retrieval returns nothing → hardcoded abstain
  6. Buffer full LLM response (groundedness check before any yield)
  7. ABSTENTION GATE #2: if response has no overlap with context → abstain
  8. Validate LLM inline [N] markers — drop any that reference a chunk
     index outside the range of what was actually passed to the prompt
  9. Build server-side Sources block from retrieved chunk metadata
 10. Yield answer token stream as SSE 'chunk' events
 11. Yield the verified Sources list as a distinct SSE 'sources' event
 12. Generate summary and update rolling context in MongoDB

Abstention events are always logged to the `retrieval_gaps` collection
so the compliance team can run periodic content-gap reports.

Citation contract:
  • LLM writes inline [N] markers only — no document names / dates / pages.
  • Server builds the Sources block exclusively from retrieved chunk metadata.
  • Any [N] marker referencing a non-existent chunk index is silently dropped
    (logged as WARNING for prompt-following quality monitoring).
  • The Sources block is transmitted as a separate structured SSE event
    (type='sources') AFTER the token stream completes, so the client can
    render it as a rich component rather than parsing free text.
"""

import re
import json
import logging
import os
import time
from typing import List, Optional

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

ABSTENTION_MESSAGE = (
    "I don't have sufficient information in approved sources to answer this. "
    "This may mean no policy document covers this topic, or you may not have "
    "clearance to view relevant documents."
)
LLM_ABSTAIN_SENTINEL = "NATWEST_ABSTAIN:"
GROUNDEDNESS_THRESHOLD = 0.10


# ── Groundedness check ────────────────────────────────────────────────────────

def _extract_tokens(text: str) -> set:
    """Extract unique lowercase alphabetic tokens of length >= 4."""
    return set(re.findall(r'\b[a-zA-Z]{4,}\b', text.lower()))


def check_groundedness(response: str, retrieved_chunks: list) -> bool:
    """
    Return True if the LLM response is sufficiently grounded.

    Fails (ungrounded) when:
      • The LLM emitted the NATWEST_ABSTAIN sentinel.
      • Lexical overlap between response and chunk text < GROUNDEDNESS_THRESHOLD.
    """
    if LLM_ABSTAIN_SENTINEL in response:
        logger.info("🔍 Groundedness: LLM emitted abstain sentinel → ungrounded")
        return False

    if not retrieved_chunks:
        return False

    chunk_text = " ".join(c.get("text", "") for c in retrieved_chunks)
    chunk_tokens = _extract_tokens(chunk_text)

    if not chunk_tokens:
        return True

    response_tokens = _extract_tokens(response)
    overlap = len(chunk_tokens & response_tokens) / len(chunk_tokens)

    is_grounded = overlap >= GROUNDEDNESS_THRESHOLD
    logger.info(
        f"🔍 Groundedness: overlap={overlap:.3f} "
        f"(threshold={GROUNDEDNESS_THRESHOLD}) → {'grounded' if is_grounded else 'UNGROUNDED'}"
    )
    return is_grounded


# ── Server-side citation helpers ──────────────────────────────────────────────

def validate_inline_markers(response: str, chunk_count: int) -> str:
    """
    Scan the LLM response for inline [N] markers and drop any whose index
    falls outside [1, chunk_count].

    A marker is valid if 1 <= N <= chunk_count.  Invalid markers are
    removed from the text and logged as warnings (they indicate that the
    model is hallucinating source references that don't exist).

    Args:
        response:    The raw LLM output text.
        chunk_count: Number of context chunks that were passed to the LLM.

    Returns:
        Cleaned response text with invalid markers stripped.
    """
    def _check_marker(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= chunk_count:
            return match.group(0)          # keep it
        logger.warning(
            f"⚠️ [Citation] LLM referenced [Source {n}] but only "
            f"{chunk_count} chunk(s) were passed — dropping marker."
        )
        return ""                          # silently drop it

    # Match [N] where N is a 1-3 digit integer
    cleaned = re.sub(r'\[(\d{1,3})\]', _check_marker, response)
    return cleaned


def build_sources_block(retrieved_chunks: List[dict]) -> List[dict]:
    """
    Build the authoritative Sources list **entirely** from the metadata of
    the chunks that were actually passed into the LLM prompt.

    This is the ONLY source of truth for citations — the LLM's text output
    is never parsed for document names, pages, dates, or status values.

    Returns a list of dicts, each with:
        index        — 1-based integer matching the [N] marker
        document_name
        page_number
        effective_date
        doc_status
        relevance_score
        snippet      — first 200 chars of chunk text for UI preview
    """
    sources = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        sources.append({
            "index":          i,
            "document_name":  chunk.get("document_name", "Unknown Document"),
            "page_number":    chunk.get("page_number"),
            "effective_date": chunk.get("effective_date", ""),
            "doc_status":     chunk.get("doc_status", "Active"),
            "relevance_score": round(chunk.get("score", 0.0), 4),
            "snippet":        chunk.get("text", "")[:200].rstrip() + "…"
                              if len(chunk.get("text", "")) > 200
                              else chunk.get("text", ""),
        })
    return sources


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

        SSE event types emitted (in order):
          metadata — {type, chat_id, turn_number}        (no sources here)
          chunk    — {type, content}  × N                (answer tokens)
          sources  — {type, sources: [...]}               (after stream done)
          done     — {type}                               (terminal sentinel)

        Two hard abstention gates protect against hallucination.
        Citations are built server-side and sent as the 'sources' event.
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
                _meta = json.dumps({"type": "metadata", "chat_id": chat_id, "turn_number": turn_count})
                yield f"data: {_meta}\n\n"
                _abstain = json.dumps({"type": "chunk", "content": ABSTENTION_MESSAGE})
                yield f"data: {_abstain}\n\n"
                _empty_sources = json.dumps({"type": "sources", "sources": []})
                yield f"data: {_empty_sources}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                await context_manager.update_context(
                    chat_id, question, ABSTENTION_MESSAGE, ABSTENTION_MESSAGE
                )
                return
        else:
            logger.info("Router: conversational intent — bypassing document search")
            retrieved_chunks = []

        # ── 5. Yield metadata (no sources yet — they come after the stream) ─
        _meta = json.dumps({"type": "metadata", "chat_id": chat_id, "turn_number": turn_count})
        yield f"data: {_meta}\n\n"

        # ── 6. Buffer full LLM response ─────────────────────────────────
        full_answer = ""
        first_token_time: Optional[float] = None

        async for token in llm_service.stream_answer(question, retrieved_chunks, previous_context):
            if first_token_time is None:
                first_token_time = time.time()
                logger.info(f"⏱️ TTFT: {first_token_time - start_time:.3f}s")
            full_answer += token

        # ── ABSTENTION GATE 2: Groundedness check ──────────────────────
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
            full_answer = ABSTENTION_MESSAGE

        # ── 7. Validate and clean LLM inline markers ────────────────────
        # Only run when we have context chunks to validate against.
        # Abstention responses pass through unchanged (they have no markers).
        if retrieved_chunks and full_answer != ABSTENTION_MESSAGE:
            full_answer = validate_inline_markers(full_answer, len(retrieved_chunks))

        # ── 8. Stream answer tokens to the client ──────────────────────
        CHUNK_SIZE = 80
        for i in range(0, len(full_answer), CHUNK_SIZE):
            _chunk = json.dumps({"type": "chunk", "content": full_answer[i:i + CHUNK_SIZE]})
            yield f"data: {_chunk}\n\n"

        # ── 9. Build and emit server-side Sources block ─────────────────
        # Sources come AFTER the token stream so the client can render
        # them as a rich structured component, not parse them from text.
        # Source metadata comes exclusively from retrieved_chunks —
        # nothing the LLM output said is used here.
        verified_sources = build_sources_block(retrieved_chunks)
        _sources_payload = json.dumps({"type": "sources", "sources": verified_sources})
        yield f"data: {_sources_payload}\n\n"

        # Terminal sentinel — lets the client know the stream is complete
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

        # ── 10. Summary + context update ────────────────────────────────
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
        logger.info(
            f"✓ Done | chat_id={chat_id} | sources={len(verified_sources)} | "
            f"total={total_time:.3f}s"
        )

        if USE_PYINSTRUMENT:
            profiler.stop()
            try:
                html = profiler.output_html()
                os.makedirs("profiles", exist_ok=True)
                with open("profiles/streaming_profile.html", "w", encoding="utf-8") as f:
                    f.write(html)
            except Exception:
                pass


rag_pipeline = RAGPipeline()
