"""
Tests for the abstention / anti-hallucination pipeline.

Covers:
  1. check_groundedness() unit tests (pure function, no I/O).
  2. Gate 1 — empty retrieval → ABSTENTION_MESSAGE (no LLM call).
  3. Gate 2 — retrieved chunks present, but LLM response is ungrounded.
  4. Gate 2 — LLM emits the NATWEST_ABSTAIN sentinel → abstention.
  5. Happy path — grounded response passes through unchanged.
  6. Gap logger — correct reason codes are written.

Run with:
    .\\venv\\Scripts\\python -m pytest tests/test_abstention.py -v
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Groundedness check — pure unit tests (no I/O)
# ─────────────────────────────────────────────────────────────────────────────

from app.rag_pipeline import check_groundedness, ABSTENTION_MESSAGE, LLM_ABSTAIN_SENTINEL


SAMPLE_CHUNKS = [
    {
        "text": (
            "NatWest mortgage products require a minimum deposit of five percent "
            "for residential purchases. The standard variable rate is currently "
            "7.49 percent per annum. Fixed rate products are available for two, "
            "three, and five year terms."
        ),
        "document_name": "Mortgage Policy 2026",
        "page_number": 4,
        "score": 0.87,
        "effective_date": "2026-01-01",
        "doc_status": "Active",
    }
]


class TestGroundednessCheck:
    def test_grounded_response_passes(self):
        """Response that reuses chunk vocabulary is grounded."""
        response = (
            "NatWest mortgage products require a minimum deposit of five percent. "
            "Fixed rate products are available for two, three, and five year terms."
        )
        assert check_groundedness(response, SAMPLE_CHUNKS) is True

    def test_unrelated_response_is_ungrounded(self):
        """A response about completely different topics has zero chunk overlap."""
        response = (
            "The French Revolution began in 1789 when citizens stormed the Bastille. "
            "Napoleon Bonaparte later rose to prominence."
        )
        assert check_groundedness(response, SAMPLE_CHUNKS) is False

    def test_llm_sentinel_triggers_ungrounded(self):
        """Even a response with some overlapping words is ungrounded if it contains the sentinel."""
        response = (
            f"{LLM_ABSTAIN_SENTINEL} I don't have sufficient information "
            "in approved NatWest mortgage policy documents to answer this."
        )
        assert check_groundedness(response, SAMPLE_CHUNKS) is False

    def test_empty_chunks_returns_false(self):
        """If chunks are empty, groundedness check always returns False."""
        assert check_groundedness("Any response here", []) is False

    def test_empty_response_is_ungrounded(self):
        """An empty LLM response has no overlap with chunks."""
        assert check_groundedness("", SAMPLE_CHUNKS) is False

    def test_exact_chunk_text_is_grounded(self):
        """Response that exactly mirrors the chunk text is definitely grounded."""
        assert check_groundedness(SAMPLE_CHUNKS[0]["text"], SAMPLE_CHUNKS) is True

    def test_partial_overlap_below_threshold_is_ungrounded(self):
        """One or two overlapping words (e.g., 'the', short words) don't meet threshold."""
        # Only contains very short words that the tokenizer filters (len < 4)
        response = "Yes. No. The and of in."
        assert check_groundedness(response, SAMPLE_CHUNKS) is False


# ─────────────────────────────────────────────────────────────────────────────
# 2–6.  Pipeline integration tests (mocked I/O)
# ─────────────────────────────────────────────────────────────────────────────

async def _collect_sse(async_gen) -> list[dict]:
    """Helper: drain an async generator and parse every SSE data line."""
    events = []
    async for raw in async_gen:
        for line in raw.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def _make_pipeline():
    """Import RAGPipeline fresh so patches applied before import take effect."""
    from app.rag_pipeline import RAGPipeline
    return RAGPipeline()


# Shared mock for context_manager (used by all pipeline tests)
_CONTEXT_MANAGER_PATCH = {
    "app.rag_pipeline.context_manager.create_chat":       AsyncMock(return_value="test-chat-id"),
    "app.rag_pipeline.context_manager.get_chat":          AsyncMock(return_value=None),
    "app.rag_pipeline.context_manager.get_rolling_context": AsyncMock(return_value=""),
    "app.rag_pipeline.context_manager.update_context":    AsyncMock(),
}


@pytest.mark.asyncio
class TestAbstentionGate1:
    """Gate 1: retrieval returns empty list → hardcoded abstention, no LLM call."""

    async def test_empty_retrieval_yields_abstention(self):
        pipeline = _make_pipeline()

        async def _fake_stream(*a, **kw):
            return
            yield  # pragma: no cover

        with patch("app.rag_pipeline.retriever.search", new=AsyncMock(return_value=[])), \
             patch("app.rag_pipeline.llm_service.classify_intent",
                   new=AsyncMock(return_value="question about uploaded documents")), \
             patch("app.rag_pipeline.llm_service.stream_answer", side_effect=_fake_stream), \
             patch("app.rag_pipeline.log_retrieval_gap", new=AsyncMock()) as mock_gap_log, \
             patch("app.rag_pipeline.context_manager.create_chat",
                   new=AsyncMock(return_value="c1")), \
             patch("app.rag_pipeline.context_manager.get_chat",
                   new=AsyncMock(return_value=None)), \
             patch("app.rag_pipeline.context_manager.get_rolling_context",
                   new=AsyncMock(return_value="")), \
             patch("app.rag_pipeline.context_manager.update_context", new=AsyncMock()), \
             patch("app.rag_pipeline.Profiler"):

            events = await _collect_sse(pipeline.ask_stream(
                question="What is the mortgage deposit requirement?",
                user_role="Teller",
                user_department="Retail",
            ))

        chunk_events = [e for e in events if e.get("type") == "chunk"]
        full_text = "".join(e["content"] for e in chunk_events)

        # The abstention message must be present
        assert ABSTENTION_MESSAGE in full_text, (
            f"Expected abstention message in response, got: '{full_text}'"
        )

        # LLM must NOT have been called (no stream_answer invocation allowed)
        # We verify by checking the gap log was called with the correct reason
        mock_gap_log.assert_called_once()
        _, kwargs = mock_gap_log.call_args
        assert kwargs.get("reason") == "no_candidates" or mock_gap_log.call_args[0][3] == "no_candidates"

    async def test_empty_retrieval_does_not_call_llm(self):
        """LLM stream_answer must be completely bypassed on Gate 1 trigger."""
        pipeline = _make_pipeline()
        llm_was_called = False

        async def _should_not_be_called(*a, **kw):
            nonlocal llm_was_called
            llm_was_called = True
            return
            yield  # pragma: no cover

        with patch("app.rag_pipeline.retriever.search", new=AsyncMock(return_value=[])), \
             patch("app.rag_pipeline.llm_service.classify_intent",
                   new=AsyncMock(return_value="question about uploaded documents")), \
             patch("app.rag_pipeline.llm_service.stream_answer",
                   side_effect=_should_not_be_called), \
             patch("app.rag_pipeline.log_retrieval_gap", new=AsyncMock()), \
             patch("app.rag_pipeline.context_manager.create_chat",
                   new=AsyncMock(return_value="c2")), \
             patch("app.rag_pipeline.context_manager.get_chat",
                   new=AsyncMock(return_value=None)), \
             patch("app.rag_pipeline.context_manager.get_rolling_context",
                   new=AsyncMock(return_value="")), \
             patch("app.rag_pipeline.context_manager.update_context", new=AsyncMock()), \
             patch("app.rag_pipeline.Profiler"):

            async for _ in pipeline.ask_stream(
                question="What is the overdraft fee?",
                user_role="Teller",
                user_department="Retail",
            ):
                pass

        assert not llm_was_called, "LLM stream_answer was called despite empty retrieval (Gate 1 failure)."


@pytest.mark.asyncio
class TestAbstentionGate2:
    """Gate 2: chunks retrieved but LLM response is ungrounded."""

    async def test_ungrounded_llm_response_yields_abstention(self):
        """An off-topic LLM response is replaced by the hardcoded abstention message."""
        pipeline = _make_pipeline()

        # LLM returns something completely unrelated to the chunks
        async def _ungrounded_stream(*a, **kw):
            yield "The French Revolution began in 1789 when citizens stormed the Bastille."

        with patch("app.rag_pipeline.retriever.search",
                   new=AsyncMock(return_value=SAMPLE_CHUNKS)), \
             patch("app.rag_pipeline.llm_service.classify_intent",
                   new=AsyncMock(return_value="question about uploaded documents")), \
             patch("app.rag_pipeline.llm_service.stream_answer",
                   return_value=_ungrounded_stream()), \
             patch("app.rag_pipeline.llm_service.generate_summary_only",
                   new=AsyncMock(return_value="summary")), \
             patch("app.rag_pipeline.log_retrieval_gap", new=AsyncMock()) as mock_gap_log, \
             patch("app.rag_pipeline.context_manager.create_chat",
                   new=AsyncMock(return_value="c3")), \
             patch("app.rag_pipeline.context_manager.get_chat",
                   new=AsyncMock(return_value=None)), \
             patch("app.rag_pipeline.context_manager.get_rolling_context",
                   new=AsyncMock(return_value="")), \
             patch("app.rag_pipeline.context_manager.update_context", new=AsyncMock()), \
             patch("app.rag_pipeline.Profiler"):

            events = await _collect_sse(pipeline.ask_stream(
                question="What is the mortgage rate?",
                user_role="Manager",
                user_department="Lending",
            ))

        chunk_events = [e for e in events if e.get("type") == "chunk"]
        full_text = "".join(e["content"] for e in chunk_events)

        assert ABSTENTION_MESSAGE in full_text, (
            f"Expected abstention message but got: '{full_text}'"
        )
        mock_gap_log.assert_called_once()
        # Verify reason = "ungrounded"
        args = mock_gap_log.call_args
        reason = args[1].get("reason") or args[0][3]
        assert reason == "ungrounded"

    async def test_llm_sentinel_triggers_gate2(self):
        """LLM emitting NATWEST_ABSTAIN sentinel triggers Gate 2."""
        pipeline = _make_pipeline()

        sentinel_response = (
            f"{LLM_ABSTAIN_SENTINEL} I don't have sufficient information "
            "in the approved NatWest mortgage policy documents."
        )

        async def _sentinel_stream(*a, **kw):
            yield sentinel_response

        with patch("app.rag_pipeline.retriever.search",
                   new=AsyncMock(return_value=SAMPLE_CHUNKS)), \
             patch("app.rag_pipeline.llm_service.classify_intent",
                   new=AsyncMock(return_value="question about uploaded documents")), \
             patch("app.rag_pipeline.llm_service.stream_answer",
                   return_value=_sentinel_stream()), \
             patch("app.rag_pipeline.llm_service.generate_summary_only",
                   new=AsyncMock(return_value="summary")), \
             patch("app.rag_pipeline.log_retrieval_gap", new=AsyncMock()) as mock_gap_log, \
             patch("app.rag_pipeline.context_manager.create_chat",
                   new=AsyncMock(return_value="c4")), \
             patch("app.rag_pipeline.context_manager.get_chat",
                   new=AsyncMock(return_value=None)), \
             patch("app.rag_pipeline.context_manager.get_rolling_context",
                   new=AsyncMock(return_value="")), \
             patch("app.rag_pipeline.context_manager.update_context", new=AsyncMock()), \
             patch("app.rag_pipeline.Profiler"):

            events = await _collect_sse(pipeline.ask_stream(
                question="What is the mortgage rate?",
                user_role="Manager",
                user_department="Lending",
            ))

        chunk_events = [e for e in events if e.get("type") == "chunk"]
        full_text = "".join(e["content"] for e in chunk_events)

        # User sees the clean abstention message — NOT the raw sentinel string
        assert ABSTENTION_MESSAGE in full_text
        assert LLM_ABSTAIN_SENTINEL not in full_text, (
            "Raw sentinel string leaked to the client — it should be replaced by the clean message."
        )
        mock_gap_log.assert_called_once()

    async def test_grounded_response_passes_through(self):
        """A properly grounded response is NOT replaced by abstention."""
        pipeline = _make_pipeline()

        grounded_response = (
            "NatWest mortgage products require a minimum deposit of five percent "
            "for residential purchases. The standard variable rate is currently "
            "7.49 percent per annum. Fixed rate products are available."
            "[Source: Mortgage Policy 2026, Page: 4, Effective: 2026-01-01, Status: Active]"
        )

        async def _grounded_stream(*a, **kw):
            yield grounded_response

        with patch("app.rag_pipeline.retriever.search",
                   new=AsyncMock(return_value=SAMPLE_CHUNKS)), \
             patch("app.rag_pipeline.llm_service.classify_intent",
                   new=AsyncMock(return_value="question about uploaded documents")), \
             patch("app.rag_pipeline.llm_service.stream_answer",
                   return_value=_grounded_stream()), \
             patch("app.rag_pipeline.llm_service.generate_summary_only",
                   new=AsyncMock(return_value="summary")), \
             patch("app.rag_pipeline.log_retrieval_gap", new=AsyncMock()) as mock_gap_log, \
             patch("app.rag_pipeline.context_manager.create_chat",
                   new=AsyncMock(return_value="c5")), \
             patch("app.rag_pipeline.context_manager.get_chat",
                   new=AsyncMock(return_value=None)), \
             patch("app.rag_pipeline.context_manager.get_rolling_context",
                   new=AsyncMock(return_value="")), \
             patch("app.rag_pipeline.context_manager.update_context", new=AsyncMock()), \
             patch("app.rag_pipeline.Profiler"):

            events = await _collect_sse(pipeline.ask_stream(
                question="What is the mortgage deposit requirement?",
                user_role="Manager",
                user_department="Lending",
            ))

        chunk_events = [e for e in events if e.get("type") == "chunk"]
        full_text = "".join(e["content"] for e in chunk_events)

        # Grounded response MUST pass through — abstention message must NOT appear
        assert ABSTENTION_MESSAGE not in full_text, (
            "Abstention message incorrectly replaced a grounded response."
        )
        assert "mortgage" in full_text.lower()

        # Gap log must NOT have been called for a grounded response
        mock_gap_log.assert_not_called()


@pytest.mark.asyncio
class TestGapLogger:
    """Verify gap logger writes the correct reason codes."""

    async def test_no_candidates_reason_logged(self):
        pipeline = _make_pipeline()

        with patch("app.rag_pipeline.retriever.search",
                   new=AsyncMock(return_value=[])), \
             patch("app.rag_pipeline.llm_service.classify_intent",
                   new=AsyncMock(return_value="question about uploaded documents")), \
             patch("app.rag_pipeline.log_retrieval_gap", new=AsyncMock()) as mock_gap_log, \
             patch("app.rag_pipeline.context_manager.create_chat",
                   new=AsyncMock(return_value="c6")), \
             patch("app.rag_pipeline.context_manager.get_chat",
                   new=AsyncMock(return_value=None)), \
             patch("app.rag_pipeline.context_manager.get_rolling_context",
                   new=AsyncMock(return_value="")), \
             patch("app.rag_pipeline.context_manager.update_context", new=AsyncMock()), \
             patch("app.rag_pipeline.Profiler"):

            async for _ in pipeline.ask_stream(
                question="What is the overdraft limit?",
                user_role="Teller",
                user_department="Retail",
            ):
                pass

        mock_gap_log.assert_called_once()
        kwargs = mock_gap_log.call_args[1] if mock_gap_log.call_args[1] else {}
        args   = mock_gap_log.call_args[0] if mock_gap_log.call_args[0] else ()
        reason = kwargs.get("reason") or (args[3] if len(args) > 3 else None)
        assert reason == "no_candidates", f"Expected 'no_candidates', got '{reason}'"

    async def test_ungrounded_reason_logged(self):
        pipeline = _make_pipeline()

        async def _bad_stream(*a, **kw):
            yield "Paris is the capital of France and is famous for the Eiffel Tower."

        with patch("app.rag_pipeline.retriever.search",
                   new=AsyncMock(return_value=SAMPLE_CHUNKS)), \
             patch("app.rag_pipeline.llm_service.classify_intent",
                   new=AsyncMock(return_value="question about uploaded documents")), \
             patch("app.rag_pipeline.llm_service.stream_answer",
                   return_value=_bad_stream()), \
             patch("app.rag_pipeline.llm_service.generate_summary_only",
                   new=AsyncMock(return_value="s")), \
             patch("app.rag_pipeline.log_retrieval_gap", new=AsyncMock()) as mock_gap_log, \
             patch("app.rag_pipeline.context_manager.create_chat",
                   new=AsyncMock(return_value="c7")), \
             patch("app.rag_pipeline.context_manager.get_chat",
                   new=AsyncMock(return_value=None)), \
             patch("app.rag_pipeline.context_manager.get_rolling_context",
                   new=AsyncMock(return_value="")), \
             patch("app.rag_pipeline.context_manager.update_context", new=AsyncMock()), \
             patch("app.rag_pipeline.Profiler"):

            async for _ in pipeline.ask_stream(
                question="What is the mortgage rate?",
                user_role="Manager",
                user_department="Lending",
            ):
                pass

        mock_gap_log.assert_called_once()
        kwargs = mock_gap_log.call_args[1] if mock_gap_log.call_args[1] else {}
        args   = mock_gap_log.call_args[0] if mock_gap_log.call_args[0] else ()
        reason = kwargs.get("reason") or (args[3] if len(args) > 3 else None)
        assert reason == "ungrounded", f"Expected 'ungrounded', got '{reason}'"
