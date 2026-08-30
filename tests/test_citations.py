"""
Tests for server-side citation correctness.

Covers:
  1. validate_inline_markers() — valid markers kept, out-of-range dropped.
  2. build_sources_block() — output matches retrieved chunk metadata exactly,
     independent of any LLM text output.
  3. Pipeline integration — 'sources' SSE event is emitted after chunks,
     and its content exactly matches retrieved chunk metadata (not LLM text).
  4. Out-of-range marker warning is logged.
  5. Conversational responses (no chunks) emit an empty sources list.

Run with:
    .\\venv\\Scripts\\python -m pytest tests/test_citations.py -v
"""

import json
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.rag_pipeline import (
    validate_inline_markers,
    build_sources_block,
    check_groundedness,
    ABSTENTION_MESSAGE,
)

# ── Sample chunk fixtures ─────────────────────────────────────────────────────

CHUNK_A = {
    "text": (
        "NatWest mortgage products require a minimum deposit of five percent "
        "for residential purchases. Fixed rate terms available for two to five years."
    ),
    "document_name": "Mortgage Policy 2026",
    "page_number": 4,
    "score": 0.87,
    "effective_date": "2026-01-01",
    "doc_status": "Active",
}

CHUNK_B = {
    "text": (
        "Personal loan interest rates range from 6.9% to 24.9% APR depending "
        "on credit profile and loan term. Maximum term is 7 years."
    ),
    "document_name": "Personal Lending Guide Q1 2026",
    "page_number": 12,
    "score": 0.74,
    "effective_date": "2026-03-01",
    "doc_status": "Active",
}

CHUNK_C = {
    "text": "Overdraft fees are capped at £8 per month under Consumer Duty guidelines.",
    "document_name": "Fee Schedule 2025",
    "page_number": 2,
    "score": 0.61,
    "effective_date": "2025-09-01",
    "doc_status": "Superseded",
}

TWO_CHUNKS  = [CHUNK_A, CHUNK_B]
THREE_CHUNKS = [CHUNK_A, CHUNK_B, CHUNK_C]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  validate_inline_markers() — pure unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateInlineMarkers:
    def test_valid_markers_are_kept(self):
        """[1] and [2] with 2 chunks — both should survive."""
        text = "The deposit is five percent [1]. Rates vary by term [2]."
        result = validate_inline_markers(text, chunk_count=2)
        assert "[1]" in result
        assert "[2]" in result

    def test_out_of_range_marker_is_dropped(self):
        """[3] with only 2 chunks → dropped."""
        text = "The deposit is five percent [1][3]."
        result = validate_inline_markers(text, chunk_count=2)
        assert "[1]" in result
        assert "[3]" not in result

    def test_zero_index_marker_is_dropped(self):
        """[0] is never valid (1-based indexing)."""
        text = "Some claim [0]."
        result = validate_inline_markers(text, chunk_count=3)
        assert "[0]" not in result

    def test_multiple_out_of_range_markers_all_dropped(self):
        """[4][5][6] with 3 chunks → all dropped."""
        text = "Wrong [4][5][6] references."
        result = validate_inline_markers(text, chunk_count=3)
        assert "[4]" not in result
        assert "[5]" not in result
        assert "[6]" not in result

    def test_text_unchanged_when_no_markers(self):
        """Response with no [N] markers passes through untouched."""
        text = "This is an answer with no inline citations."
        result = validate_inline_markers(text, chunk_count=3)
        assert result == text

    def test_all_valid_when_chunk_count_matches(self):
        """[1][2][3] with 3 chunks — all valid."""
        text = "A [1], B [2], C [3]."
        result = validate_inline_markers(text, chunk_count=3)
        assert result == text

    def test_warning_logged_for_invalid_marker(self, caplog):
        """Dropping an out-of-range marker must emit a WARNING log."""
        text = "Claim [5] is hallucinated."
        with caplog.at_level(logging.WARNING, logger="app.rag_pipeline"):
            validate_inline_markers(text, chunk_count=2)
        assert any("[Source 5]" in r.message for r in caplog.records), (
            "Expected a WARNING log mentioning [Source 5]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  build_sources_block() — metadata fidelity tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSourcesBlock:
    def test_index_is_one_based(self):
        """First source must have index=1."""
        sources = build_sources_block([CHUNK_A])
        assert sources[0]["index"] == 1

    def test_document_name_matches_chunk(self):
        """document_name must come from the chunk, not the LLM."""
        sources = build_sources_block([CHUNK_A, CHUNK_B])
        assert sources[0]["document_name"] == "Mortgage Policy 2026"
        assert sources[1]["document_name"] == "Personal Lending Guide Q1 2026"

    def test_page_number_matches_chunk(self):
        sources = build_sources_block(TWO_CHUNKS)
        assert sources[0]["page_number"] == 4
        assert sources[1]["page_number"] == 12

    def test_effective_date_matches_chunk(self):
        sources = build_sources_block(TWO_CHUNKS)
        assert sources[0]["effective_date"] == "2026-01-01"
        assert sources[1]["effective_date"] == "2026-03-01"

    def test_doc_status_matches_chunk(self):
        sources = build_sources_block(THREE_CHUNKS)
        assert sources[2]["doc_status"] == "Superseded"   # CHUNK_C

    def test_relevance_score_matches_chunk(self):
        sources = build_sources_block([CHUNK_A])
        assert sources[0]["relevance_score"] == round(CHUNK_A["score"], 4)

    def test_snippet_is_truncated_to_200_chars(self):
        """Text longer than 200 chars must be trimmed with an ellipsis."""
        long_chunk = dict(CHUNK_A, text="X" * 500)
        sources = build_sources_block([long_chunk])
        assert len(sources[0]["snippet"]) <= 204   # 200 + "…"
        assert sources[0]["snippet"].endswith("…")

    def test_short_text_not_truncated(self):
        """Text under 200 chars must not be modified."""
        short_chunk = dict(CHUNK_A, text="Short text.")
        sources = build_sources_block([short_chunk])
        assert sources[0]["snippet"] == "Short text."

    def test_empty_chunks_returns_empty_list(self):
        assert build_sources_block([]) == []

    def test_three_chunks_all_indexed_correctly(self):
        sources = build_sources_block(THREE_CHUNKS)
        assert [s["index"] for s in sources] == [1, 2, 3]

    def test_source_metadata_independent_of_llm_output(self):
        """
        Core assertion: build_sources_block() output must exactly match the
        retrieved chunk metadata regardless of what any LLM text output said.
        Even a completely fabricated LLM response doesn't affect citations.
        """
        fake_llm_output = (
            "The deposit is actually 50% [Source: Fake Document, Page: 999, "
            "Effective: 2000-01-01, Status: Fake]."
        )
        # Sources are built from chunks — LLM text is not an input
        sources = build_sources_block([CHUNK_A])
        assert sources[0]["document_name"] == "Mortgage Policy 2026"
        assert sources[0]["page_number"]   == 4
        assert sources[0]["effective_date"] == "2026-01-01"
        assert sources[0]["doc_status"]     == "Active"
        # LLM fabrications must NOT appear in sources
        assert "Fake Document" not in str(sources)
        assert "999" not in str(sources)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Pipeline integration — SSE event ordering and source fidelity
# ─────────────────────────────────────────────────────────────────────────────

async def _collect_events(async_gen) -> list[dict]:
    """Drain an async generator and parse every SSE data line."""
    events = []
    async for raw in async_gen:
        for line in raw.split("\n"):
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except Exception:
                    pass
    return events


def _make_pipeline():
    from app.rag_pipeline import RAGPipeline
    return RAGPipeline()


@pytest.mark.asyncio
class TestPipelineSourcesEvent:
    async def test_sources_event_emitted_after_chunks(self):
        """
        'sources' event must appear AFTER all 'chunk' events in the SSE stream.
        """
        pipeline = _make_pipeline()

        async def _grounded_stream(*a, **kw):
            yield (
                "NatWest mortgage products require a minimum deposit [1]. "
                "Personal loan rates vary [2]."
            )

        with patch("app.rag_pipeline.retriever.search",
                   new=AsyncMock(return_value=TWO_CHUNKS)), \
             patch("app.rag_pipeline.llm_service.classify_intent",
                   new=AsyncMock(return_value="question about uploaded documents")), \
             patch("app.rag_pipeline.llm_service.stream_answer",
                   return_value=_grounded_stream()), \
             patch("app.rag_pipeline.llm_service.generate_summary_only",
                   new=AsyncMock(return_value="Summary")), \
             patch("app.rag_pipeline.log_retrieval_gap", new=AsyncMock()), \
             patch("app.rag_pipeline.context_manager.create_chat",
                   new=AsyncMock(return_value="chat-cite-1")), \
             patch("app.rag_pipeline.context_manager.get_chat",
                   new=AsyncMock(return_value=None)), \
             patch("app.rag_pipeline.context_manager.get_rolling_context",
                   new=AsyncMock(return_value="")), \
             patch("app.rag_pipeline.context_manager.update_context",
                   new=AsyncMock()), \
             patch("app.rag_pipeline.Profiler"):

            events = await _collect_events(pipeline.ask_stream(
                question="What are the mortgage requirements?",
                user_role="Manager",
                user_department="Lending",
            ))

        types = [e["type"] for e in events]

        # sources must appear after all chunks
        last_chunk_idx  = max((i for i, t in enumerate(types) if t == "chunk"), default=-1)
        sources_idx     = next((i for i, t in enumerate(types) if t == "sources"), -1)
        assert sources_idx > last_chunk_idx, (
            f"'sources' event (idx={sources_idx}) must come after last 'chunk' "
            f"event (idx={last_chunk_idx})"
        )

    async def test_sources_event_metadata_matches_retrieved_chunks(self):
        """
        CORE TEST: The 'sources' event must contain metadata that exactly matches
        the retrieved chunks passed into the pipeline, regardless of LLM output.
        """
        pipeline = _make_pipeline()

        # LLM hallucinates a completely different document name in its text
        async def _hallucinating_stream(*a, **kw):
            yield (
                "According to the NatWest deposit policy [1], a five percent "
                "minimum applies. See also the Personal Lending Guide [2]."
            )

        with patch("app.rag_pipeline.retriever.search",
                   new=AsyncMock(return_value=TWO_CHUNKS)), \
             patch("app.rag_pipeline.llm_service.classify_intent",
                   new=AsyncMock(return_value="question about uploaded documents")), \
             patch("app.rag_pipeline.llm_service.stream_answer",
                   return_value=_hallucinating_stream()), \
             patch("app.rag_pipeline.llm_service.generate_summary_only",
                   new=AsyncMock(return_value="Summary")), \
             patch("app.rag_pipeline.log_retrieval_gap", new=AsyncMock()), \
             patch("app.rag_pipeline.context_manager.create_chat",
                   new=AsyncMock(return_value="chat-cite-2")), \
             patch("app.rag_pipeline.context_manager.get_chat",
                   new=AsyncMock(return_value=None)), \
             patch("app.rag_pipeline.context_manager.get_rolling_context",
                   new=AsyncMock(return_value="")), \
             patch("app.rag_pipeline.context_manager.update_context",
                   new=AsyncMock()), \
             patch("app.rag_pipeline.Profiler"):

            events = await _collect_events(pipeline.ask_stream(
                question="What is the mortgage deposit?",
                user_role="Manager",
                user_department="Lending",
            ))

        sources_events = [e for e in events if e["type"] == "sources"]
        assert len(sources_events) == 1, "Expected exactly one 'sources' event"

        sources = sources_events[0]["sources"]
        assert len(sources) == 2

        # Source 1 must exactly match CHUNK_A
        assert sources[0]["index"]          == 1
        assert sources[0]["document_name"]  == "Mortgage Policy 2026"
        assert sources[0]["page_number"]    == 4
        assert sources[0]["effective_date"] == "2026-01-01"
        assert sources[0]["doc_status"]     == "Active"

        # Source 2 must exactly match CHUNK_B
        assert sources[1]["index"]          == 2
        assert sources[1]["document_name"]  == "Personal Lending Guide Q1 2026"
        assert sources[1]["page_number"]    == 12
        assert sources[1]["effective_date"] == "2026-03-01"

    async def test_done_event_is_terminal(self):
        """'done' event must be the last event in the stream."""
        pipeline = _make_pipeline()

        async def _grounded_stream(*a, **kw):
            yield "NatWest mortgage products require a deposit [1]."

        with patch("app.rag_pipeline.retriever.search",
                   new=AsyncMock(return_value=[CHUNK_A])), \
             patch("app.rag_pipeline.llm_service.classify_intent",
                   new=AsyncMock(return_value="question about uploaded documents")), \
             patch("app.rag_pipeline.llm_service.stream_answer",
                   return_value=_grounded_stream()), \
             patch("app.rag_pipeline.llm_service.generate_summary_only",
                   new=AsyncMock(return_value="Summary")), \
             patch("app.rag_pipeline.log_retrieval_gap", new=AsyncMock()), \
             patch("app.rag_pipeline.context_manager.create_chat",
                   new=AsyncMock(return_value="chat-cite-3")), \
             patch("app.rag_pipeline.context_manager.get_chat",
                   new=AsyncMock(return_value=None)), \
             patch("app.rag_pipeline.context_manager.get_rolling_context",
                   new=AsyncMock(return_value="")), \
             patch("app.rag_pipeline.context_manager.update_context",
                   new=AsyncMock()), \
             patch("app.rag_pipeline.Profiler"):

            events = await _collect_events(pipeline.ask_stream(
                question="What is the mortgage deposit requirement?",
                user_role="Manager",
                user_department="Lending",
            ))

        assert events[-1]["type"] == "done", (
            f"Last event should be 'done', got '{events[-1]['type']}'"
        )

    async def test_empty_retrieval_emits_empty_sources(self):
        """Gate 1 abstention must still emit a 'sources' event with an empty list."""
        pipeline = _make_pipeline()

        with patch("app.rag_pipeline.retriever.search",
                   new=AsyncMock(return_value=[])), \
             patch("app.rag_pipeline.llm_service.classify_intent",
                   new=AsyncMock(return_value="question about uploaded documents")), \
             patch("app.rag_pipeline.log_retrieval_gap", new=AsyncMock()), \
             patch("app.rag_pipeline.context_manager.create_chat",
                   new=AsyncMock(return_value="chat-cite-4")), \
             patch("app.rag_pipeline.context_manager.get_chat",
                   new=AsyncMock(return_value=None)), \
             patch("app.rag_pipeline.context_manager.get_rolling_context",
                   new=AsyncMock(return_value="")), \
             patch("app.rag_pipeline.context_manager.update_context",
                   new=AsyncMock()), \
             patch("app.rag_pipeline.Profiler"):

            events = await _collect_events(pipeline.ask_stream(
                question="What is the penalty for early repayment on a £0 loan?",
                user_role="Teller",
                user_department="Retail",
            ))

        sources_events = [e for e in events if e["type"] == "sources"]
        assert len(sources_events) == 1
        assert sources_events[0]["sources"] == [], (
            "Gate 1 abstention must emit sources=[] not omit the event"
        )
