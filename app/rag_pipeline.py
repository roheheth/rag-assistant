"""
RAG Pipeline — the main orchestrator that ties everything together.

Flow for each question:
  1. Create or load chat
  2. Load rolling context from MongoDB (if follow-up)
  3. Retrieve relevant chunks via vector search
  4. Generate full answer with LLM (context + chunks + question)
  5. Generate summary answer from the full answer
  6. Update rolling context in MongoDB
  7. Return both answers + sources + token stats
"""

from typing import Optional
from app.retriever import retriever
from app.llm_service import llm_service
from app.context_manager import context_manager
from app.models import AnswerResponse, Source, TokenStats
import logging

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates the full RAG flow from question to answer."""

    async def ask_stream(
        self, question: str, chat_id: Optional[str] = None
    ):
        """
        Process a user question and stream the response.
        """
        import json
        import os
        import time
        from pyinstrument import Profiler
        
        # Toggle this to True if you ever want the heavy HTML flame graph again!
        USE_PYINSTRUMENT = True 
        
        start_time = time.time()
        
        if USE_PYINSTRUMENT:
            profiler = Profiler(async_mode="enabled")
            profiler.start()

        # ── 1. Handle chat ─────────────────────────────────────────
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

        # ── 2. Load rolling previous context ───────────────────────
        previous_context = await context_manager.get_rolling_context(chat_id)

        # ── 3. Query Router (Zero-Shot Classification) ───────────────
        intent = await llm_service.classify_intent(question)

        # ── 4. Retrieve relevant chunks (Conditional) ────────────────
        if intent == "question about uploaded documents":
            retrieved_chunks = await retriever.search(question)
            if not retrieved_chunks:
                logger.warning("No relevant chunks found for query")
                # Yield metadata
                yield f"data: {json.dumps({'type': 'metadata', 'chat_id': chat_id, 'sources': [], 'turn_number': turn_count})}\n\n"
                
                msg = "I couldn't find any relevant information in the uploaded documents to answer your question."
                yield f"data: {json.dumps({'type': 'chunk', 'content': msg})}\n\n"
                
                # Update context
                await context_manager.update_context(chat_id, question, msg, msg)
                return
        else:
            logger.info("Router bypassed database search (Conversational intent)")
            retrieved_chunks = []

        sources = [
            Source(
                text=chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"],
                document_name=chunk["document_name"],
                page_number=chunk.get("page_number"),
                relevance_score=round(chunk["score"], 4),
            )
            for chunk in retrieved_chunks
        ]

        # ── 5. Yield Metadata first ──────────────────────────────
        metadata = {
            "type": "metadata",
            "chat_id": chat_id,
            "sources": [s.dict() for s in sources],
            "turn_number": turn_count
        }
        yield f"data: {json.dumps(metadata)}\n\n"

        # ── 6. Stream full answer ──────────────────────
        full_answer = ""
        first_token_arrived = False
        async for chunk in llm_service.stream_answer(question, retrieved_chunks, previous_context):
            if not first_token_arrived:
                ttft = time.time() - start_time
                logger.info(f"⏱️ Time To First Token (TTFT): {ttft:.3f} seconds")
                first_token_arrived = True
                
            full_answer += chunk
            # Yield token
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

        # ── 7. Generate Summary and Update Context in Background ───────────────────────────────
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
        logger.info(f"✓ Answer generated & Context Updated | chat_id={chat_id}")
        
        total_time = time.time() - start_time
        logger.info(f"⏱️ Total Processing Time: {total_time:.3f} seconds")
        
        if USE_PYINSTRUMENT:
            profiler.stop()
            os.makedirs("profiles", exist_ok=True)
            with open("profiles/streaming_profile.html", "w", encoding="utf-8") as f:
                f.write(profiler.output_html())


rag_pipeline = RAGPipeline()
