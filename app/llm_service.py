"""
LLM service using Hugging Face Inference API.

Provides:
- generate_full_answer
- generate_summary_answer
- compress_context
"""

import asyncio
import logging
from typing import Dict, List, Optional

from huggingface_hub import InferenceClient, AsyncInferenceClient

from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a secure internal Banking Assistant for NatWest, operating under strict FCA
(Financial Conduct Authority) Consumer Duty and UK regulatory guidelines.

Rules:
1. Base your answer STRICTLY on the retrieved document context provided below.
   Do NOT use general knowledge or speculate beyond what the sources state.
2. If the context does not contain the information needed, respond ONLY with this
   exact phrase and nothing else:
   "NATWEST_ABSTAIN: I don't have sufficient information in approved sources to answer this."
   Do NOT expand on it. Do NOT apologise. Do NOT guess.
3. When you use information from a context source, mark it with an inline reference
   such as [1], [2], [3] corresponding to the [Source N] label shown before each
   context block below. You may cite multiple sources per sentence, e.g. [1][3].
   IMPORTANT: Do NOT write out document names, page numbers, dates, or any other
   metadata yourself — the system appends verified citations automatically.
4. If the answer involves numeric calculations (e.g., interest rates, fees, ratios),
   show every step of the working explicitly before stating the result.
5. If comparing rates or policy terms, present the output as a Markdown table.
6. Do not disclose, repeat, or reference any personally identifiable information (PII)
   such as account numbers, sort codes, or customer names that may appear in the context.
7. Use formal, professional language appropriate for a UK banking environment.
8. CRITICAL — NEVER improvise, hallucinate, or use outside knowledge. If in any doubt,
   emit the NATWEST_ABSTAIN phrase from Rule 2.
"""



CONTEXT_COMPRESSION_TEMPLATE = """
Compress the following conversation history into a brief summary
(maximum 3 sentences).

Preserve:
- Key facts
- Questions asked
- Answers given

History:
{context}

Compressed summary:
"""


class LLMService:
    """Handles all LLM interactions."""

    def __init__(self):
        token = settings.HF_API_TOKEN or None

        self.client = InferenceClient(token=token)
        self.async_client = AsyncInferenceClient(token=token)

        self.model = settings.LLM_MODEL

        if not self.model:
            raise ValueError("LLM_MODEL is not configured")

        logger.info(f"Using LLM model: {self.model}")

    def _build_full_prompt(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        previous_context: Optional[str] = None,
    ) -> List[Dict]:

        chunks_text = ""

        for i, chunk in enumerate(retrieved_chunks, start=1):
            source       = chunk.get("document_name", "Unknown Document")
            page         = chunk.get("page_number", "N/A")
            eff_date     = chunk.get("effective_date") or "Not specified"
            doc_status   = chunk.get("doc_status") or "Active"

            chunks_text += (
                f"\n[Source {i}: {source} | Page: {page} | "
                f"Effective: {eff_date} | Status: {doc_status}]\n"
                f"{chunk.get('text', '')}\n"
            )

        user_parts = []

        if previous_context:
            user_parts.append(
                f"Previous conversation context:\n{previous_context}"
            )

        user_parts.append(
            f"Retrieved document context:\n{chunks_text}"
        )

        if retrieved_chunks:
            user_parts.append(
                f"Question: {question}\n\n"
                "Provide a detailed answer based only on the context above."
            )
        else:
            user_parts.append(
                f"Question: {question}\n\n"
                "Respond to the user naturally and conversationally. Do NOT attempt to summarize the previous context unless explicitly asked to do so."
            )

        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": "\n\n".join(user_parts),
            },
        ]

    async def _chat(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
    ) -> str:

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.exception("HF generation failed")

            raise RuntimeError(
                f"LLM generation failed using model "
                f"'{self.model}'. Original error: {e}"
            )

    async def stream_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        previous_context: Optional[str] = None,
    ):
        logger.info(f"Streaming answer for question: {question[:100]}")
        messages = self._build_full_prompt(question, retrieved_chunks, previous_context)
        
        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=settings.FULL_ANSWER_MAX_TOKENS,
                temperature=0.3,
                stream=True
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            logger.exception("HF streaming failed")
            yield f"\n\nError: LLM streaming failed ({e})"

    async def generate_summary_only(self, full_answer: str) -> str:
        """Background task to generate a 1-sentence summary of the final answer for the database."""
        logger.info("Generating summary for MongoDB context...")
        if len(full_answer) < 100:
            return full_answer # Too short to summarize
            
        messages = [
            {"role": "system", "content": "You are a concise summarizer."},
            {"role": "user", "content": f"Summarize this answer in 1 concise sentence:\n\n{full_answer}"}
        ]
        
        try:
            summary = await self._chat(messages, max_tokens=settings.SUMMARY_MAX_TOKENS, temperature=0.1)
            return summary
        except Exception:
            return full_answer[:300] + "..."

    async def compress_context(
        self,
        context: str,
    ) -> str:

        logger.info(
            f"Compressing context of length {len(context)}"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You compress conversation history while "
                    "preserving key facts."
                ),
            },
            {
                "role": "user",
                "content": CONTEXT_COMPRESSION_TEMPLATE.format(
                    context=context
                ),
            },
        ]

        result = await self._chat(
            messages=messages,
            max_tokens=settings.CONTEXT_SUMMARY_MAX_TOKENS,
            temperature=0.1,
        )

        logger.info(
            f"Compressed context from {len(context)} "
            f"to {len(result)} characters"
        )

        return result

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return len(text) // 4 if text else 0

    async def classify_intent(self, question: str) -> str:
        """
        Zero-Shot Query Router:
        Instantly classifies whether the question requires a database search
        or if it's just a conversational follow-up/greeting.
        """
        logger.info(f"Routing intent for question: '{question[:50]}...'")
        
        # We use a fast zero-shot model. Updated labels for better accuracy.
        candidate_labels = ["question about uploaded documents", "casual greeting or conversation"]
        
        try:
            # Run the zero-shot classifier synchronously in a thread
            result = await asyncio.to_thread(
                self.client.zero_shot_classification,
                text=question,
                candidate_labels=candidate_labels,
                multi_label=False
            )
            # The API returns a list of objects with .label and .score attributes
            winning_label = result[0].label
            score = result[0].score
            
            logger.info(f"Router decided: {winning_label} (score: {score:.2f})")
            
            # Bias towards database search: If it thinks it's conversational but isn't highly confident,
            # force it to search the database to prevent hallucinations on short factual questions.
            if winning_label == "casual greeting or conversation" and score < 0.70:
                logger.info(f"Conversational score too low ({score:.2f} < 0.70), defaulting to database search.")
                return "question about uploaded documents"
                
            return winning_label
        except Exception as e:
            logger.warning(f"Router failed ({e}), defaulting to database search.")
            return "question about uploaded documents"


llm_service = LLMService()