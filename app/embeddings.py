"""
Embedding generation using BAAI/bge-small-en-v1.5 via Hugging Face Inference API.

BGE models use a query prefix for asymmetric retrieval:
- Queries: prepend "Represent this sentence for searching relevant passages: "
- Documents/passages: no prefix needed

Embeddings are L2-normalized so dot product equals cosine similarity.
"""

import asyncio
import numpy as np
from typing import List
from huggingface_hub import InferenceClient
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# BGE model query instruction for asymmetric retrieval
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingService:
    """Generates text embeddings via the HF Inference API."""

    def __init__(self):
        token = settings.HF_API_TOKEN if settings.HF_API_TOKEN else None
        self.client = InferenceClient(token=token)
        self.model = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION

    async def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """
        Generate a single embedding vector.

        Args:
            text: The text to embed.
            is_query: If True, prepend the BGE query instruction prefix.
                      Set to True for user queries, False for document chunks.

        Returns:
            Normalized embedding vector of dimension 384.
        """
        if is_query:
            text = QUERY_PREFIX + text

        # Run synchronous HF client call in a thread to avoid blocking
        result = await asyncio.to_thread(
            self.client.feature_extraction,
            text,
            model=self.model,
        )

        embedding = np.array(result, dtype=np.float32)

        # Handle different output shapes from the API:
        #   (1, seq_len, dim) → take [CLS] token at [0][0]
        #   (seq_len, dim)    → take [CLS] token at [0]
        #   (dim,)            → already a single vector
        if embedding.ndim == 3:
            embedding = embedding[0][0]
        elif embedding.ndim == 2:
            embedding = embedding[0]

        # L2 normalize so dot product = cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    async def get_embeddings_batch(
        self, texts: List[str], is_query: bool = False, batch_size: int = 8
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with concurrency control.

        Processes in small batches to respect HF rate limits on the free tier.
        """
        embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = await asyncio.gather(
                *[self.get_embedding(t, is_query=is_query) for t in batch]
            )
            embeddings.extend(batch_results)
            logger.debug(
                f"Embedded batch {i // batch_size + 1}/"
                f"{(len(texts) + batch_size - 1) // batch_size}"
            )

        return embeddings


embedding_service = EmbeddingService()
