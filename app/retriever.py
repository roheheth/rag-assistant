"""
Hybrid retriever using Qdrant for vector search and MongoDB for BM25 keyword search.

Flow:
  1. Query Qdrant for top-K ANN vector matches (fast, indexed, off-server math).
  2. Load child chunk tokens from MongoDB for BM25 scoring (in-memory keyword match).
  3. Merge both rankings using Reciprocal Rank Fusion (RRF).
  4. Fetch full Parent chunk text from MongoDB for LLM context.

Since vectors are stored in Qdrant with an HNSW index, vector search now runs in
O(log N) instead of O(N), making it scalable to hundreds of thousands of chunks.
"""

import re
import numpy as np
from typing import List, Dict
from rank_bm25 import BM25Okapi
from app.database import get_db
from app.embeddings import embedding_service
from app.qdrant_service import qdrant_service
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves the most relevant document chunks for a query via Hybrid Search."""

    def _tokenize(self, text: str) -> List[str]:
        """Convert text into lowercase word tokens, filtering out punctuation."""
        return re.findall(r'\w+', text.lower())

    async def search(
        self, query: str, top_k: int = None
    ) -> List[Dict]:
        """
        Find the top-k most relevant Parent chunks using Hybrid Search (BM25 + Qdrant + RRF).
        """
        if top_k is None:
            top_k = settings.TOP_K

        db = get_db()
        candidate_count = min(top_k * 4, 100)

        # ── 1. Embed the Query ─────────────────────────────────────
        query_embedding = await embedding_service.get_embedding(
            query, is_query=True
        )

        # ── 2. Vector Search via Qdrant (ANN, off-server) ──────────
        # Returns [(chunk_id, score)] sorted by descending cosine similarity
        qdrant_hits = qdrant_service.search_vectors(
            query_vector=query_embedding,
            top_k=candidate_count,
        )

        if not qdrant_hits:
            logger.warning("Qdrant returned 0 vector results — no documents indexed yet.")
            return []

        # Build a quick lookup: chunk_id → qdrant_score
        qdrant_score_map: Dict[str, float] = {
            chunk_id: score for chunk_id, score in qdrant_hits
        }
        qdrant_chunk_ids = list(qdrant_score_map.keys())

        # ── 3. Fetch matching child chunk metadata from MongoDB ─────
        # We only fetch the candidates returned by Qdrant (not ALL chunks)
        cursor = db.chunks.find(
            {"chunk_id": {"$in": qdrant_chunk_ids}},
            {
                "_id": 0,
                "chunk_id": 1,
                "parent_id": 1,
                "text": 1,
                "tokens": 1,
                "document_id": 1,
                "document_name": 1,
                "page_number": 1,
            },
        )
        children = await cursor.to_list(length=None)

        if not children:
            logger.warning("Qdrant returned hits but MongoDB has no matching chunk metadata.")
            return []

        # ── 4. BM25 Keyword Search ─────────────────────────────────
        tokenized_corpus = [c.get("tokens") or self._tokenize(c["text"]) for c in children]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = self._tokenize(query)
        bm25_scores = bm25.get_scores(tokenized_query)

        # ── 5. Build rank maps for both algorithms ─────────────────
        # Qdrant rank: order by Qdrant score descending
        qdrant_scores_ordered = np.array(
            [qdrant_score_map.get(c["chunk_id"], 0.0) for c in children],
            dtype=np.float32,
        )
        qdrant_rankings = np.argsort(qdrant_scores_ordered)[::-1]
        qdrant_rank_map = {idx: rank for rank, idx in enumerate(qdrant_rankings)}

        # BM25 rank: order by BM25 score descending
        bm25_rankings = np.argsort(bm25_scores)[::-1]
        bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_rankings)}

        # ── 6. Reciprocal Rank Fusion (RRF) ────────────────────────
        rrf_scores = np.array([
            (1.0 / (60.0 + qdrant_rank_map[idx])) + (1.0 / (60.0 + bm25_rank_map[idx]))
            for idx in range(len(children))
        ])

        # ── 7. Filter & Deduplicate by Parent ─────────────────────
        top_rrf_indices = np.argsort(rrf_scores)[::-1]

        seen_parent_ids = []
        best_score_for_parent: Dict[str, float] = {}
        best_vector_similarity: Dict[str, float] = {}

        for idx in top_rrf_indices:
            v_score = float(qdrant_scores_ordered[idx])
            b_score = float(bm25_scores[idx])

            # Keep if it passes the similarity threshold OR is a keyword hit
            if v_score < settings.SIMILARITY_THRESHOLD and b_score <= 0:
                continue

            child = children[idx]
            parent_id = child.get("parent_id")

            if parent_id and parent_id not in best_score_for_parent:
                seen_parent_ids.append(parent_id)
                best_score_for_parent[parent_id] = float(rrf_scores[idx])
                best_vector_similarity[parent_id] = v_score

            if len(seen_parent_ids) >= top_k:
                break

        if not seen_parent_ids:
            logger.info("No chunks passed the similarity or lexical threshold")
            return []

        # ── 8. Fetch Parent chunks from MongoDB ─────────────────────
        parent_cursor = db.parent_chunks.find(
            {"parent_id": {"$in": seen_parent_ids}},
            {"_id": 0, "parent_id": 1, "text": 1, "document_name": 1,
             "document_id": 1, "page_number": 1},
        )
        parent_docs = await parent_cursor.to_list(length=None)
        parent_map = {p["parent_id"]: p for p in parent_docs}

        # ── 9. Build final ranked results ──────────────────────────
        results = []
        for parent_id in seen_parent_ids:
            parent = parent_map.get(parent_id)
            if not parent:
                continue
            results.append({
                "text":              parent["text"],
                "document_name":     parent["document_name"],
                "document_id":       parent["document_id"],
                "page_number":       parent.get("page_number"),
                "score":             best_score_for_parent[parent_id],
                "vector_similarity": best_vector_similarity[parent_id],
            })

        logger.info(
            f"✓ Retrieved {len(results)} parent chunks via Qdrant+BM25+RRF "
            f"(top scores: {[round(r['score'], 4) for r in results[:3]]})"
        )
        return results


retriever = Retriever()
