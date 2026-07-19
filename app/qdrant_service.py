"""
Qdrant vector store service.

Uses local file-based persistence — no Docker or external server required.
The Qdrant data is stored in the QDRANT_PATH directory (default: ./qdrant_storage).

Responsibilities:
  - Initialize the Qdrant collection with the correct vector dimension on startup.
  - upsert_vectors(): Bulk-insert child chunk embeddings with metadata payload.
  - search_vectors(): Run ANN vector search and return (chunk_id, score) pairs.
"""

import logging
from typing import List, Tuple, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.config import settings

logger = logging.getLogger(__name__)


class QdrantService:
    """Manages the local Qdrant vector collection."""

    def __init__(self):
        self._client = None
        self.collection = settings.QDRANT_COLLECTION
        self.dimension = settings.EMBEDDING_DIMENSION

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(path=settings.QDRANT_PATH)
        return self._client

    def initialize_collection(self):
        """
        Create the Qdrant collection if it does not already exist.
        Called once on application startup from the lifespan event.
        """
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                f"✅ Qdrant collection '{self.collection}' created "
                f"(dimension={self.dimension}, metric=cosine)"
            )
        else:
            logger.info(
                f"✅ Qdrant collection '{self.collection}' already exists — skipping creation."
            )

    def upsert_vectors(self, records: List[Dict]):
        """
        Bulk-insert a list of chunk records into Qdrant.

        Each record must have:
          - chunk_id  (str): unique ID used as Qdrant point ID
          - embedding (List[float]): the 384-float vector
          - parent_id, document_id, document_name, page_number: stored as payload
        """
        if not records:
            return

        # Qdrant requires integer or UUID point IDs.
        # We derive a stable integer from the chunk_id string using Python's hash.
        points = []
        for rec in records:
            # Use a consistent deterministic integer ID from chunk_id string
            point_id = abs(hash(rec["chunk_id"])) % (2**63)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=rec["embedding"],
                    payload={
                        "chunk_id":      rec["chunk_id"],
                        "parent_id":     rec["parent_id"],
                        "document_id":   rec["document_id"],
                        "document_name": rec["document_name"],
                        "page_number":   rec.get("page_number"),
                    },
                )
            )

        self.client.upsert(collection_name=self.collection, points=points)
        logger.info(f"💾 Qdrant: Upserted {len(points)} vectors into '{self.collection}'")

    def search_vectors(
        self, query_vector: List[float], top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Run ANN (Approximate Nearest Neighbor) search in Qdrant.

        Returns:
            List of (chunk_id, score) tuples sorted by descending similarity.
        """
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        hits = []
        for point in results.points:
            chunk_id = point.payload.get("chunk_id", "")
            hits.append((chunk_id, point.score))

        logger.info(
            f"🔍 Qdrant: Vector search returned {len(hits)} candidates "
            f"(top score: {hits[0][1]:.4f})" if hits else
            f"🔍 Qdrant: Vector search returned 0 results"
        )
        return hits

    def delete_by_document(self, document_id: str):
        """
        Delete all Qdrant vectors belonging to a specific document.
        Called during document deletion.
        """
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
        logger.info(f"🗑️ Qdrant: Deleted all vectors for document_id='{document_id}'")


qdrant_service = QdrantService()
