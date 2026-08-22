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
from typing import List, Optional, Tuple, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)
from datetime import date as date_type
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
            point_id = abs(hash(rec["chunk_id"])) % (2**63)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=rec["embedding"],
                    payload={
                        "chunk_id":        rec["chunk_id"],
                        "parent_id":       rec["parent_id"],
                        "document_id":     rec["document_id"],
                        "document_name":   rec["document_name"],
                        "page_number":     rec.get("page_number"),
                        # Banking compliance fields
                        "clearance_level": rec.get("clearance_level", "Internal"),
                        "department":      rec.get("department", "Retail"),
                        "effective_date":  rec.get("effective_date", ""),
                        "expiry_date":     rec.get("expiry_date", "2099-12-31"),
                        "doc_status":      rec.get("doc_status", "Active"),
                    },
                )
            )

        self.client.upsert(collection_name=self.collection, points=points)
        logger.info(f"💾 Qdrant: Upserted {len(points)} vectors into '{self.collection}'")

    # ── RBAC Clearance Map ───────────────────────────────────────
    ROLE_CLEARANCE_MAP: dict = {
        "Teller":    ["Public"],
        "Manager":   ["Public", "Internal"],
        "Executive": ["Public", "Internal", "Restricted"],
        "Admin":     ["Public", "Internal", "Restricted"],
    }

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 20,
        user_role: str = "Admin",
        user_department: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Run ANN search in Qdrant with RBAC and temporal pre-filtering.

        Filters applied:
          1. doc_status == "Active"
          2. clearance_level in allowed levels for user_role
          3. Chunks whose effective_date <= today <= expiry_date

        Returns:
            List of (chunk_id, score) tuples sorted by descending similarity.
        """
        today = date_type.today().isoformat()   # e.g. "2026-08-22"

        # Determine which clearance levels this user may access
        allowed_levels = self.ROLE_CLEARANCE_MAP.get(user_role, ["Public"])

        # Build Qdrant must-filters
        must_conditions = [
            # Only retrieve Active documents
            FieldCondition(key="doc_status", match=MatchValue(value="Active")),
            # RBAC: limit to clearance levels allowed for this role
            FieldCondition(key="clearance_level", match=MatchAny(any=allowed_levels)),
        ]

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=Filter(must=must_conditions),
        )

        # Post-filter: date range check (effective_date <= today <= expiry_date)
        hits = []
        for point in results.points:
            eff  = point.payload.get("effective_date", "0000-01-01") or "0000-01-01"
            exp  = point.payload.get("expiry_date",    "2099-12-31") or "2099-12-31"
            if not (eff <= today <= exp):
                continue    # skip temporally expired or not-yet-active policies
            chunk_id = point.payload.get("chunk_id", "")
            hits.append((chunk_id, point.score))

        logger.info(
            f"🔍 Qdrant: Vector search returned {len(hits)} candidates "
            f"(role={user_role}, clearances={allowed_levels}) "
            + (f"(top score: {hits[0][1]:.4f})" if hits else "")
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
