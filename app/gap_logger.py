"""
Retrieval Gap Logger — compliance audit trail for abstention events.

Every time the pipeline cannot ground an answer (either because retrieval
returned no candidates, or because the LLM response had insufficient overlap
with the retrieved context) this module writes a structured record to the
`retrieval_gaps` MongoDB collection.

The compliance team can query this collection to identify:
  - Topics not covered by any ingested policy document ("no_candidates").
  - Cases where documents exist but the LLM chose not to use them ("ungrounded").
  - Department / role patterns that correlate with information gaps.

Schema:
  {
    "query_text":  str,   # PII-redacted query
    "user_role":   str,
    "department":  str,
    "timestamp":   str,   # ISO 8601 UTC
    "reason":      "no_candidates" | "ungrounded"
  }
"""

import logging
from datetime import datetime, timezone

from app.database import get_db
from app.pii_utils import redact_query

logger = logging.getLogger(__name__)

# ── Reason constants ─────────────────────────────────────────────────────────
REASON_NO_CANDIDATES = "no_candidates"
REASON_UNGROUNDED    = "ungrounded"


async def log_retrieval_gap(
    query_text: str,
    user_role: str,
    department: str,
    reason: str,
) -> None:
    """
    Persist an abstention event to the `retrieval_gaps` collection.

    Always PII-redacts the query before storing it so the audit log
    itself cannot become a source of data leakage.

    Args:
        query_text: The raw user question (will be redacted before storage).
        user_role:  Token-resolved user role (Teller / Manager / …).
        department: Token-resolved user department.
        reason:     REASON_NO_CANDIDATES or REASON_UNGROUNDED.
    """
    try:
        db = get_db()
        record = {
            "query_text": redact_query(query_text),
            "user_role":  user_role,
            "department": department,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "reason":     reason,
        }
        await db.retrieval_gaps.insert_one(record)
        logger.info(
            f"📋 [Gap Log] reason={reason} | role={user_role} | "
            f"dept={department} | query='{record['query_text'][:60]}…'"
        )
    except Exception as exc:
        # Gap logging must never crash the main pipeline.
        logger.error(f"❌ [Gap Log] Failed to write retrieval gap: {exc}")
