"""
PII (Personally Identifiable Information) Redaction Engine.

Provides regex-based masking of sensitive UK banking data before any text
is embedded, stored, or sent to external APIs.

Redacted patterns:
  - UK Sort Codes         → [REDACTED_SORT_CODE]
  - UK Account Numbers    → [REDACTED_ACCOUNT]
  - National Insurance    → [REDACTED_NI_NUMBER]
  - Credit Card Numbers   → [REDACTED_CARD]
  - Email Addresses       → [REDACTED_EMAIL]
  - Named Individuals     → [REDACTED_NAME]
  - UK Phone Numbers      → [REDACTED_PHONE]

Usage:
    from app.pii_utils import redact_pii
    clean_text = redact_pii(raw_text)
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Ordered list of (pattern, replacement, label) tuples ──────────────
# Order matters: more specific patterns come before broader ones.
_PII_RULES = [
    # Credit card: 16 digits with optional spaces or dashes between groups of 4
    (
        r'\b(?:\d{4}[- ]?){3}\d{4}\b',
        "[REDACTED_CARD]",
        "credit card",
    ),
    # UK Sort Code: XX-XX-XX or XXXXXX (6 consecutive digits)
    (
        r'\b\d{2}-\d{2}-\d{2}\b',
        "[REDACTED_SORT_CODE]",
        "sort code (hyphenated)",
    ),
    (
        r'\b\d{6}\b(?!\d)',           # 6 digits NOT followed by more digits
        "[REDACTED_SORT_CODE]",
        "sort code (plain)",
    ),
    # UK Account Number: 8 consecutive digits (not already swallowed by card)
    (
        r'\b\d{8}\b',
        "[REDACTED_ACCOUNT]",
        "account number",
    ),
    # National Insurance Number: AB 12 34 56 C (with or without spaces)
    (
        r'\b[A-CEGHJ-PR-TW-Z]{2}\s*\d{2}\s*\d{2}\s*\d{2}\s*[A-D]\b',
        "[REDACTED_NI_NUMBER]",
        "NI number",
    ),
    # Email address
    (
        r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
        "[REDACTED_EMAIL]",
        "email",
    ),
    # UK phone numbers: 07xxx xxxxxx or +44 xxxx xxxxxx or 01xxx xxxxxx
    (
        r'(?:\+44\s?|0)(?:\d\s?){9,10}',
        "[REDACTED_PHONE]",
        "phone number",
    ),
    # Named individuals: Mr./Mrs./Ms./Dr./Prof. followed by capitalised word(s)
    (
        r'\b(?:Mr\.|Mrs\.|Ms\.|Miss|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b',
        "[REDACTED_NAME]",
        "named individual",
    ),
]


def redact_pii(text: str) -> str:
    """
    Apply all PII redaction rules to a block of text.

    Returns the sanitised text with all sensitive patterns replaced
    by placeholder tokens. A log entry is emitted if any redactions occur.
    """
    if not text:
        return text

    redacted = text
    total_hits = 0

    for pattern, replacement, label in _PII_RULES:
        new_text, count = re.subn(pattern, replacement, redacted)
        if count:
            total_hits += count
            logger.debug(f"🔒 PII Redacted: {count} x {label}")
        redacted = new_text

    if total_hits:
        logger.info(f"🔒 [PII Redaction] Masked {total_hits} sensitive items in text block.")

    return redacted


def redact_query(query: str) -> str:
    """
    Redact PII from a user query before it is embedded and searched.
    This ensures query terms align with the redacted index content.
    """
    return redact_pii(query)
