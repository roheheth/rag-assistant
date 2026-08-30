"""
Authentication & Identity dependency for NatWest RAG Assistant.

Strategy: Bearer Token → Server-side role resolution.

In PRODUCTION mode:
  - Every request MUST carry an `Authorization: Bearer <token>` header.
  - The token is looked up in USERS_TOKEN_MAP (loaded from .env).
  - role and department are derived exclusively from the server-side map.
  - Client-supplied user_role / user_department fields are IGNORED.
  - Missing or unrecognised tokens → HTTP 401.

In DEV mode:
  - If a valid token is present, it is honoured (same lookup).
  - If NO token is present, a dev default identity (Admin / Retail) is
    returned so that local development works without credentials.
  - A visible banner is rendered in the Streamlit UI to signal dev mode.

Usage (FastAPI dependency):
    from app.auth import get_current_user, UserIdentity
    ...
    @app.post("/api/ask")
    async def ask_question(
        request: QuestionRequest,
        current_user: UserIdentity = Depends(get_current_user),
    ):
        ...
"""

import json
import logging
from functools import lru_cache
from typing import Dict

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)

# ── Optional bearer extractor (auto_error=False so we can give a nicer msg) ─
_bearer_scheme = HTTPBearer(auto_error=False)


# ── UserIdentity ────────────────────────────────────────────────────────────

class UserIdentity(BaseModel):
    """Resolved, server-side identity — never trust values from the client."""
    role: str
    department: str
    token: str = "dev-default"   # for audit logging


# ── Token map (parsed once at startup) ─────────────────────────────────────

@lru_cache(maxsize=1)
def _get_token_map() -> Dict[str, dict]:
    """
    Parse USERS_TOKEN_MAP JSON string into a dict.
    Example .env entry:
        USERS_TOKEN_MAP={"tok-teller-1": {"role": "Teller", "department": "Retail"}}
    """
    raw = settings.USERS_TOKEN_MAP.strip()
    if not raw or raw == "{}":
        logger.warning(
            "⚠️  USERS_TOKEN_MAP is empty. "
            "All authenticated production requests will be rejected."
        )
        return {}
    try:
        token_map = json.loads(raw)
        logger.info(f"🔑 Auth: Loaded {len(token_map)} user token(s) from USERS_TOKEN_MAP.")
        return token_map
    except json.JSONDecodeError as exc:
        logger.error(f"❌ Auth: Failed to parse USERS_TOKEN_MAP JSON — {exc}")
        return {}


# ── Dev-mode default identity ────────────────────────────────────────────────

_DEV_DEFAULT_IDENTITY = UserIdentity(
    role="Admin",
    department="Retail",
    token="dev-default",
)


# ── FastAPI dependency ───────────────────────────────────────────────────────

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> UserIdentity:
    """
    FastAPI dependency that resolves the caller's server-side identity.

    Production: token required; role/department come from USERS_TOKEN_MAP.
    Dev:        token optional; falls back to Admin/Retail dev default.

    SECURITY NOTE: This function is the single authoritative source of
    role/department for every protected endpoint.  The request body
    fields `user_role` and `user_department` MUST be ignored by all
    callers of this dependency.
    """
    mode = settings.DEPLOYMENT_MODE.lower()
    token_map = _get_token_map()

    if credentials is not None:
        token = credentials.credentials
        identity_data = token_map.get(token)

        if identity_data is None:
            logger.warning(
                f"🚫 Auth: Unrecognised token '{token[:8]}…' "
                f"rejected (mode={mode})."
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired Bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        identity = UserIdentity(
            role=identity_data.get("role", "Teller"),
            department=identity_data.get("department", "Retail"),
            token=token[:8] + "…",   # truncated for safe logging
        )
        logger.info(
            f"✅ Auth: Token accepted — role={identity.role}, "
            f"dept={identity.department} (mode={mode})"
        )
        return identity

    # ── No token supplied ────────────────────────────────────────────────
    if mode == "production":
        logger.warning("🚫 Auth: No Bearer token in production mode — rejecting.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Dev mode: fall back to unrestricted dev identity
    logger.debug("🛠️  Auth: No token in dev mode — using dev-default identity.")
    return _DEV_DEFAULT_IDENTITY
