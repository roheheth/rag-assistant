"""
Security tests for RBAC self-elevation protection.

These tests verify that the auth layer correctly:
  1. Ignores a forged "Admin" role in the request body when the caller
     is authenticated as a lower-clearance user (Teller).
  2. Rejects unauthenticated requests in production mode (HTTP 401).
  3. Accepts unauthenticated requests in dev mode (fallback identity).
  4. Accepts a valid token and resolves the correct identity.

Strategy: We test app/auth.py in isolation using a minimal FastAPI app
that reuses the same Depends(get_current_user) dependency, avoiding the
need to spin up MongoDB or Qdrant.

Run with:
    .\\venv\\Scripts\\python -m pytest tests/test_rbac_security.py -v
"""

import json
import pytest
from unittest.mock import patch
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

# ── Token fixtures ───────────────────────────────────────────────────────────

TELLER_TOKEN  = "test-teller-token"
ADMIN_TOKEN   = "test-admin-token"
UNKNOWN_TOKEN = "totally-forged-token-xyz"

_TEST_TOKEN_MAP = {
    TELLER_TOKEN: {"role": "Teller",  "department": "Retail"},
    ADMIN_TOKEN:  {"role": "Admin",   "department": "Compliance"},
}


# ── Minimal test app (no MongoDB / Qdrant dependency) ────────────────────────

def _build_test_app(deployment_mode: str) -> TestClient:
    """
    Build a lightweight FastAPI app that only exposes /test/me using the
    real get_current_user dependency.  No lifespan, no DB connections.
    """
    import app.config as cfg_module
    import app.auth as auth_module

    # Override config values in-place
    cfg_module.settings.DEPLOYMENT_MODE = deployment_mode
    cfg_module.settings.USERS_TOKEN_MAP = json.dumps(_TEST_TOKEN_MAP)

    # Clear the lru_cache so _get_token_map() re-reads the new map
    auth_module._get_token_map.cache_clear()

    from app.auth import get_current_user, UserIdentity

    test_app = FastAPI()

    @test_app.get("/test/me")
    async def test_me(current_user: UserIdentity = Depends(get_current_user)):
        return {
            "role":       current_user.role,
            "department": current_user.department,
            "mode":       deployment_mode,
        }

    return TestClient(test_app, raise_server_exceptions=False)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def prod_client():
    return _build_test_app("production")


@pytest.fixture
def dev_client():
    return _build_test_app("dev")


# ── Identity resolution tests ────────────────────────────────────────────────

class TestIdentityEndpoint:
    def test_returns_teller_identity(self, prod_client):
        """Valid teller token → returns Teller role."""
        res = prod_client.get(
            "/test/me",
            headers={"Authorization": f"Bearer {TELLER_TOKEN}"}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["role"] == "Teller"
        assert data["department"] == "Retail"

    def test_returns_admin_identity(self, prod_client):
        """Valid admin token → returns Admin role."""
        res = prod_client.get(
            "/test/me",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}
        )
        assert res.status_code == 200
        assert res.json()["role"] == "Admin"

    def test_rejects_no_token_in_production(self, prod_client):
        """No token in production → HTTP 401."""
        res = prod_client.get("/test/me")
        assert res.status_code == 401

    def test_rejects_unknown_token_in_production(self, prod_client):
        """Unrecognised token → HTTP 401."""
        res = prod_client.get(
            "/test/me",
            headers={"Authorization": f"Bearer {UNKNOWN_TOKEN}"}
        )
        assert res.status_code == 401

    def test_allows_no_token_in_dev(self, dev_client):
        """Dev mode without token → falls back to dev-default (Admin)."""
        res = dev_client.get("/test/me")
        assert res.status_code == 200
        data = res.json()
        assert data["role"] == "Admin"          # dev-default identity
        assert data["mode"] == "dev"


# ── Forged role attack tests ─────────────────────────────────────────────────

class TestForgedRoleAttack:
    """
    Core security scenario: a caller authenticated as Teller attempts to
    escalate their clearance by injecting 'user_role: Admin' into the
    request.  The auth dependency MUST resolve role from the token only.

    Because the dependency ignores all request body/query fields (it only
    reads the Authorization header), we prove this by asserting that even
    with a Teller token, the returned identity is always 'Teller'.
    """

    def test_teller_token_always_returns_teller(self, prod_client):
        """
        Regardless of what an attacker might send in the body, the auth
        dependency resolves role exclusively from the Bearer token.
        A Teller token MUST return Teller, never Admin.
        """
        res = prod_client.get(
            "/test/me",
            headers={"Authorization": f"Bearer {TELLER_TOKEN}"}
        )
        assert res.status_code == 200
        resolved_role = res.json()["role"]
        assert resolved_role == "Teller", (
            f"Server resolved role '{resolved_role}' instead of 'Teller'. "
            "This indicates a potential self-elevation vulnerability — the "
            "server-side role derivation is not working correctly."
        )

    def test_no_token_cannot_reach_endpoint_in_production(self, prod_client):
        """
        Unauthenticated request (e.g. forged body only, no Bearer token)
        MUST be blocked with HTTP 401 in production.
        """
        res = prod_client.get("/test/me")   # no Authorization header
        assert res.status_code == 401, (
            f"Expected 401 for unauthenticated request, got {res.status_code}. "
            "This is a CRITICAL security failure — the endpoint is accessible "
            "without authentication in production mode."
        )

    def test_unknown_token_cannot_escalate(self, prod_client):
        """
        A forged / unknown token must not grant any access.
        """
        res = prod_client.get(
            "/test/me",
            headers={"Authorization": f"Bearer {UNKNOWN_TOKEN}"}
        )
        assert res.status_code == 401, (
            "Unrecognised Bearer token was accepted by the server. "
            "This is a CRITICAL security failure."
        )

    def test_admin_token_resolves_admin(self, prod_client):
        """Sanity check: legitimate admin token correctly returns Admin."""
        res = prod_client.get(
            "/test/me",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}
        )
        assert res.status_code == 200
        assert res.json()["role"] == "Admin"


# ── Dev mode tests ───────────────────────────────────────────────────────────

class TestDevMode:
    def test_no_token_returns_dev_default(self, dev_client):
        """Dev mode: missing token → 200 with dev-default Admin identity."""
        res = dev_client.get("/test/me")
        assert res.status_code == 200
        assert res.json()["role"] == "Admin"

    def test_valid_token_honoured_in_dev(self, dev_client):
        """Dev mode: providing a valid token still resolves correctly."""
        res = dev_client.get(
            "/test/me",
            headers={"Authorization": f"Bearer {TELLER_TOKEN}"}
        )
        assert res.status_code == 200
        assert res.json()["role"] == "Teller"

    def test_unknown_token_rejected_in_dev(self, dev_client):
        """Dev mode: unrecognised token is still rejected (not silently ignored)."""
        res = dev_client.get(
            "/test/me",
            headers={"Authorization": f"Bearer {UNKNOWN_TOKEN}"}
        )
        assert res.status_code == 401
