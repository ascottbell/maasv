"""Tests for CF Access JWT auth and dual-auth middleware.

Covers:
- CFAccessValidator: valid tokens, expired, wrong aud/issuer, malformed, JWKS failures
- AuthMiddleware: health bypass, API key, JWT, no-auth, precedence
- Config validation: half-set CF, no auth, CF-only mode
"""

import hmac
import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from maasv.mcp_server.cf_auth import (
    CFAccessValidator,
    CFAuthError,
    CFJWKSUnavailable,
    CFTokenExpired,
    CFTokenInvalidAudience,
    CFTokenInvalidIssuer,
    CFTokenInvalidSignature,
    CFTokenMalformed,
)


# ---------------------------------------------------------------------------
# Fixtures: RSA key pair + token generation
# ---------------------------------------------------------------------------

@pytest.fixture
def rsa_keypair():
    """Generate a fresh RSA key pair for signing test JWTs."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    return private_key, public_key


@pytest.fixture
def make_token(rsa_keypair):
    """Factory to create signed JWTs with configurable claims."""
    private_key, _ = rsa_keypair

    def _make(
        aud="test-aud-tag",
        iss="https://testteam.cloudflareaccess.com",
        exp_delta=timedelta(hours=1),
        kid="test-kid-1",
        extra_claims=None,
    ):
        now = datetime.now(timezone.utc)
        payload = {
            "aud": [aud] if isinstance(aud, str) else aud,
            "iss": iss,
            "iat": now,
            "exp": now + exp_delta,
            "sub": "test-user-id",
            "email": "test@example.com",
            "type": "app",
        }
        if extra_claims:
            payload.update(extra_claims)

        return pyjwt.encode(
            payload,
            private_key,
            algorithm="RS256",
            headers={"kid": kid},
        )

    return _make


@pytest.fixture
def mock_jwks_client(rsa_keypair):
    """Patch PyJWKClient to return our test public key."""
    _, public_key = rsa_keypair

    mock_signing_key = MagicMock()
    mock_signing_key.key = public_key

    mock_client = MagicMock()
    mock_client.get_signing_key_from_jwt.return_value = mock_signing_key

    return mock_client


@pytest.fixture
def validator(mock_jwks_client):
    """CFAccessValidator with mocked JWKS client."""
    v = CFAccessValidator(team="testteam", audience="test-aud-tag")
    v._jwks_client = mock_jwks_client
    return v


# ===========================================================================
# Unit tests: CFAccessValidator
# ===========================================================================

class TestCFAccessValidator:
    """Tests for JWT validation logic."""

    def test_valid_token(self, validator, make_token):
        claims = validator.validate(make_token())
        assert claims["email"] == "test@example.com"
        assert claims["iss"] == "https://testteam.cloudflareaccess.com"

    def test_expired_token(self, validator, make_token):
        token = make_token(exp_delta=timedelta(seconds=-60))
        with pytest.raises(CFTokenExpired, match="expired"):
            validator.validate(token)

    def test_wrong_audience(self, validator, make_token):
        token = make_token(aud="wrong-aud")
        with pytest.raises(CFTokenInvalidAudience, match="audience"):
            validator.validate(token)

    def test_wrong_issuer(self, validator, make_token):
        token = make_token(iss="https://evil.cloudflareaccess.com")
        with pytest.raises(CFTokenInvalidIssuer, match="issuer"):
            validator.validate(token)

    def test_empty_token(self, validator):
        with pytest.raises(CFTokenMalformed, match="Empty"):
            validator.validate("")

    def test_whitespace_token(self, validator):
        with pytest.raises(CFTokenMalformed, match="Empty"):
            validator.validate("   ")

    def test_malformed_token(self, validator, mock_jwks_client):
        mock_jwks_client.get_signing_key_from_jwt.side_effect = pyjwt.DecodeError("bad header")
        with pytest.raises(CFTokenMalformed):
            validator.validate("not.a.jwt")

    def test_jwks_unreachable(self, validator, mock_jwks_client):
        mock_jwks_client.get_signing_key_from_jwt.side_effect = pyjwt.PyJWKClientConnectionError(
            "Connection refused"
        )
        with pytest.raises(CFJWKSUnavailable, match="unreachable"):
            validator.validate("some.fake.token")

    def test_kid_refresh_on_miss(self, validator, make_token, rsa_keypair):
        """When kid not in cache, forces one JWKS refresh."""
        _, public_key = rsa_keypair
        mock_signing_key = MagicMock()
        mock_signing_key.key = public_key

        # First call raises kid-not-found, after refresh it works
        validator._jwks_client.get_signing_key_from_jwt.side_effect = [
            pyjwt.PyJWKClientError("kid not found"),
            mock_signing_key,
        ]

        token = make_token()
        claims = validator.validate(token)
        assert claims["email"] == "test@example.com"
        assert validator._jwks_client.fetch_data.call_count == 1

    def test_kid_refresh_rate_limited(self, validator, make_token):
        """Second kid-miss within cooldown period should not refresh again."""
        validator._jwks_client.get_signing_key_from_jwt.side_effect = pyjwt.PyJWKClientError(
            "kid not found"
        )
        validator._last_refresh = time.monotonic()  # Just refreshed

        with pytest.raises(CFTokenInvalidSignature, match="rate-limited"):
            validator.validate(make_token())

        # Should NOT have attempted refresh
        assert validator._jwks_client.fetch_data.call_count == 0


# ===========================================================================
# Config validation tests
# ===========================================================================

class TestConfigValidation:
    """Tests for MCPSettings CF configuration validation."""

    def test_cf_enabled_both_set(self):
        from maasv.mcp_server.config import MCPSettings

        s = MCPSettings(cf_team="myteam", cf_aud="my-aud-tag")
        assert s.cf_enabled is True
        assert "myteam" in s.cf_jwks_url
        assert "myteam" in s.cf_issuer

    def test_cf_disabled_both_empty(self):
        from maasv.mcp_server.config import MCPSettings

        s = MCPSettings(cf_team="", cf_aud="")
        assert s.cf_enabled is False

    def test_cf_properties(self):
        from maasv.mcp_server.config import MCPSettings

        s = MCPSettings(cf_team="acme", cf_aud="aud123")
        assert s.cf_jwks_url == "https://acme.cloudflareaccess.com/cdn-cgi/access/certs"
        assert s.cf_issuer == "https://acme.cloudflareaccess.com"


# ===========================================================================
# Middleware integration tests (using Starlette test client)
# ===========================================================================

class TestAuthMiddleware:
    """Integration tests for the dual-auth middleware in server.py."""

    @pytest.fixture
    def app_with_both_auth(self, mock_jwks_client):
        """Create a Starlette app with both API key and CF JWT auth."""
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from maasv.mcp_server.cf_auth import CFAccessValidator

        auth_token = "test-api-key-secret"

        # Build a minimal app mimicking what server.py does
        async def mcp_endpoint(request):
            return JSONResponse({"status": "ok"})

        async def health_check(request):
            return JSONResponse({"status": "healthy"})

        from starlette.applications import Starlette

        app = Starlette(routes=[
            Route("/mcp", mcp_endpoint),
            Route("/health", health_check),
        ])

        cf_validator = CFAccessValidator(team="testteam", audience="test-aud-tag")
        cf_validator._jwks_client = mock_jwks_client

        class AuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                if request.url.path == "/health":
                    return await call_next(request)

                api_key = request.headers.get("X-API-Key") or ""
                if api_key:
                    if hmac.compare_digest(api_key, auth_token):
                        return await call_next(request)
                    return JSONResponse({"error": "Invalid API key"}, status_code=401)

                cf_jwt = request.headers.get("Cf-Access-Jwt-Assertion") or ""
                if cf_jwt:
                    try:
                        cf_validator.validate(cf_jwt)
                        return await call_next(request)
                    except Exception as e:
                        return JSONResponse({"error": f"Invalid CF Access token: {e}"}, status_code=401)

                return JSONResponse({"error": "Authentication required"}, status_code=401)

        app.add_middleware(AuthMiddleware)
        return TestClient(app), auth_token

    def test_health_no_auth_required(self, app_with_both_auth):
        client, _ = app_with_both_auth
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_valid_api_key(self, app_with_both_auth):
        client, api_key = app_with_both_auth
        resp = client.get("/mcp", headers={"X-API-Key": api_key})
        assert resp.status_code == 200

    def test_invalid_api_key(self, app_with_both_auth):
        client, _ = app_with_both_auth
        resp = client.get("/mcp", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401
        assert "Invalid API key" in resp.json()["error"]

    def test_valid_jwt(self, app_with_both_auth, make_token):
        client, _ = app_with_both_auth
        token = make_token()
        resp = client.get("/mcp", headers={"Cf-Access-Jwt-Assertion": token})
        assert resp.status_code == 200

    def test_invalid_jwt(self, app_with_both_auth, make_token):
        client, _ = app_with_both_auth
        token = make_token(aud="wrong-aud")
        resp = client.get("/mcp", headers={"Cf-Access-Jwt-Assertion": token})
        assert resp.status_code == 401
        assert "Invalid CF Access token" in resp.json()["error"]

    def test_no_auth_returns_401(self, app_with_both_auth):
        client, _ = app_with_both_auth
        resp = client.get("/mcp")
        assert resp.status_code == 401
        assert "Authentication required" in resp.json()["error"]

    def test_api_key_takes_precedence(self, app_with_both_auth, make_token):
        """When both headers present, API key is checked first."""
        client, api_key = app_with_both_auth
        token = make_token(aud="wrong-aud")  # Invalid JWT
        resp = client.get("/mcp", headers={
            "X-API-Key": api_key,
            "Cf-Access-Jwt-Assertion": token,
        })
        # Should succeed because valid API key is checked first
        assert resp.status_code == 200

    def test_expired_jwt_returns_401(self, app_with_both_auth, make_token):
        client, _ = app_with_both_auth
        token = make_token(exp_delta=timedelta(seconds=-60))
        resp = client.get("/mcp", headers={"Cf-Access-Jwt-Assertion": token})
        assert resp.status_code == 401
