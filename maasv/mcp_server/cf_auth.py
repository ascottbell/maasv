"""Cloudflare Access JWT validation for MCP Server Portal auth.

CF MCP Server Portals handle OAuth 2.1 for clients like ChatGPT, then forward
requests with a CF Access JWT in the `Cf-Access-Jwt-Assertion` header. This
module validates those JWTs against CF's JWKS endpoint.
"""

import logging
import time

import jwt
from jwt import PyJWKClient

logger = logging.getLogger("maasv.mcp.cf_auth")


class CFAuthError(Exception):
    """Base error for CF Access JWT validation failures."""


class CFTokenExpired(CFAuthError):
    """JWT has expired."""


class CFTokenInvalidAudience(CFAuthError):
    """JWT audience doesn't match our application."""


class CFTokenInvalidIssuer(CFAuthError):
    """JWT issuer doesn't match our team."""


class CFTokenInvalidSignature(CFAuthError):
    """JWT signature verification failed."""


class CFTokenMalformed(CFAuthError):
    """JWT is malformed or unparseable."""


class CFJWKSUnavailable(CFAuthError):
    """JWKS endpoint is unreachable or returned invalid data."""


class CFAccessValidator:
    """Validates Cloudflare Access JWTs using RS256 + JWKS.

    Args:
        team: CF Zero Trust team name (e.g., "mycompany")
        audience: CF Access application audience tag
        jwks_url: Override JWKS URL (for testing)
        issuer: Override issuer (for testing)
    """

    # Rate limit JWKS refreshes: max 1 refresh per this many seconds
    _JWKS_REFRESH_COOLDOWN = 5.0

    def __init__(
        self,
        team: str,
        audience: str,
        jwks_url: str | None = None,
        issuer: str | None = None,
    ):
        self._audience = audience
        self._issuer = issuer or f"https://{team}.cloudflareaccess.com"
        jwks_endpoint = jwks_url or f"https://{team}.cloudflareaccess.com/cdn-cgi/access/certs"

        # PyJWKClient handles caching with lifespan (5 min default)
        self._jwks_client = PyJWKClient(jwks_endpoint, lifespan=300)
        self._last_refresh: float = 0.0

    def validate(self, token: str) -> dict:
        """Validate a CF Access JWT and return decoded claims.

        Args:
            token: Raw JWT string from Cf-Access-Jwt-Assertion header

        Returns:
            Decoded JWT claims dict

        Raises:
            CFTokenExpired: Token has expired
            CFTokenInvalidAudience: Wrong audience
            CFTokenInvalidIssuer: Wrong issuer
            CFTokenInvalidSignature: Signature verification failed
            CFTokenMalformed: Can't parse the token
            CFJWKSUnavailable: Can't reach JWKS endpoint
        """
        if not token or not token.strip():
            raise CFTokenMalformed("Empty token")

        # Get the signing key from JWKS
        signing_key = self._get_signing_key(token)

        try:
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self._audience,
                issuer=self._issuer,
                options={"require": ["exp", "iss", "aud"]},
            )
            return claims
        except jwt.ExpiredSignatureError as e:
            raise CFTokenExpired("Token has expired") from e
        except jwt.InvalidAudienceError as e:
            raise CFTokenInvalidAudience(f"Invalid audience: expected {self._audience}") from e
        except jwt.InvalidIssuerError as e:
            raise CFTokenInvalidIssuer(f"Invalid issuer: expected {self._issuer}") from e
        except jwt.InvalidSignatureError as e:
            raise CFTokenInvalidSignature("Signature verification failed") from e
        except jwt.DecodeError as e:
            raise CFTokenMalformed(f"Malformed token: {e}") from e
        except jwt.InvalidTokenError as e:
            raise CFAuthError(f"Token validation failed: {e}") from e

    def _get_signing_key(self, token: str):
        """Get signing key from JWKS, with retry on kid miss.

        If the kid isn't found in the cached JWKS, forces one refresh
        (rate-limited to avoid hammering the endpoint).
        """
        try:
            return self._jwks_client.get_signing_key_from_jwt(token)
        except jwt.PyJWKClientConnectionError as e:
            raise CFJWKSUnavailable(f"JWKS endpoint unreachable: {e}") from e
        except jwt.DecodeError as e:
            raise CFTokenMalformed(f"Cannot parse token header: {e}") from e
        except jwt.PyJWKClientError:
            # kid not found in cache — try one refresh if cooldown allows
            pass

        now = time.monotonic()
        if now - self._last_refresh < self._JWKS_REFRESH_COOLDOWN:
            raise CFTokenInvalidSignature("Signing key not found (refresh rate-limited)")

        logger.info("JWKS kid miss — refreshing key set")
        self._last_refresh = now

        try:
            self._jwks_client.fetch_data()
            return self._jwks_client.get_signing_key_from_jwt(token)
        except jwt.PyJWKClientConnectionError as e:
            raise CFJWKSUnavailable(f"JWKS endpoint unreachable on refresh: {e}") from e
        except jwt.PyJWKClientError as e:
            raise CFTokenInvalidSignature(f"Signing key not found after refresh: {e}") from e
        except jwt.DecodeError as e:
            raise CFTokenMalformed(f"Cannot parse token header: {e}") from e
