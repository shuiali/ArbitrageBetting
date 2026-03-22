"""
utils/kalshi_auth.py

RSA-PS256 request signing for Kalshi Trade API v2.

Three headers required on every authenticated request:
  KALSHI-ACCESS-KEY:       <key_id UUID>
  KALSHI-ACCESS-TIMESTAMP: <unix milliseconds as string>
  KALSHI-ACCESS-SIGNATURE: <base64(RSA-PS256(timestamp + METHOD.upper() + path))>

The `path` is only the URL path (e.g. /trade-api/v2/events),
NOT the full URL and NOT including the query string.

Usage:
    from utils.kalshi_auth import get_auth_headers
    headers = get_auth_headers("GET", "/trade-api/v2/events")
"""

import base64
import logging
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazily loaded signer — (key_id, private_key) tuple or None if unconfigured
_signer: Optional[Tuple[str, object]] = None
_signer_loaded: bool = False


def _load_signer() -> Optional[Tuple[str, object]]:
    """Load the RSA private key once and cache it. Returns None if not configured."""
    global _signer, _signer_loaded
    if _signer_loaded:
        return _signer

    _signer_loaded = True

    # Import settings here to avoid circular imports at module load time
    from config import settings

    key_path = settings.kalshi_private_key_path
    key_id = settings.kalshi_api_key_id

    if not key_path or not key_id:
        logger.info(
            "Kalshi API credentials not configured "
            "(KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH). "
            "Requests will be sent without authentication — this will fail on production."
        )
        return None

    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.serialization import load_pem_private_key

        with open(key_path, "rb") as fh:
            pem_data = fh.read()

        # Support PEM files that contain the raw key text (no PEM header)
        if not pem_data.strip().startswith(b"-----"):
            # Wrap bare base64 in a PEM header
            pem_data = b"-----BEGIN RSA PRIVATE KEY-----\n" + pem_data.strip() + b"\n-----END RSA PRIVATE KEY-----\n"

        private_key = load_pem_private_key(pem_data, password=None)
        _signer = (key_id, private_key)
        logger.info("Kalshi RSA signer loaded (key_id=%s)", key_id)
        return _signer

    except FileNotFoundError:
        logger.error(
            "Kalshi private key file not found: %s. "
            "Set KALSHI_PRIVATE_KEY_PATH in .env to the correct path.",
            key_path,
        )
    except Exception as exc:
        logger.error("Failed to load Kalshi private key from %s: %s", key_path, exc)

    return None


def get_auth_headers(method: str, path: str) -> Dict[str, str]:
    """
    Generate the three RSA-PS256 auth headers for a Kalshi API request.

    Args:
        method: HTTP method ("GET", "POST", etc.)
        path:   URL path only — e.g. "/trade-api/v2/events".
                Do NOT include the host or query string.

    Returns:
        Dict of headers to merge into the request.
        Empty dict if credentials are not configured (unauthenticated).
    """
    signer = _load_signer()
    if signer is None:
        return {}

    key_id, private_key = signer

    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        timestamp_ms = str(int(time.time() * 1000))
        message = (timestamp_ms + method.upper() + path).encode("utf-8")

        signature_bytes = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature_bytes).decode("ascii"),
        }

    except Exception as exc:
        logger.error("Failed to sign Kalshi request (%s %s): %s", method, path, exc)
        return {}
