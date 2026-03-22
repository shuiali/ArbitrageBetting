"""
utils/http.py

Shared async HTTP helper with exponential-backoff retry.
Used by both connectors so retry logic lives in one place.

Usage:
    async with get_session() as session:
        data = await fetch_json(session, url, params=params, headers=headers)
"""

import asyncio
import logging
import ssl
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)

_TIMEOUT = aiohttp.ClientTimeout(total=30)
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 4
_BACKOFF_BASE = 1.5  # seconds: 1.5 → 2.25 → 3.375 → 5.06


def _make_ssl(verify: bool) -> "ssl.SSLContext | bool":
    """
    Return an SSL context or False.

    aiohttp's ssl parameter accepts:
      - None / True  → use default verification (valid)
      - False        → skip verification entirely
      - SSLContext   → custom context

    Passing the Python bool True directly to session.get(ssl=...) is
    technically valid in newer aiohttp but semantically ambiguous.
    We always build an explicit context when disabling, and pass None
    when using defaults — this is unambiguous across all aiohttp versions.
    """
    if not verify:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return True  # aiohttp default — full verification


@asynccontextmanager
async def get_session(
    headers: Optional[Dict[str, str]] = None,
    proxy: Optional[str] = None,
    verify_ssl: bool = True,
) -> AsyncIterator[aiohttp.ClientSession]:
    """
    Yield a configured aiohttp session. Always use as `async with`.

    Args:
        headers:    Optional default headers added to every request.
        proxy:      Proxy URL (stored on session; pass to fetch_json too).
        verify_ssl: Set False to skip certificate verification (e.g. when
                    routing through an intercepting proxy).
    """
    ssl_ctx = _make_ssl(verify_ssl)
    # Pass ssl_context to TCPConnector so it applies to all connections
    connector = aiohttp.TCPConnector(ssl=ssl_ctx)

    async with aiohttp.ClientSession(
        timeout=_TIMEOUT,
        headers=headers or {},
        connector=connector,
    ) as session:
        yield session


async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
    label: str = "",
    proxy: Optional[str] = None,
    verify_ssl: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    GET `url` and return parsed JSON, or None on unrecoverable failure.
    Retries on transient errors with exponential backoff.

    Note: verify_ssl here applies only as a per-request override if you
    bypass get_session. When using get_session(), ssl is already set on
    the connector — this parameter is kept for backward compatibility.
    """
    # Per-request ssl arg — only used when caller wants to override connector
    # For aiohttp, False disables verification; None means "use connector default"
    req_ssl: Optional[Any] = None  # use connector's setting by default
    if not verify_ssl:
        req_ssl = _make_ssl(False)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            kwargs: Dict[str, Any] = dict(params=params, headers=headers)
            if proxy:
                kwargs["proxy"] = proxy
            if req_ssl is not None:
                kwargs["ssl"] = req_ssl

            async with session.get(url, **kwargs) as resp:
                if resp.status == 200:
                    return await resp.json(content_type=None)

                if resp.status == 401:
                    body = await resp.text()
                    logger.error(
                        "%s HTTP 401 Unauthorized — check API credentials. Body: %s",
                        label, body[:300],
                    )
                    return None  # No point retrying auth failures

                if resp.status == 403:
                    body = await resp.text()
                    logger.error("%s HTTP 403 Forbidden: %s", label, body[:300])
                    return None

                if resp.status == 404:
                    logger.warning("%s HTTP 404 — endpoint not found: %s", label, url)
                    return None

                if resp.status in _RETRY_STATUSES:
                    wait = _BACKOFF_BASE ** attempt
                    logger.warning(
                        "%s HTTP %d (attempt %d/%d) — retrying in %.1fs",
                        label, resp.status, attempt, _MAX_RETRIES, wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                # Other 4xx — not retryable
                body = await resp.text()
                logger.warning(
                    "%s HTTP %d — not retrying. Body: %s",
                    label, resp.status, body[:300],
                )
                return None

        except asyncio.TimeoutError:
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                "%s timeout (attempt %d/%d) — retrying in %.1fs",
                label, attempt, _MAX_RETRIES, wait,
            )
            await asyncio.sleep(wait)

        except aiohttp.ClientResponseError as exc:
            status = int(exc.status or 0)
            if status in {401, 403, 404}:
                logger.error(
                    "%s response error HTTP %d — not retrying: %s",
                    label, status, exc,
                )
                return None

            if status in _RETRY_STATUSES:
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    "%s response error HTTP %d (attempt %d/%d) — retrying in %.1fs",
                    label, status, attempt, _MAX_RETRIES, wait,
                )
                await asyncio.sleep(wait)
                continue

            logger.warning("%s response error HTTP %d — not retrying: %s", label, status, exc)
            return None

        except aiohttp.ClientHttpProxyError as exc:
            logger.error("%s proxy error — not retrying: %s", label, exc)
            return None

        except aiohttp.ClientConnectorError as exc:
            logger.error("%s connection error: %s", label, exc)
            return None

        except aiohttp.ClientError as exc:
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                "%s client error (attempt %d/%d): %s — retrying in %.1fs",
                label, attempt, _MAX_RETRIES, exc, wait,
            )
            await asyncio.sleep(wait)

    logger.error("%s exhausted %d retries on %s", label, _MAX_RETRIES, url)
    return None
