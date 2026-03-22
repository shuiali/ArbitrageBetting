"""
connectors/kalshi.py

Kalshi Trade API v2 connector.

Key facts about the API:
  - Base URL: from settings.kalshi_base_url
  - Endpoint: GET /events?status=open&with_nested_markets=true&limit=200
  - Auth: RSA-PS256 signed headers (see utils/kalshi_auth.py)
  - Prices: integers in CENTS [0–100] — divide by 100 to get probability
  - Volume: integer count of contracts traded (not USD)
  - Pagination: cursor-based; stop when cursor is empty/absent
"""

import logging
from typing import Any, Dict, List, Tuple

from connectors.base import BaseConnector
from models.market import Platform, UnifiedMarket
from utils.http import get_session, fetch_json
from utils.kalshi_auth import get_auth_headers
from storage.market_cache import load_markets_with_status, save_markets

logger = logging.getLogger(__name__)

# Path used for URL construction and auth signing
_EVENTS_PATH = "/events"


def _to_prob(value: Any) -> float:
    """
    Convert a Kalshi price to probability space [0, 1].

    Supports both legacy integer cents [0,100] and current decimal-dollar
    payloads [0,1] (e.g. "0.1100").
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0

    if v <= 0:
        return 0.0
    if v > 1.0:
        return v / 100.0
    return v


def _first_prob(market: Dict[str, Any], *keys: str) -> float:
    """Return the first parseable non-zero probability from candidate keys."""
    for key in keys:
        value = market.get(key)
        prob = _to_prob(value)
        if prob > 0:
            return prob
    return 0.0


def _first_float(market: Dict[str, Any], *keys: str) -> float:
    """Return the first parseable float from candidate keys."""
    for key in keys:
        value = market.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


class KalshiConnector(BaseConnector):
    def platform_name(self) -> str:
        return "kalshi"

    @property
    def _ready(self) -> bool:
        """True if auth credentials are configured. Public for test_connectors."""
        from config import settings
        return bool(settings.kalshi_api_key_id and settings.kalshi_private_key_path)

    def _base_url(self) -> str:
        from config import settings
        return settings.kalshi_base_url

    def _event_url(self) -> str:
        return self._base_url() + _EVENTS_PATH

    def _auth(self, method: str = "GET") -> Dict[str, str]:
        return get_auth_headers(method, _EVENTS_PATH)

    # ── Public API ───────────────────────────────────────────────

    async def fetch_all_with_counts(self) -> Tuple[List[UnifiedMarket], int]:
        """
        Fetch all open markets and return (markets, event_count).
        Uses strict cache policy and falls back to stale cache on API failures.
        """
        cache = load_markets_with_status(Platform.KALSHI, allow_stale=False)
        if cache.markets is not None:
            if cache.markets and all(not m.has_price for m in cache.markets):
                logger.warning(
                    "Ignoring %s Kalshi cache: snapshot has no price data for all %d markets.",
                    cache.status,
                    len(cache.markets),
                )
            else:
                age_min = (cache.age_seconds or 0.0) / 60.0
                logger.info(
                    "Using %s Kalshi cache (%d markets, %.1fm old)",
                    cache.status, len(cache.markets), age_min,
                )
                return cache.markets, len({m.event_id for m in cache.markets if m.event_id})

        logger.info("Kalshi cache status=%s — fetching live.", cache.status)

        url = self._event_url()
        auth_headers = self._auth("GET")

        markets_parsed: List[UnifiedMarket] = []
        event_count = 0
        cursor = ""
        page_errors = 0

        async with get_session(headers=auth_headers) as session:
            while True:
                params: Dict[str, str] = {
                    "status": "open",
                    "with_nested_markets": "true",
                    "limit": "200",
                }
                if cursor:
                    params["cursor"] = cursor

                data = await fetch_json(
                    session,
                    url=url,
                    params=params,
                    label="Kalshi Events",
                )
                if data is None:
                    page_errors += 1
                    break
                if not data:
                    break

                events: List[Dict] = data.get("events", [])
                event_count += len(events)

                for event in events:
                    event_markets: List[Dict] = event.get("markets", [])
                    event_market_count = len(event_markets)
                    for market in event_markets:
                        parsed = self._parse_market(event, market, event_market_count)
                        if parsed and parsed.is_valid():
                            markets_parsed.append(parsed)

                cursor = data.get("cursor") or ""
                if not cursor:
                    break

        if page_errors == 0:
            save_markets(
                Platform.KALSHI,
                markets_parsed,
                source_ok=True,
                note="live_fetch",
            )
        else:
            logger.warning("Kalshi fetch had %d page errors; skipping cache overwrite.", page_errors)

        if not markets_parsed:
            stale = load_markets_with_status(Platform.KALSHI, allow_stale=True)
            if stale.markets is not None and stale.status == "stale_fallback":
                age_min = (stale.age_seconds or 0.0) / 60.0
                logger.warning(
                    "Kalshi live fetch empty; using stale cache (%d markets, %.1fm old).",
                    len(stale.markets), age_min,
                )
                return stale.markets, len({m.event_id for m in stale.markets if m.event_id})

        logger.info(
            "Kalshi: fetched %d events → %d valid markets",
            event_count, len(markets_parsed),
        )
        return markets_parsed, event_count

    async def fetch_all(self) -> List[UnifiedMarket]:
        markets, _ = await self.fetch_all_with_counts()
        return markets

    # ── Parsing ──────────────────────────────────────────────────

    def _parse_market(
        self,
        event: Dict[str, Any],
        market: Dict[str, Any],
        event_market_count: int = 1,
    ) -> "UnifiedMarket | None":
        try:
            # Filter non-open markets (API filter may not be exhaustive)
            status = market.get("status", "")
            if status not in ("open", "active", "initialized", "unopened", ""):
                return None

            ticker: str = market.get("ticker", "")
            if not ticker:
                return None

            # ── Title ────────────────────────────────────────────
            event_title: str = event.get("title", "").strip()
            market_subtitle: str = market.get("subtitle", "").strip()
            market_title: str = market.get("title", "").strip()

            # Prefer subtitle for the market-level label (it's more concise)
            sub = market_subtitle or market_title
            if sub and sub.lower() not in event_title.lower():
                title = f"{event_title} — {sub}"
            else:
                title = event_title or sub

            if not title:
                return None

            # ── Prices ───────────────────────────────────────────
            # Supports both legacy cent fields and newer *_dollars fields.
            y_bid = _first_prob(market, "yes_bid", "yes_bid_dollars")
            y_ask = _first_prob(market, "yes_ask", "yes_ask_dollars")
            n_bid = _first_prob(market, "no_bid", "no_bid_dollars")
            n_ask = _first_prob(market, "no_ask", "no_ask_dollars")

            if y_bid > 0 and y_ask > 0:
                yes_price = (y_bid + y_ask) / 2.0
            else:
                yes_price = _first_prob(market, "last_price", "last_price_dollars")

            if n_bid > 0 and n_ask > 0:
                no_price = (n_bid + n_ask) / 2.0
            else:
                no_price = 0.0

            # Infer missing leg from complement
            if yes_price > 0 and no_price == 0:
                no_price = max(0.001, 1.0 - yes_price)
            elif no_price > 0 and yes_price == 0:
                yes_price = max(0.001, 1.0 - no_price)

            # Only clamp prices if at least one has real data
            if yes_price > 0 or no_price > 0:
                yes_price = max(0.001, min(0.999, yes_price))
                no_price = max(0.001, min(0.999, no_price))

            # ── Volume (contracts -> approximate USD) ───────────
            volume_contracts = _first_float(market, "volume", "volume_fp")
            volume_usd = volume_contracts * yes_price if yes_price else volume_contracts * 0.5

            # ── URL ──────────────────────────────────────────────
            series_ticker: str = event.get("series_ticker", "")
            market_url = f"https://kalshi.com/markets/{series_ticker}/{ticker}"

            # ── Outcome count ─────────────────────────────────────
            outcome_count = event_market_count if event_market_count > 1 else 2

            return UnifiedMarket(
                platform=Platform.KALSHI,
                market_id=ticker,
                title=title,
                yes_price=yes_price,
                no_price=no_price,
                volume_usd=volume_usd,
                url=market_url,
                outcome_count=outcome_count,
                event_id=event.get("event_ticker", "") or series_ticker,
                event_title=event_title,
            )

        except Exception as exc:
            logger.debug("Failed to parse Kalshi market ticker=%s: %s", market.get("ticker"), exc)
            return None
