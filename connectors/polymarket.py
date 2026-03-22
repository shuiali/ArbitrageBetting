"""
connectors/polymarket.py

Polymarket Gamma API connector.

Fetches events (with nested markets) from the Gamma API, which provides
richer event metadata than the CLOB API and is fully public (no auth).

Key facts:
  - Base URL: https://gamma-api.polymarket.com/events
  - Prices: floats in [0, 1] already (probability space)
  - Volume: USD amount as a float string
  - Pagination: offset-based; stop when response list is empty
"""

import json
import logging
import asyncio
import time
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from config import settings
from connectors.base import BaseConnector
from models.market import Platform, UnifiedMarket
from utils.http import get_session, fetch_json
from storage.market_cache import load_markets_with_status, save_markets

logger = logging.getLogger(__name__)

_GAMMA_URL = "https://gamma-api.polymarket.com/events"


class _AsyncWindowRateLimiter:
    """Simple sliding-window limiter for async request bursts."""

    def __init__(self, limit: int, window_seconds: float) -> None:
        self._limit = max(1, limit)
        self._window = max(0.1, window_seconds)
        self._hits: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._hits and (now - self._hits[0]) >= self._window:
                    self._hits.popleft()

                if len(self._hits) < self._limit:
                    self._hits.append(now)
                    return

                wait_for = self._window - (now - self._hits[0]) + 0.001
            await asyncio.sleep(wait_for)


class PolymarketConnector(BaseConnector):
    def platform_name(self) -> str:
        return "polymarket"

    # ── Public API ───────────────────────────────────────────────

    async def fetch_all_with_counts(self) -> Tuple[List[UnifiedMarket], int]:
        """
        Fetch all open markets and return (markets, event_count).
        Uses strict cache policy and near-limit concurrent Gamma event pagination.
        """
        cache = load_markets_with_status(Platform.POLYMARKET, allow_stale=False)
        if cache.markets is not None:
            age_min = (cache.age_seconds or 0.0) / 60.0
            logger.info(
                "Using %s Polymarket cache (%d markets, %.1fm old)",
                cache.status, len(cache.markets), age_min,
            )
            return cache.markets, self._event_count(cache.markets)

        logger.info("Polymarket cache status=%s — fetching live.", cache.status)

        proxy = settings.polymarket_proxy or None
        verify_ssl = proxy is None

        page_size = max(10, min(500, settings.polymarket_events_page_size))
        workers = max(1, settings.polymarket_events_parallel_pages)
        request_budget = max(1, settings.polymarket_events_rate_limit_per_10s)

        limiter = _AsyncWindowRateLimiter(request_budget, 10.0)
        stride = workers * page_size

        event_count = 0
        stop_offset: Optional[int] = None
        seen_event_ids: Set[str] = set()
        parsed_markets: List[UnifiedMarket] = []
        fetch_errors = 0
        state_lock = asyncio.Lock()

        async with get_session(verify_ssl=verify_ssl) as session:
            async def _worker(start_offset: int) -> None:
                nonlocal stop_offset, fetch_errors, event_count
                offset = start_offset
                while True:
                    if stop_offset is not None and offset >= stop_offset:
                        return

                    await limiter.acquire()
                    params = {
                        "active": "true",
                        "closed": "false",
                        "limit": str(page_size),
                        "offset": str(offset),
                        "order": "volume",
                        "ascending": "false",
                    }

                    events = await fetch_json(
                        session,
                        url=_GAMMA_URL,
                        params=params,
                        label="Polymarket Events",
                        proxy=proxy,
                        verify_ssl=verify_ssl,
                    )
                    if events is None:
                        fetch_errors += 1
                        # If proxy/auth/network fails on one page, stop quickly so we can
                        # use partial snapshot or stale cache instead of waiting on all retries.
                        async with state_lock:
                            if stop_offset is None:
                                stop_offset = offset
                        return
                    if not isinstance(events, list):
                        fetch_errors += 1
                        logger.warning("Polymarket Events returned non-list payload at offset=%d", offset)
                        return
                    if not events:
                        async with state_lock:
                            if stop_offset is None or offset < stop_offset:
                                stop_offset = offset
                        return

                    local_event_ids: Set[str] = set()
                    local_markets: List[UnifiedMarket] = []
                    for event in events:
                        event_id = str(event.get("id") or "")
                        if event_id:
                            local_event_ids.add(event_id)

                        event_mkts: List[Dict] = event.get("markets", [])
                        event_market_count = len(event_mkts)
                        for market in event_mkts:
                            parsed = self._parse_market(event, market, event_market_count)
                            if parsed and parsed.is_valid():
                                local_markets.append(parsed)

                    async with state_lock:
                        new_ids = local_event_ids - seen_event_ids
                        seen_event_ids.update(local_event_ids)
                        event_count += len(new_ids)
                        parsed_markets.extend(local_markets)

                    if len(events) < page_size:
                        edge = offset + len(events)
                        async with state_lock:
                            if stop_offset is None or edge < stop_offset:
                                stop_offset = edge
                        return

                    offset += stride

            tasks = [_worker(i * page_size) for i in range(workers)]
            await asyncio.gather(*tasks)

        deduped: "OrderedDict[str, UnifiedMarket]" = OrderedDict()
        for market in parsed_markets:
            if market.market_id not in deduped:
                deduped[market.market_id] = market
        markets_parsed = list(deduped.values())

        if fetch_errors == 0:
            save_markets(
                Platform.POLYMARKET,
                markets_parsed,
                source_ok=True,
                note="live_fetch",
            )
        else:
            logger.warning(
                "Polymarket fetch had %d page errors; skipping cache overwrite.",
                fetch_errors,
            )

        if not markets_parsed:
            stale = load_markets_with_status(Platform.POLYMARKET, allow_stale=True)
            if stale.markets is not None and stale.status == "stale_fallback":
                age_min = (stale.age_seconds or 0.0) / 60.0
                logger.warning(
                    "Polymarket live fetch empty; using stale cache (%d markets, %.1fm old).",
                    len(stale.markets), age_min,
                )
                return stale.markets, self._event_count(stale.markets)

        if fetch_errors > 0 and markets_parsed:
            logger.warning(
                "Polymarket using partial live snapshot (%d markets) due %d page errors.",
                len(markets_parsed), fetch_errors,
            )

        logger.info(
            "Polymarket: fetched %d events → %d valid markets (workers=%d, budget=%d/10s)",
            event_count, len(markets_parsed), workers, request_budget,
        )
        return markets_parsed, event_count

    async def fetch_all(self) -> List[UnifiedMarket]:
        markets, _ = await self.fetch_all_with_counts()
        return markets

    @staticmethod
    def _event_count(markets: List[UnifiedMarket]) -> int:
        return len({m.event_id for m in markets if m.event_id})

    # ── Parsing ──────────────────────────────────────────────────

    def _parse_market(
        self,
        event: Dict[str, Any],
        market: Dict[str, Any],
        event_market_count: int = 1,
    ) -> "Optional[UnifiedMarket]":
        try:
            if market.get("closed"):
                return None
            if market.get("active") is False:
                return None

            market_id: str = (
                market.get("conditionId")
                or market.get("id")
                or ""
            )
            if not market_id:
                return None

            # ── Title ─────────────────────────────────────────────
            event_title: str = (event.get("title") or "").strip()
            market_question: str = (market.get("question") or "").strip()

            # If there's only one market in the event, the event title
            # and market question usually say the same thing. Avoid
            # duplication like "Biden wins — Will Biden win?"
            if (
                market_question
                and market_question.lower() != event_title.lower()
                and event_market_count > 1
            ):
                title = f"{event_title} — {market_question}"
            else:
                title = event_title or market_question

            if not title:
                return None

            # ── Prices ────────────────────────────────────────────
            yes_price, no_price = self._extract_prices(market)

            # ── Volume ────────────────────────────────────────────
            volume_usd = float(market.get("volume") or 0)

            # ── URL ───────────────────────────────────────────────
            slug = event.get("slug") or market.get("slug") or ""
            url = f"https://polymarket.com/event/{slug}" if slug else "https://polymarket.com/"

            # ── Outcome count ─────────────────────────────────────
            outcomes_raw = market.get("outcomes", "[]")
            outcomes = self._parse_json_list(outcomes_raw)
            outcome_count = (
                event_market_count if event_market_count > 1
                else max(2, len(outcomes))
            )

            return UnifiedMarket(
                platform=Platform.POLYMARKET,
                market_id=market_id,
                title=title,
                yes_price=yes_price,
                no_price=no_price,
                volume_usd=volume_usd,
                url=url,
                outcome_count=outcome_count,
                event_id=str(event.get("id", "")),
                event_title=event_title,
            )

        except Exception as exc:
            logger.debug(
                "Failed to parse Polymarket market id=%s: %s",
                market.get("conditionId", "?"), exc,
            )
            return None

    @staticmethod
    def _parse_json_list(value: Any) -> List:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                pass
        return []

    def _extract_prices(self, market: Dict[str, Any]) -> Tuple[float, float]:
        """
        Extract YES and NO prices from market data.

        Polymarket Gamma API provides:
          outcomes:      '["Yes", "No"]'  or  '["Biden", "Trump", ...]'
          outcomePrices: '["0.65", "0.35"]'

        Prices are already in [0, 1] probability space.
        """
        prices_raw = self._parse_json_list(market.get("outcomePrices", "[]"))
        outcomes_raw = self._parse_json_list(market.get("outcomes", "[]"))

        yes_price = 0.0
        no_price = 0.0

        if outcomes_raw and prices_raw:
            for i, outcome in enumerate(outcomes_raw):
                if i >= len(prices_raw):
                    break
                label = str(outcome).strip().upper()
                try:
                    price = float(prices_raw[i])
                except (ValueError, TypeError):
                    continue

                if "YES" in label:
                    yes_price = price
                elif "NO" in label:
                    no_price = price

        # Fallback: if no YES/NO labels, use first two prices
        if yes_price == 0.0 and no_price == 0.0 and len(prices_raw) >= 2:
            try:
                yes_price = float(prices_raw[0])
                no_price = float(prices_raw[1])
            except (ValueError, TypeError):
                pass

        # Infer missing leg
        if yes_price > 0 and no_price == 0:
            no_price = max(0.001, 1.0 - yes_price)
        elif no_price > 0 and yes_price == 0:
            yes_price = max(0.001, 1.0 - no_price)

        # Only clamp prices if at least one has real data
        # If both are 0, keep them at 0 so has_price will be False
        if yes_price > 0 or no_price > 0:
            yes_price = max(0.001, min(0.999, yes_price))
            no_price = max(0.001, min(0.999, no_price))

        return yes_price, no_price
