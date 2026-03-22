"""
main.py
───────────────────────────────────────────────────────────────────────
Arbitrage bot entry point.

Loop flow
─────────────────────────────────────────────────────────────────────
Every POLL_INTERVAL_SECONDS:

  1. Fetch all open markets from Polymarket + Kalshi (paginated REST).
  2. If REMATCH_INTERVAL has elapsed (or first run):
       a. Re-embed any new/changed market titles (BGE-M3, cached).
       b. Run full cross-platform cosine similarity matching.
       c. Store new MatchGroup list in StateManager.
  3. Run ArbEngine over current MatchGroups → list of ArbOpportunity.
  4. For each opportunity: if StateManager says we should alert,
       send Telegram message and record the alert timestamp.
  5. Sleep until next cycle.

The match step is the expensive one (~seconds for thousands of markets).
The price-fetch + arb-calc step is cheap (sub-second).
───────────────────────────────────────────────────────────────────────
"""

import asyncio
import logging
import signal
import sys

from alerts.telegram_bot import TelegramBot
from arbitrage.engine import ArbEngine
from config import settings
from connectors.kalshi import KalshiConnector
from connectors.polymarket import PolymarketConnector
from matching.embedder import Embedder
from matching.matcher import Matcher
from storage.state import StateManager

logger = logging.getLogger(__name__)

# Send a Telegram heartbeat every this many cycles (0 = disabled)
_HEARTBEAT_EVERY_N_CYCLES = 10


async def run() -> None:
    logger.info("═" * 60)
    logger.info("  Prediction Market Arbitrage Bot — starting up")
    logger.info("═" * 60)

    # ── Initialise components ──────────────────────────────────
    embedder = Embedder()          # Downloads/loads BGE-M3 on construction
    matcher = Matcher(embedder)
    engine = ArbEngine()
    state = StateManager()
    telegram = TelegramBot()

    poly_conn = PolymarketConnector()
    kalshi_conn = KalshiConnector()

    cycle = 0

    logger.info("Starting main loop (poll every %ds, rematch every %ds)",
                settings.poll_interval_seconds, settings.rematch_interval_seconds)

    while True:
        cycle += 1
        logger.info("── Cycle %d ──────────────────────────────", cycle)

        # 1. Fetch markets from both platforms concurrently
        try:
            poly_markets, kalshi_markets = await asyncio.gather(
                poly_conn.fetch_all(),
                kalshi_conn.fetch_all(),
            )
        except Exception as exc:
            logger.error("Fatal fetch error: %s — retrying next cycle", exc)
            await asyncio.sleep(settings.poll_interval_seconds)
            continue

        state.record_fetch_counts(len(poly_markets), len(kalshi_markets))

        if not poly_markets or not kalshi_markets:
            logger.warning("One or both connectors returned 0 markets — skipping cycle.")
            await asyncio.sleep(settings.poll_interval_seconds)
            continue

        # Send startup Telegram message on first successful cycle
        if cycle == 1:
            await telegram.send_startup_message(len(poly_markets), len(kalshi_markets))

        # 2. Re-match if needed
        if state.needs_rematch():
            logger.info(
                "Running semantic match: %d poly × %d kalshi markets",
                len(poly_markets), len(kalshi_markets),
            )
            try:
                groups = matcher.match(poly_markets, kalshi_markets)
                state.update_match_groups(groups)
            except Exception as exc:
                logger.error("Matcher error: %s", exc, exc_info=True)
                # Keep stale groups rather than crash
        else:
            # Update prices in existing match groups with freshly fetched data
            _refresh_prices(state, poly_markets, kalshi_markets)

        # 3. Scan for opportunities
        groups = state.get_match_groups()
        if not groups:
            logger.info("No match groups yet — waiting for first match cycle.")
            await asyncio.sleep(settings.poll_interval_seconds)
            continue

        opportunities = engine.find_opportunities(groups)

        # 4. Alert new / cooled-down opportunities
        alerted_this_cycle = 0
        for opp in opportunities:
            if state.should_alert(opp):
                success = await telegram.send_alert(opp)
                if success:
                    state.mark_alerted(opp)
                    alerted_this_cycle += 1
                    logger.info(
                        "ALERT sent: group=%s net_roi=%.2f%% buy=%s",
                        opp.group_id, opp.net_roi_pct, opp.buy_platform.value,
                    )

        # Periodic heartbeat
        if _HEARTBEAT_EVERY_N_CYCLES > 0 and cycle % _HEARTBEAT_EVERY_N_CYCLES == 0:
            await telegram.send_cycle_summary(len(groups), len(opportunities), cycle)

        state.purge_stale_alerts()

        logger.info(
            "Cycle %d done: %s | opps=%d alerted=%d",
            cycle, state.summary(), len(opportunities), alerted_this_cycle,
        )

        await asyncio.sleep(settings.poll_interval_seconds)


def _refresh_prices(
    state: StateManager,
    poly_markets: list,
    kalshi_markets: list,
) -> None:
    """
    Update the prices in existing MatchGroup objects using freshly fetched
    market data. This keeps price data current between full re-matches.
    """
    poly_by_id = {m.market_id: m for m in poly_markets}
    kalshi_by_id = {m.market_id: m for m in kalshi_markets}
    updated = 0

    for group in state.get_match_groups():
        pm = poly_by_id.get(group.poly_market.market_id)
        km = kalshi_by_id.get(group.kalshi_market.market_id)
        if pm:
            group.poly_market.yes_price = pm.yes_price
            group.poly_market.no_price = pm.no_price
            group.poly_market.fetched_at = pm.fetched_at
            updated += 1
        if km:
            group.kalshi_market.yes_price = km.yes_price
            group.kalshi_market.no_price = km.no_price
            group.kalshi_market.fetched_at = km.fetched_at

    logger.debug("Refreshed prices for %d match groups.", updated)


def _handle_shutdown(sig, frame) -> None:
    logger.info("Shutdown signal received — exiting.")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)
    asyncio.run(run())
