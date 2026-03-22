#!/usr/bin/env python3
"""
scripts/test_connectors.py

Validates that both connectors can reach their APIs and fetch ALL markets.

Usage:
    python scripts/test_connectors.py
    python scripts/test_connectors.py --platform polymarket
    python scripts/test_connectors.py --platform kalshi
"""

import argparse
import asyncio
import logging
import sys
import os

# Make imports work from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.kalshi import KalshiConnector
from connectors.polymarket import PolymarketConnector
from models.market import Platform, UnifiedMarket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def _print_report(markets: list[UnifiedMarket], platform: str, n: int = 5) -> None:
    print(f"\n{'═'*70}")
    print(f"  {platform.upper()} — {len(markets)} markets fetched")
    print(f"{'═'*70}")

    # Breakdown
    with_price = [m for m in markets if m.has_price]
    with_volume = [m for m in markets if m.volume_usd > 0]
    binary = [m for m in markets if m.outcome_count == 2]

    print(f"\n  Breakdown:")
    print(f"    Total markets:        {len(markets)}")
    print(f"    With prices:          {len(with_price)}")
    print(f"    With volume > 0:      {len(with_volume)}")
    print(f"    Binary (2 outcomes):  {len(binary)}")
    print(f"    Multi-outcome:        {len(markets) - len(binary)}")

    # Price stats
    if with_price:
        yes_prices = [m.yes_price for m in with_price]
        sums = [m.implied_sum for m in with_price if m.implied_sum > 0]
        print(f"\n  Price stats (markets with prices):")
        print(f"    YES range:  {min(yes_prices):.4f} – {max(yes_prices):.4f}")
        if sums:
            import statistics
            print(f"    Implied sum:  min={min(sums):.4f}  median={statistics.median(sums):.4f}  max={max(sums):.4f}")
        overround = [s for s in sums if s > 1.05]
        print(f"    Overround (sum > 1.05): {len(overround)}")

    # Volume stats
    if with_volume:
        vols = sorted([m.volume_usd for m in with_volume], reverse=True)
        print(f"\n  Volume stats:")
        print(f"    Top 5 volumes: {', '.join(f'${v:,.0f}' for v in vols[:5])}")
        print(f"    Total volume:  ${sum(vols):,.0f}")

    # Sample markets
    print(f"\n  Sample markets (first {min(n, len(markets))}):\n")
    for m in markets[:n]:
        price_str = f"YES={m.yes_price:.4f}  NO={m.no_price:.4f}" if m.has_price else "NO PRICE"
        print(f"  [{m.market_id[:30]:30s}] {price_str}  Vol=${m.volume_usd:>12,.0f}")
        print(f"    {m.title[:90]}")
        print()

    # Top-volume markets
    if with_volume:
        sorted_by_vol = sorted(with_volume, key=lambda m: m.volume_usd, reverse=True)
        print(f"  Top {min(5, len(sorted_by_vol))} by volume:\n")
        for m in sorted_by_vol[:5]:
            print(f"  [{m.market_id[:30]:30s}] YES={m.yes_price:.4f}  Vol=${m.volume_usd:>12,.0f}")
            print(f"    {m.title[:90]}")
            print()


async def test_polymarket() -> tuple[list[UnifiedMarket], int]:
    logger.info("Testing Polymarket connector…")
    conn = PolymarketConnector()
    try:
        markets, event_count = await conn.fetch_all_with_counts()
        if not markets:
            print("\n  ⚠ Polymarket returned 0 markets. Check your network / proxy.")
        else:
            print(f"\n  ✓ Polymarket connector OK — {event_count} events, {len(markets)} market outcomes")
        _print_report(markets, "polymarket")
        return markets, event_count
    except Exception as exc:
        logger.error("Polymarket connector failed: %s", exc, exc_info=True)
        return [], 0


async def test_kalshi() -> tuple[list[UnifiedMarket], int]:
    logger.info("Testing Kalshi connector…")
    conn = KalshiConnector()
    if not conn._ready:
        print("\n  ⚠ Kalshi signer not configured — will attempt unauthenticated.")
        print("    Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH in your .env")
        print("    to use production. For demo, set KALSHI_ENV=demo.\n")
    try:
        markets, event_count = await conn.fetch_all_with_counts()
        if not markets:
            print("\n  ⚠ Kalshi returned 0 markets. Check credentials / environment.")
        else:
            print(f"\n  ✓ Kalshi connector OK — {event_count} events, {len(markets)} market outcomes")
        _print_report(markets, "kalshi")
        return markets, event_count
    except Exception as exc:
        logger.error("Kalshi connector failed: %s", exc, exc_info=True)
        return [], 0


async def main(platform: str | None) -> None:
    print("\n  Arb Bot — Connector Test")
    print("  " + "─" * 40)

    poly_markets, poly_events = [], 0
    kalshi_markets, kalshi_events = [], 0

    if platform in (None, "polymarket"):
        poly_markets, poly_events = await test_polymarket()
    if platform in (None, "kalshi"):
        kalshi_markets, kalshi_events = await test_kalshi()

    # Summary
    total_events = poly_events + kalshi_events
    total_markets = len(poly_markets) + len(kalshi_markets)
    
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Polymarket:  {poly_events:>6} events  |  {len(poly_markets):>6} market outcomes")
    print(f"  Kalshi:      {kalshi_events:>6} events  |  {len(kalshi_markets):>6} market outcomes")
    print("─" * 70)
    print(f"  TOTAL:       {total_events:>6} events  |  {total_markets:>6} market outcomes")
    print("=" * 70 + "\n")

    print("\n  Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test exchange connectors.")
    parser.add_argument(
        "--platform",
        choices=["polymarket", "kalshi"],
        default=None,
        help="Test only one platform (default: both)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.platform))
