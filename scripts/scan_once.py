#!/usr/bin/env python3
"""
scripts/scan_once.py

Run one full fetch → match → arb scan and print results.
No Telegram alerts. Use this to verify everything works end-to-end
before starting the live bot.

Usage:
    python scripts/scan_once.py
    python scripts/scan_once.py --min-roi 0.0    # show all spreads (even losers)
    python scripts/scan_once.py --min-roi 0.05   # only show ≥5% net ROI
"""

import argparse
import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

_COL = {
    "RESET":  "\033[0m",
    "BOLD":   "\033[1m",
    "GREEN":  "\033[92m",
    "YELLOW": "\033[93m",
    "RED":    "\033[91m",
    "DIM":    "\033[2m",
}


def _roi_color(roi: float) -> str:
    if roi >= 0.05:
        return _COL["GREEN"]
    if roi >= 0.02:
        return _COL["YELLOW"]
    return _COL["RED"]


async def main(min_roi: float) -> None:
    from arbitrage.engine import ArbEngine
    from connectors.kalshi import KalshiConnector
    from connectors.polymarket import PolymarketConnector
    from config import settings
    from matching.embedder import Embedder
    from matching.matcher import Matcher
    from storage.market_cache import load_markets_with_status
    from models.market import Platform

    print(f"\n  {_COL['BOLD']}Arb Bot — One-Shot Scan{_COL['RESET']}")
    print("  " + "─" * 44)
    print(f"  Min ROI filter : {min_roi * 100:.2f}%")
    print(f"  Fees           : poly {settings.polymarket_fee_rate*100:.0f}% "
          f"+ kalshi {settings.kalshi_fee_rate*100:.0f}% "
          f"= {(settings.polymarket_fee_rate + settings.kalshi_fee_rate)*100:.0f}% total")
    print(f"  Min gross spread needed: "
          f"{(settings.polymarket_fee_rate + settings.kalshi_fee_rate + min_roi)*100:.0f}%")

    t0 = time.time()

    # ── Step 1: Fetch ─────────────────────────────────────────────
    print("\n  [1/3] Fetching markets…")
    poly_conn   = PolymarketConnector()
    kalshi_conn = KalshiConnector()
    poly_markets, kalshi_markets = await asyncio.gather(
        poly_conn.fetch_all(),
        kalshi_conn.fetch_all(),
    )
    t1 = time.time()
    print(f"        Polymarket : {len(poly_markets)} markets")
    print(f"        Kalshi     : {len(kalshi_markets)} markets")
    print(f"        Fetch time : {t1 - t0:.1f}s")

    if not poly_markets or not kalshi_markets:
        poly_status = load_markets_with_status(Platform.POLYMARKET, allow_stale=True)
        kalshi_status = load_markets_with_status(Platform.KALSHI, allow_stale=True)
        print("\n  ✗ One or both connectors returned 0 markets.")
        print(f"    Polymarket cache status: {poly_status.status}")
        print(f"    Kalshi cache status    : {kalshi_status.status}")
        print("    Check credentials, proxy, and API reachability.")
        return

    # ── Step 2: Match ─────────────────────────────────────────────
    print("\n  [2/3] Running semantic match (BGE-M3)…")
    embedder = Embedder()
    matcher  = Matcher(embedder)
    groups   = matcher.match(poly_markets, kalshi_markets)
    t2 = time.time()
    print(f"        Match groups : {len(groups)}")
    print(f"        Match time   : {t2 - t1:.1f}s")

    if not groups:
        print("\n  No match groups found. Lower MIN_SIMILARITY in .env and retry.")
        return

    # ── Step 3: Arb scan ─────────────────────────────────────────
    print("\n  [3/3] Scanning for opportunities…")
    engine = ArbEngine()
    # Pass min_roi directly — no need to hack private attributes
    opportunities = engine.find_opportunities(groups, min_roi=min_roi)
    t3 = time.time()

    print(f"        Opportunities : {len(opportunities)}")
    print(f"        Scan time     : {t3 - t2:.1f}s")
    print(f"        Total time    : {t3 - t0:.1f}s")

    # ── Results ───────────────────────────────────────────────────
    total_fees = settings.polymarket_fee_rate + settings.kalshi_fee_rate

    if not opportunities:
        print(f"\n  No opportunities above {min_roi * 100:.1f}% net ROI.")
        print(f"  (Total fees: {total_fees * 100:.0f}%  —  need gross spread > "
              f"{(total_fees + min_roi) * 100:.0f}%)")
        print(f"  Try --min-roi 0.0 to see all spreads.\n")
        return

    print(f"\n{'═' * 67}")
    print(f"  {'ARBITRAGE OPPORTUNITIES':^63}")
    print(f"{'═' * 67}")

    for i, opp in enumerate(opportunities, 1):
        roi_c = _roi_color(opp.net_roi)
        buy_outcome = opp.outcome_text_for(opp.buy_platform, opp.buy_side)
        hedge_outcome = opp.outcome_text_for(opp.sell_platform, opp.hedge_side)
        print(
            f"\n  #{i}  {roi_c}{_COL['BOLD']}Net ROI: +{opp.net_roi_pct:.2f}%{_COL['RESET']}"
            f"  (gross {opp.gross_spread_pct:.2f}% − fees {opp.total_fees * 100:.1f}%)"
        )
        print(f"       Poly:  {opp.poly_market.title[:76]}")
        print(f"       Kals:  {opp.kalshi_market.title[:76]}")
        print(
            f"       Action: BUY {opp.buy_side} on {opp.buy_platform.value.upper()}"
            f" @ {opp.buy_price:.4f}  |  BUY {opp.hedge_side} on"
            f" {opp.sell_platform.value.upper()} @ {opp.hedge_price:.4f}"
        )
        print(f"       Bet 1: {buy_outcome[:90]}")
        print(f"       Bet 2: {hedge_outcome[:90]}")
        print(
            f"       Sim: {opp.similarity_score:.4f}"
            f"  |  Poly: {opp.poly_market.url}"
        )
        print(f"       {_COL['DIM']}{opp.kalshi_market.url}{_COL['RESET']}")

    print(f"\n{'═' * 67}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-shot arb scan.")
    parser.add_argument(
        "--min-roi", type=float, default=None,
        help="Minimum net ROI to display (default: MIN_NET_ROI from .env)",
    )
    args = parser.parse_args()

    from config import settings
    min_roi = args.min_roi if args.min_roi is not None else settings.min_net_roi
    asyncio.run(main(min_roi=min_roi))
