#!/usr/bin/env python3
"""
scripts/inspect_matches.py

Fetches live markets from both platforms, runs BGE-M3 semantic matching,
and prints the top matches so you can manually verify:
  - The semantic match actually represents the same real-world event.
  - Outcome alignment is correct (YES↔YES or inverted YES↔NO).

Run this before the first live bot run. Review the output, then adjust
MIN_SIMILARITY in .env.

Usage:
    python scripts/inspect_matches.py
    python scripts/inspect_matches.py --top 100 --save matches.csv
    python scripts/inspect_matches.py --min-sim 0.75     # see more matches
    python scripts/inspect_matches.py --min-sim 0.90     # stricter filter
"""

import argparse
import asyncio
import csv
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connectors.kalshi import KalshiConnector
from connectors.polymarket import PolymarketConnector
from matching.embedder import Embedder
from matching.matcher import Matcher
from models.market import MatchGroup

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
    "CYAN":   "\033[96m",
    "DIM":    "\033[2m",
}


def _sim_color(sim: float) -> str:
    if sim >= 0.92:
        return _COL["GREEN"]
    if sim >= 0.85:
        return _COL["YELLOW"]
    return _COL["RED"]


def _print_group(i: int, g: MatchGroup) -> None:
    sim     = g.similarity_score
    aligned = "YES↔YES ✓" if g.outcomes_aligned else "YES↔NO  ⚠ INVERTED"
    color   = _sim_color(sim)

    p_yes, k_yes = g.get_comparable_prices()
    spread     = abs(p_yes - k_yes)
    spread_str = f"{spread * 100:.2f}%"
    spread_col = _COL["GREEN"] if spread > 0.08 else _COL["RESET"]

    print(
        f"\n{_COL['BOLD']}#{i:>4}  sim={color}{sim:.4f}{_COL['RESET']}"
        f"  align={aligned}  spread={spread_col}{spread_str}{_COL['RESET']}"
    )
    print(f"       POLY  {g.poly_market.title[:90]}")
    print(f"       KALS  {g.kalshi_market.title[:90]}")
    print(
        f"       {_COL['DIM']}P-YES={p_yes:.3f}  K-equiv={k_yes:.3f}"
        f"  P-vol=${g.poly_market.volume_usd:,.0f}"
        f"  K-vol=${g.kalshi_market.volume_usd:,.0f}"
        f"  group={g.group_id}{_COL['RESET']}"
    )


def _save_csv(groups: list, path: str) -> None:
    fields = [
        "rank", "similarity", "outcomes_aligned",
        "poly_title", "kalshi_title",
        "poly_yes", "kalshi_equiv_yes", "spread_pct",
        "poly_vol_usd", "kalshi_vol_usd",
        "poly_url", "kalshi_url", "group_id",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, g in enumerate(groups, 1):
            p_yes, k_yes = g.get_comparable_prices()
            writer.writerow({
                "rank":             i,
                "similarity":       round(g.similarity_score, 5),
                "outcomes_aligned": g.outcomes_aligned,
                "poly_title":       g.poly_market.title,
                "kalshi_title":     g.kalshi_market.title,
                "poly_yes":         round(p_yes, 4),
                "kalshi_equiv_yes": round(k_yes, 4),
                "spread_pct":       round(abs(p_yes - k_yes) * 100, 3),
                "poly_vol_usd":     round(g.poly_market.volume_usd, 2),
                "kalshi_vol_usd":   round(g.kalshi_market.volume_usd, 2),
                "poly_url":         g.poly_market.url,
                "kalshi_url":       g.kalshi_market.url,
                "group_id":         g.group_id,
            })
    print(f"\n  Saved {len(groups)} matches → {path}")


async def main(top: int, min_sim: float, save_path: str | None) -> None:
    print("\n  Arb Bot — Match Inspector")
    print("  " + "─" * 44)
    print(f"  Similarity threshold : {min_sim}")
    print(f"  Showing top          : {top}\n")

    # ── Fetch ─────────────────────────────────────────────────────
    print("  [1/3] Fetching markets from both platforms…")
    poly_conn   = PolymarketConnector()
    kalshi_conn = KalshiConnector()
    poly_markets, kalshi_markets = await asyncio.gather(
        poly_conn.fetch_all(),
        kalshi_conn.fetch_all(),
    )
    print(f"  ✓ Fetched: {len(poly_markets)} Polymarket, {len(kalshi_markets)} Kalshi\n")

    # ── Embed + Match ─────────────────────────────────────────────
    print("  [2/3] Loading embedder and running hierarchical match…")
    embedder = Embedder()
    matcher  = Matcher(embedder)

    # Pass min_sim directly — no monkey-patching of settings required
    groups = matcher.match(poly_markets, kalshi_markets, min_similarity=min_sim)

    # ── Report ────────────────────────────────────────────────────
    print(f"\n  [3/3] Sorting {len(groups)} matches…\n")
    groups.sort(key=lambda g: g.similarity_score, reverse=True)

    print(f"  Total match groups found : {len(groups)}")
    if not groups:
        print("  No matches. Try lowering --min-sim.")
        return

    inverted = [g for g in groups if not g.outcomes_aligned]
    print(
        f"  Inverted outcome pairs   : {len(inverted)}"
        f"  {'⚠ review before trading' if inverted else '✓ none'}"
    )

    print(f"  Showing top {min(top, len(groups))}:")

    for i, g in enumerate(groups[:top], 1):
        _print_group(i, g)

    # ── Spread distribution ───────────────────────────────────────
    spreads = [abs(p - k) for g in groups for p, k in [g.get_comparable_prices()]]
    from config import settings
    total_fees = settings.polymarket_fee_rate + settings.kalshi_fee_rate
    breakeven = total_fees + settings.min_net_roi

    print(f"\n  Spread distribution:")
    print(f"    > {total_fees * 100:.0f}% (raw fees)  : {sum(1 for s in spreads if s > total_fees)}")
    print(f"    > {breakeven * 100:.0f}% (breakeven) : {sum(1 for s in spreads if s > breakeven)}")
    print(f"    > 10%                  : {sum(1 for s in spreads if s > 0.10)}")

    if save_path:
        _save_csv(groups, save_path)

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect semantic match quality.")
    parser.add_argument("--top",     type=int,   default=50,   help="Show top N matches")
    parser.add_argument("--min-sim", type=float, default=None,
                        help="Override similarity threshold (default: MIN_SIMILARITY from .env)")
    parser.add_argument("--save",    type=str,   default=None,
                        help="Save all matches to this CSV file path")
    args = parser.parse_args()

    from config import settings
    threshold = args.min_sim if args.min_sim is not None else settings.min_similarity
    asyncio.run(main(top=args.top, min_sim=threshold, save_path=args.save))
