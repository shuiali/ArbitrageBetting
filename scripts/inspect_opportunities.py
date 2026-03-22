#!/usr/bin/env python3
"""
scripts/inspect_opportunities.py

Detailed inspection tool for match groups and opportunity detection.
Loads match groups and prints comprehensive diagnostics to help debug
why opportunities are not being detected.

Usage:
    python scripts/inspect_opportunities.py
    python scripts/inspect_opportunities.py --count 50
"""

import argparse
import asyncio
import logging
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

async def main(count: int) -> None:
    from arbitrage.engine import ArbEngine
    from connectors.kalshi import KalshiConnector
    from connectors.polymarket import PolymarketConnector
    from config import settings
    from matching.embedder import Embedder
    from matching.matcher import Matcher

    print("\n" + "=" * 80)
    print("MATCH GROUP INSPECTOR — Diagnostic Tool")
    print("=" * 80)

    # Fetch markets
    print("\n[1/3] Fetching markets...")
    poly_conn   = PolymarketConnector()
    kalshi_conn = KalshiConnector()
    poly_markets, kalshi_markets = await asyncio.gather(
        poly_conn.fetch_all(),
        kalshi_conn.fetch_all(),
    )
    
    if not poly_markets or not kalshi_markets:
        print("✗ One or both connectors returned 0 markets. Cannot proceed.")
        return
    
    print(f"  Polymarket: {len(poly_markets)} markets")
    print(f"  Kalshi:     {len(kalshi_markets)} markets")

    # Match
    print("\n[2/3] Running semantic match...")
    embedder = Embedder()
    matcher  = Matcher(embedder)
    groups   = matcher.match(poly_markets, kalshi_markets)
    
    if not groups:
        print("✗ No match groups found.")
        return
    
    print(f"  Match groups: {len(groups)}")

    # Analyze
    print(f"\n[3/3] Analyzing match groups (showing first {count})...\n")
    print("=" * 80)
    
    # Statistics
    stats = {
        'has_price_both': 0,
        'has_price_poly_only': 0,
        'has_price_kalshi_only': 0,
        'has_price_neither': 0,
        'outcomes_aligned': 0,
        'outcomes_inverted': 0,
        'spreads': [],
    }
    
    for i, g in enumerate(groups[:count], 1):
        print(f"\nGROUP #{i}  [similarity={g.similarity_score:.4f}]")
        print("-" * 80)
        
        # Market info
        poly_m = g.poly_market
        kalshi_m = g.kalshi_market
        
        print(f"Polymarket:")
        print(f"  Title:     {poly_m.title}")
        print(f"  YES price: {poly_m.yes_price:.6f}")
        print(f"  NO price:  {poly_m.no_price:.6f}")
        print(f"  has_price: {poly_m.has_price}")
        print(f"  Volume:    ${poly_m.volume_usd:,.0f}")
        
        print(f"\nKalshi:")
        print(f"  Title:     {kalshi_m.title}")
        print(f"  YES price: {kalshi_m.yes_price:.6f}")
        print(f"  NO price:  {kalshi_m.no_price:.6f}")
        print(f"  has_price: {kalshi_m.has_price}")
        print(f"  Volume:    ${kalshi_m.volume_usd:,.0f}")
        
        # Outcome alignment
        print(f"\nOutcome alignment: {g.outcomes_aligned}")
        if g.outcomes_aligned:
            print("  (YES on both platforms = same real-world outcome)")
        else:
            print("  (YES on Polymarket = NO on Kalshi — inverted)")
        
        # Comparable prices
        poly_price, kalshi_price = g.get_comparable_prices()
        gross_spread = abs(poly_price - kalshi_price)
        
        print(f"\nComparable prices:")
        print(f"  Poly (YES-equivalent):   {poly_price:.6f}")
        print(f"  Kalshi (YES-equivalent): {kalshi_price:.6f}")
        print(f"  Gross spread:            {gross_spread:.6f} ({gross_spread*100:.4f}%)")
        
        # Net ROI calculation
        total_fees = settings.polymarket_fee_rate + settings.kalshi_fee_rate
        net_roi = gross_spread - total_fees
        print(f"  Total fees:              {total_fees:.6f} ({total_fees*100:.2f}%)")
        print(f"  Net ROI:                 {net_roi:.6f} ({net_roi*100:.4f}%)")
        
        # Price range check
        in_range_poly = 0.01 < poly_price < 0.99
        in_range_kalshi = 0.01 < kalshi_price < 0.99
        print(f"\nPrice range validation:")
        print(f"  Poly in [0.01, 0.99]:   {in_range_poly}")
        print(f"  Kalshi in [0.01, 0.99]: {in_range_kalshi}")
        
        # Decision
        print(f"\nWould be detected as opportunity:")
        passed_has_price = poly_m.has_price and kalshi_m.has_price
        passed_range = in_range_poly and in_range_kalshi
        passed_roi = net_roi >= settings.min_net_roi
        print(f"  ✓ has_price check:    {passed_has_price}")
        print(f"  ✓ price range check:  {passed_range}")
        print(f"  ✓ ROI ≥ {settings.min_net_roi:.4f}:     {passed_roi}")
        
        if passed_has_price and passed_range and passed_roi:
            print("  → YES — This IS a valid opportunity")
        else:
            print("  → NO — Filtered out")
            if not passed_has_price:
                print(f"     Reason: has_price failed (poly={poly_m.has_price}, kalshi={kalshi_m.has_price})")
            elif not passed_range:
                print(f"     Reason: price range check failed")
            else:
                print(f"     Reason: spread too small (need {(total_fees + settings.min_net_roi)*100:.2f}%, got {gross_spread*100:.4f}%)")
        
        # Collect stats
        if poly_m.has_price and kalshi_m.has_price:
            stats['has_price_both'] += 1
        elif poly_m.has_price:
            stats['has_price_poly_only'] += 1
        elif kalshi_m.has_price:
            stats['has_price_kalshi_only'] += 1
        else:
            stats['has_price_neither'] += 1
        
        if g.outcomes_aligned:
            stats['outcomes_aligned'] += 1
        else:
            stats['outcomes_inverted'] += 1
        
        stats['spreads'].append(gross_spread)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total groups analyzed: {min(count, len(groups))}")
    print(f"\nhas_price distribution:")
    print(f"  Both have price:   {stats['has_price_both']}")
    print(f"  Poly only:         {stats['has_price_poly_only']}")
    print(f"  Kalshi only:       {stats['has_price_kalshi_only']}")
    print(f"  Neither has price: {stats['has_price_neither']}")
    
    print(f"\nOutcome alignment:")
    print(f"  Aligned (YES↔YES): {stats['outcomes_aligned']}")
    print(f"  Inverted (YES↔NO): {stats['outcomes_inverted']}")
    
    if stats['spreads']:
        spreads = stats['spreads']
        spreads_sorted = sorted(spreads)
        print(f"\nGross spread distribution:")
        print(f"  Min:    {min(spreads):.6f} ({min(spreads)*100:.4f}%)")
        print(f"  Max:    {max(spreads):.6f} ({max(spreads)*100:.4f}%)")
        print(f"  Median: {spreads_sorted[len(spreads)//2]:.6f} ({spreads_sorted[len(spreads)//2]*100:.4f}%)")
        print(f"  Mean:   {sum(spreads)/len(spreads):.6f} ({sum(spreads)/len(spreads)*100:.4f}%)")
        
        # Histogram
        print(f"\nSpread histogram:")
        buckets = defaultdict(int)
        for s in spreads:
            if s < 0.001:
                buckets['<0.1%'] += 1
            elif s < 0.01:
                buckets['0.1-1%'] += 1
            elif s < 0.02:
                buckets['1-2%'] += 1
            elif s < 0.05:
                buckets['2-5%'] += 1
            elif s < 0.10:
                buckets['5-10%'] += 1
            else:
                buckets['>10%'] += 1
        
        for bucket in ['<0.1%', '0.1-1%', '1-2%', '2-5%', '5-10%', '>10%']:
            count = buckets.get(bucket, 0)
            pct = 100 * count / len(spreads) if spreads else 0
            bar = '█' * int(pct / 2)
            print(f"  {bucket:>8} | {bar} {count} ({pct:.1f}%)")
    
    print("\n" + "=" * 80)
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect match groups for debugging.")
    parser.add_argument(
        "--count", type=int, default=20,
        help="Number of match groups to inspect (default: 20)",
    )
    args = parser.parse_args()
    asyncio.run(main(count=args.count))
