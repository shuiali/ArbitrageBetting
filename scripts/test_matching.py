#!/usr/bin/env python3
"""
scripts/test_matching.py

Diagnostic script to verify the matching system is working correctly.

Tests:
1. Fetches markets from both platforms
2. Shows sample market titles from each
3. Reports data quality metrics (event_id, event_title population)
4. Tests embedding generation on sample titles
5. Computes similarity between sample pairs
6. Runs a full match and reports statistics

Usage:
    python scripts/test_matching.py
    python scripts/test_matching.py --samples 10   # Show more sample markets
    python scripts/test_matching.py --threshold 0.60  # Test with different threshold
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
logger = logging.getLogger(__name__)

_COL = {
    "RESET":  "\033[0m",
    "BOLD":   "\033[1m",
    "GREEN":  "\033[92m",
    "YELLOW": "\033[93m",
    "RED":    "\033[91m",
    "DIM":    "\033[2m",
    "CYAN":   "\033[96m",
}


def _header(text: str) -> None:
    print(f"\n{_COL['BOLD']}{_COL['CYAN']}{'═' * 60}{_COL['RESET']}")
    print(f"{_COL['BOLD']}{_COL['CYAN']}  {text}{_COL['RESET']}")
    print(f"{_COL['BOLD']}{_COL['CYAN']}{'═' * 60}{_COL['RESET']}")


def _subheader(text: str) -> None:
    print(f"\n{_COL['BOLD']}  {text}{_COL['RESET']}")
    print(f"  {'-' * 50}")


async def main(n_samples: int, threshold: float) -> None:
    from connectors.kalshi import KalshiConnector
    from connectors.polymarket import PolymarketConnector
    from matching.embedder import Embedder
    from matching.matcher import Matcher
    from models.market import Platform

    _header("Matching System Diagnostics")

    # ══════════════════════════════════════════════════════════════
    # Step 1: Fetch markets
    # ══════════════════════════════════════════════════════════════
    _subheader("Step 1: Fetching Markets")
    
    t0 = time.time()
    poly_conn = PolymarketConnector()
    kalshi_conn = KalshiConnector()
    
    poly_markets, kalshi_markets = await asyncio.gather(
        poly_conn.fetch_all(),
        kalshi_conn.fetch_all(),
    )
    fetch_time = time.time() - t0
    
    print(f"  Polymarket: {len(poly_markets):,} markets")
    print(f"  Kalshi:     {len(kalshi_markets):,} markets")
    print(f"  Fetch time: {fetch_time:.1f}s")

    if not poly_markets:
        print(f"\n  {_COL['RED']}✗ Polymarket returned 0 markets!{_COL['RESET']}")
        print("    Check: network, proxy settings, API status")
        return
    
    if not kalshi_markets:
        print(f"\n  {_COL['RED']}✗ Kalshi returned 0 markets!{_COL['RESET']}")
        print("    Check: API credentials, network")
        return

    # ══════════════════════════════════════════════════════════════
    # Step 2: Data Quality Report
    # ══════════════════════════════════════════════════════════════
    _subheader("Step 2: Data Quality Report")
    
    def _quality_report(markets: list, name: str) -> None:
        total = len(markets)
        has_event_id = sum(1 for m in markets if m.event_id)
        has_event_title = sum(1 for m in markets if m.event_title)
        has_title = sum(1 for m in markets if m.title.strip())
        has_price = sum(1 for m in markets if m.has_price)
        
        print(f"\n  {_COL['BOLD']}{name}{_COL['RESET']} ({total:,} markets)")
        print(f"    ├─ Has title:       {has_title:,} ({100*has_title/total:.1f}%)")
        print(f"    ├─ Has event_id:    {has_event_id:,} ({100*has_event_id/total:.1f}%)")
        print(f"    ├─ Has event_title: {has_event_title:,} ({100*has_event_title/total:.1f}%)")
        print(f"    └─ Has price data:  {has_price:,} ({100*has_price/total:.1f}%)")
    
    _quality_report(poly_markets, "Polymarket")
    _quality_report(kalshi_markets, "Kalshi")

    # ══════════════════════════════════════════════════════════════
    # Step 3: Sample Market Titles
    # ══════════════════════════════════════════════════════════════
    _subheader(f"Step 3: Sample Market Titles (showing {n_samples})")
    
    print(f"\n  {_COL['BOLD']}Polymarket samples:{_COL['RESET']}")
    for i, m in enumerate(poly_markets[:n_samples], 1):
        title = m.title[:70] + "…" if len(m.title) > 70 else m.title
        print(f"    {i:2}. {title}")
    
    print(f"\n  {_COL['BOLD']}Kalshi samples:{_COL['RESET']}")
    for i, m in enumerate(kalshi_markets[:n_samples], 1):
        title = m.title[:70] + "…" if len(m.title) > 70 else m.title
        print(f"    {i:2}. {title}")

    # ══════════════════════════════════════════════════════════════
    # Step 4: Embedding Test
    # ══════════════════════════════════════════════════════════════
    _subheader("Step 4: Embedding Test")
    
    print("  Loading embedder (may download model on first run)…")
    t0 = time.time()
    embedder = Embedder()
    load_time = time.time() - t0
    print(f"  Embedder loaded in {load_time:.1f}s")
    
    # Test embedding a few titles
    test_titles = [
        poly_markets[0].title if poly_markets else "Test title A",
        kalshi_markets[0].title if kalshi_markets else "Test title B",
    ]
    
    print(f"\n  Testing embedding generation…")
    embeddings = embedder.embed_strings(test_titles)
    
    if len(embeddings) == len(test_titles):
        vec = list(embeddings.values())[0]
        print(f"  {_COL['GREEN']}✓ Embeddings generated successfully{_COL['RESET']}")
        print(f"    Vector dimension: {vec.shape[0]}")
        print(f"    Vector norm: {float(sum(vec**2)**0.5):.4f} (should be ~1.0 for L2-normalized)")
    else:
        print(f"  {_COL['RED']}✗ Embedding failed!{_COL['RESET']}")
        print(f"    Expected {len(test_titles)} embeddings, got {len(embeddings)}")
        return

    # ══════════════════════════════════════════════════════════════
    # Step 5: Sample Similarity Computation
    # ══════════════════════════════════════════════════════════════
    _subheader("Step 5: Sample Similarity Scores")
    
    import numpy as np
    
    # Embed first few markets from each platform
    sample_poly = poly_markets[:min(5, len(poly_markets))]
    sample_kalshi = kalshi_markets[:min(5, len(kalshi_markets))]
    
    poly_emb = embedder.embed_markets(sample_poly)
    kalshi_emb = embedder.embed_markets(sample_kalshi)
    
    print(f"\n  Computing similarity between {len(poly_emb)} Poly × {len(kalshi_emb)} Kalshi samples:")
    
    for pm in sample_poly[:3]:
        pv = poly_emb.get(pm.market_id)
        if pv is None:
            continue
        for km in sample_kalshi[:3]:
            kv = kalshi_emb.get(km.market_id)
            if kv is None:
                continue
            sim = float(np.dot(pv, kv))
            sim_color = _COL['GREEN'] if sim >= threshold else _COL['DIM']
            pt = pm.title[:30] + "…" if len(pm.title) > 30 else pm.title
            kt = km.title[:30] + "…" if len(km.title) > 30 else km.title
            print(f"    {sim_color}{sim:.3f}{_COL['RESET']} | {pt} ↔ {kt}")

    # ══════════════════════════════════════════════════════════════
    # Step 6: Full Match Test
    # ══════════════════════════════════════════════════════════════
    _subheader(f"Step 6: Full Match Test (threshold={threshold})")
    
    print("  Running full matching algorithm…")
    t0 = time.time()
    matcher = Matcher(embedder)
    groups = matcher.match(poly_markets, kalshi_markets, min_similarity=threshold)
    match_time = time.time() - t0
    
    print(f"\n  {_COL['GREEN'] if groups else _COL['RED']}Match Results:{_COL['RESET']}")
    print(f"    Total match groups: {len(groups):,}")
    print(f"    Match time: {match_time:.1f}s")
    
    if groups:
        scores = [g.similarity_score for g in groups]
        print(f"    Similarity scores: min={min(scores):.3f}, max={max(scores):.3f}, mean={sum(scores)/len(scores):.3f}")
        
        # Show top matches
        print(f"\n  {_COL['BOLD']}Top 5 Matches:{_COL['RESET']}")
        for i, g in enumerate(groups[:5], 1):
            pt = g.poly_market.title[:35] + "…" if len(g.poly_market.title) > 35 else g.poly_market.title
            kt = g.kalshi_market.title[:35] + "…" if len(g.kalshi_market.title) > 35 else g.kalshi_market.title
            align = "aligned" if g.outcomes_aligned else "INVERTED"
            print(f"    {i}. [{g.similarity_score:.3f}] {align}")
            print(f"       Poly:   {pt}")
            print(f"       Kalshi: {kt}")
    else:
        print(f"\n  {_COL['RED']}✗ No matches found!{_COL['RESET']}")
        print(f"    Possible causes:")
        print(f"      - Threshold too high ({threshold}) — try lowering it")
        print(f"      - Embedding generation failed")
        print(f"      - No semantically similar markets exist")
        print(f"\n    Try running with: --threshold 0.50")

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    _header("Summary")
    
    status = _COL['GREEN'] + "✓ PASS" if groups else _COL['RED'] + "✗ FAIL"
    print(f"\n  Status: {status}{_COL['RESET']}")
    print(f"  Markets:   {len(poly_markets):,} Poly × {len(kalshi_markets):,} Kalshi")
    print(f"  Matches:   {len(groups):,} groups")
    print(f"  Threshold: {threshold}")
    print(f"  Time:      {fetch_time + match_time:.1f}s total\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matching system diagnostics")
    parser.add_argument(
        "--samples", type=int, default=5,
        help="Number of sample markets to display (default: 5)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Similarity threshold to test (default: from config)",
    )
    args = parser.parse_args()

    from config import settings
    threshold = args.threshold if args.threshold is not None else settings.min_similarity
    
    asyncio.run(main(n_samples=args.samples, threshold=threshold))
