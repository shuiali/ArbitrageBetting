#!/usr/bin/env python3
"""
scripts/cache_manager.py

Manage market cache.

Usage:
  python scripts/cache_manager.py status    # Show cache info
  python scripts/cache_manager.py clear     # Clear all caches
  python scripts/cache_manager.py clear-poly   # Clear Polymarket cache only
  python scripts/cache_manager.py clear-kalshi # Clear Kalshi cache only
"""

import sys
import logging
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.market_cache import (
    load_markets,
    load_markets_with_status,
    clear_cache,
    cache_size,
)
from models.market import Platform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_status():
    """Show cache status"""
    logger.info("Market Cache Status")
    logger.info("─" * 40)
    
    sizes = cache_size()
    poly_mb = sizes["polymarket"] / (1024 ** 2)
    kalshi_mb = sizes["kalshi"] / (1024 ** 2)
    
    logger.info(f"Polymarket cache: {poly_mb:.2f} MB")
    logger.info(f"Kalshi cache:     {kalshi_mb:.2f} MB")
    
    # Show policy status + loadability
    poly_status = load_markets_with_status(Platform.POLYMARKET, allow_stale=True)
    kalshi_status = load_markets_with_status(Platform.KALSHI, allow_stale=True)
    poly = load_markets(Platform.POLYMARKET)
    kalshi = load_markets(Platform.KALSHI)

    if poly:
        logger.info(f"  → {len(poly)} markets (loaded successfully)")
    else:
        logger.info(f"  → Not available or expired")
    logger.info(
        "Polymarket status: %s age=%.1fm",
        poly_status.status,
        (poly_status.age_seconds or 0.0) / 60.0 if poly_status.age_seconds is not None else -1.0,
    )

    if kalshi:
        logger.info(f"  → {len(kalshi)} markets (loaded successfully)")
    else:
        logger.info(f"  → Not available or expired")
    logger.info(
        "Kalshi status: %s age=%.1fm",
        kalshi_status.status,
        (kalshi_status.age_seconds or 0.0) / 60.0 if kalshi_status.age_seconds is not None else -1.0,
    )


def cmd_clear(platform=None):
    """Clear cache"""
    if platform is None:
        logger.info("Clearing ALL caches...")
        clear_cache()
        logger.info("✓ All caches cleared")
    elif platform == "poly":
        logger.info("Clearing Polymarket cache...")
        clear_cache(Platform.POLYMARKET)
        logger.info("✓ Polymarket cache cleared")
    elif platform == "kalshi":
        logger.info("Clearing Kalshi cache...")
        clear_cache(Platform.KALSHI)
        logger.info("✓ Kalshi cache cleared")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == "status":
        cmd_status()
    elif cmd == "clear":
        cmd_clear()
    elif cmd == "clear-poly":
        cmd_clear("poly")
    elif cmd == "clear-kalshi":
        cmd_clear("kalshi")
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
