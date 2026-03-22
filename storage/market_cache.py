"""
storage/market_cache.py

Persistent disk cache for connector market snapshots.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config import settings
from models.market import Platform, UnifiedMarket

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent / "cache"
_POLY_CACHE_FILE = _CACHE_DIR / "polymarket_markets.json"
_KALSHI_CACHE_FILE = _CACHE_DIR / "kalshi_markets.json"


@dataclass(frozen=True)
class CacheLoadResult:
    markets: Optional[List[UnifiedMarket]]
    status: str
    age_seconds: Optional[float]


def _cache_file(platform: Platform) -> Path:
    return _POLY_CACHE_FILE if platform == Platform.POLYMARKET else _KALSHI_CACHE_FILE


def _ensure_cache_dir() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def market_to_dict(m: UnifiedMarket) -> dict:
    return {
        "platform": m.platform.name,
        "market_id": m.market_id,
        "title": m.title,
        "yes_price": m.yes_price,
        "no_price": m.no_price,
        "volume_usd": m.volume_usd,
        "url": m.url,
        "outcome_count": m.outcome_count,
        "event_id": m.event_id,
        "event_title": m.event_title,
        "fetched_at": m.fetched_at,
    }


def dict_to_market(d: dict) -> UnifiedMarket:
    return UnifiedMarket(
        platform=Platform[d["platform"]],
        market_id=d["market_id"],
        title=d["title"],
        yes_price=d["yes_price"],
        no_price=d["no_price"],
        volume_usd=d["volume_usd"],
        url=d["url"],
        outcome_count=d.get("outcome_count", 2),
        event_id=d.get("event_id", ""),
        event_title=d.get("event_title", ""),
        fetched_at=d.get("fetched_at", time.time()),
    )


def save_markets(
    platform: Platform,
    markets: List[UnifiedMarket],
    *,
    source_ok: bool = True,
    note: str = "",
) -> None:
    """
    Save markets to cache file.

    Empty snapshots are not persisted when MARKET_CACHE_REJECT_EMPTY is enabled.
    """
    snapshot = list(markets)
    if settings.market_cache_reject_empty and source_ok and not snapshot:
        logger.warning(
            "Skipping cache write for %s: empty successful snapshot rejected.",
            platform.name,
        )
        return

    _ensure_cache_dir()
    cache_file = _cache_file(platform)

    data = {
        "timestamp": time.time(),
        "platform": platform.name,
        "count": len(snapshot),
        "source_ok": bool(source_ok),
        "note": note,
        "markets": [market_to_dict(m) for m in snapshot],
    }

    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "Cached %d %s markets to %s (source_ok=%s)",
            len(snapshot), platform.name, cache_file.name, source_ok,
        )
    except Exception as exc:
        logger.warning("Failed to cache %s markets: %s", platform.name, exc)


def load_markets_with_status(
    platform: Platform,
    *,
    allow_stale: bool = False,
) -> CacheLoadResult:
    cache_file = _cache_file(platform)
    if not cache_file.exists():
        return CacheLoadResult(markets=None, status="missing", age_seconds=None)

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load %s cache: %s", platform.name, exc)
        return CacheLoadResult(markets=None, status="read_error", age_seconds=None)

    timestamp = float(data.get("timestamp", 0.0))
    if timestamp <= 0:
        return CacheLoadResult(markets=None, status="invalid_timestamp", age_seconds=None)

    age_seconds = max(0.0, time.time() - timestamp)
    fresh_ttl = max(1, settings.market_cache_ttl_seconds)
    stale_ttl = max(fresh_ttl, settings.market_cache_stale_ttl_seconds)

    if age_seconds > stale_ttl:
        return CacheLoadResult(markets=None, status="expired", age_seconds=age_seconds)

    markets = [dict_to_market(d) for d in data.get("markets", [])]
    if settings.market_cache_reject_empty and not markets:
        return CacheLoadResult(markets=None, status="empty_cache", age_seconds=age_seconds)

    if age_seconds <= fresh_ttl:
        return CacheLoadResult(markets=markets, status="fresh_hit", age_seconds=age_seconds)

    if allow_stale and settings.market_cache_allow_stale_on_error:
        return CacheLoadResult(markets=markets, status="stale_fallback", age_seconds=age_seconds)

    return CacheLoadResult(markets=None, status="stale_rejected", age_seconds=age_seconds)


def load_markets(platform: Platform) -> Optional[List[UnifiedMarket]]:
    result = load_markets_with_status(platform, allow_stale=False)
    if result.markets is not None:
        age = result.age_seconds or 0.0
        logger.info(
            "Loaded %d %s markets from cache (%s, %.1fm old)",
            len(result.markets),
            platform.name,
            result.status,
            age / 60.0,
        )
    return result.markets


def clear_cache(platform: Optional[Platform] = None) -> None:
    files: List[Path]
    if platform is None:
        files = [_POLY_CACHE_FILE, _KALSHI_CACHE_FILE]
    elif platform == Platform.POLYMARKET:
        files = [_POLY_CACHE_FILE]
    else:
        files = [_KALSHI_CACHE_FILE]

    for file_path in files:
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info("Cleared cache: %s", file_path.name)
        except Exception as exc:
            logger.warning("Failed to clear %s: %s", file_path.name, exc)


def cache_size() -> dict:
    return {
        "polymarket": _POLY_CACHE_FILE.stat().st_size if _POLY_CACHE_FILE.exists() else 0,
        "kalshi": _KALSHI_CACHE_FILE.stat().st_size if _KALSHI_CACHE_FILE.exists() else 0,
    }
