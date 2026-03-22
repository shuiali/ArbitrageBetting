"""
config.py
All runtime configuration loaded from environment / .env file.
Import `settings` anywhere in the codebase.
"""

import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Telegram (optional — empty = disabled)
    telegram_bot_token: str
    telegram_chat_id: str

    # Kalshi
    kalshi_api_key: str
    kalshi_api_key_id: str
    kalshi_private_key_path: str
    kalshi_env: str  # "production" | "demo"

    # Thresholds
    min_net_roi: float
    min_similarity: float
    alert_cooldown_seconds: int

    # Polling
    poll_interval_seconds: int
    rematch_interval_seconds: int

    # Fees
    polymarket_fee_rate: float
    kalshi_fee_rate: float

    # Logging
    log_level: str

    # Proxies
    polymarket_proxy: str

    # Cache policy
    market_cache_ttl_seconds: int
    market_cache_stale_ttl_seconds: int
    market_cache_reject_empty: bool
    market_cache_allow_stale_on_error: bool

    # Polymarket fetch tuning
    polymarket_events_rate_limit_per_10s: int
    polymarket_events_parallel_pages: int
    polymarket_events_page_size: int

    # Embedding cache / performance
    embedding_cache_enabled: bool
    embedding_cache_path: str
    embedding_cache_save_interval_seconds: int
    embedding_batch_size_override: int
    embedding_batch_vram_factor: float

    @property
    def kalshi_base_url(self) -> str:
        if self.kalshi_env == "demo":
            return "https://demo-api.kalshi.co/trade-api/v2"
        return "https://api.elections.kalshi.com/trade-api/v2"

    @property
    def polymarket_clob_url(self) -> str:
        return "https://clob.polymarket.com"

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token) and bool(self.telegram_chat_id)


def _load_settings() -> Settings:
    return Settings(
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        kalshi_api_key=os.getenv("KALSHI_API_KEY", ""),
        kalshi_api_key_id=os.getenv("KALSHI_API_KEY_ID", ""),
        kalshi_private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH", ""),
        kalshi_env=os.getenv("KALSHI_ENV", "production"),
        min_net_roi=float(os.getenv("MIN_NET_ROI", "0.02")),
        # Market-level matching threshold (0.82) — part of three-tier system:
        # Event auto-accept: ≥0.88, Event review: 0.70-0.88, Market: ≥0.82
        min_similarity=float(os.getenv("MIN_SIMILARITY", "0.82")),
        alert_cooldown_seconds=int(os.getenv("ALERT_COOLDOWN_SECONDS", "300")),
        poll_interval_seconds=int(os.getenv("POLL_INTERVAL_SECONDS", "60")),
        rematch_interval_seconds=int(os.getenv("REMATCH_INTERVAL_SECONDS", "1800")),
        polymarket_fee_rate=float(os.getenv("POLYMARKET_FEE_RATE", "0.02")),
        kalshi_fee_rate=float(os.getenv("KALSHI_FEE_RATE", "0.04")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        polymarket_proxy=os.getenv("POLYMARKET_PROXY", ""),
        market_cache_ttl_seconds=int(os.getenv("MARKET_CACHE_TTL_SECONDS", "3600")),
        market_cache_stale_ttl_seconds=int(os.getenv("MARKET_CACHE_STALE_TTL_SECONDS", "21600")),
        market_cache_reject_empty=_env_bool("MARKET_CACHE_REJECT_EMPTY", True),
        market_cache_allow_stale_on_error=_env_bool("MARKET_CACHE_ALLOW_STALE_ON_ERROR", True),
        polymarket_events_rate_limit_per_10s=int(os.getenv("POLYMARKET_EVENTS_RATE_LIMIT_PER_10S", "290")),
        polymarket_events_parallel_pages=int(os.getenv("POLYMARKET_EVENTS_PARALLEL_PAGES", "24")),
        polymarket_events_page_size=int(os.getenv("POLYMARKET_EVENTS_PAGE_SIZE", "100")),
        embedding_cache_enabled=_env_bool("EMBEDDING_CACHE_ENABLED", True),
        embedding_cache_path=os.getenv("EMBEDDING_CACHE_PATH", "cache\\embedding_cache.pkl"),
        embedding_cache_save_interval_seconds=int(os.getenv("EMBEDDING_CACHE_SAVE_INTERVAL_SECONDS", "20")),
        embedding_batch_size_override=int(os.getenv("EMBEDDING_BATCH_SIZE_OVERRIDE", "0")),
        embedding_batch_vram_factor=float(os.getenv("EMBEDDING_BATCH_VRAM_FACTOR", "24")),
    )


settings = _load_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
