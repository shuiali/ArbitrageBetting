"""
models/market.py
Shared dataclasses that flow through every layer of the system.
All prices are stored as floats in [0.0, 1.0] (probability space).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class Platform(str, Enum):
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"


@dataclass
class UnifiedMarket:
    """
    Normalised view of a prediction market from any platform.
    Accepts ALL market types (binary, multi-outcome, etc.).
    YES/NO prices are in [0, 1] probability space.
    """
    platform: Platform
    market_id: str          # Native ID (condition_id / ticker)
    title: str              # The question text used for semantic matching
    yes_price: float        # Mid-price for YES outcome  [0, 1]
    no_price: float         # Mid-price for NO outcome   [0, 1]
    volume_usd: float       # Total traded volume
    url: str                # Direct link to market page
    outcome_count: int = 2  # Number of outcomes (2 = binary)
    event_id: str = ""      # The parent event ID
    event_title: str = ""   # The parent event title
    fetched_at: float = field(default_factory=time.time)

    def is_valid(self) -> bool:
        """Accept any market that has an ID and a title."""
        return bool(self.market_id) and bool(self.title.strip())

    @property
    def has_price(self) -> bool:
        """True if this market has meaningful price data."""
        return self.yes_price > 0.001 or self.no_price > 0.001

    @property
    def implied_sum(self) -> float:
        """Should be ~1.0 in an efficient market; >1 means overround."""
        return self.yes_price + self.no_price

    def yes_outcome_text(self) -> str:
        """Best-effort human label for what YES means on this market."""
        title = (self.title or "").strip()
        if not title:
            return "YES outcome"

        if "—" in title:
            _, rhs = title.split("—", 1)
            rhs = rhs.strip()
            if rhs:
                return rhs.rstrip(" ?")

        return title.rstrip(" ?")

    def no_outcome_text(self) -> str:
        """Best-effort human label for what NO means on this market."""
        return f"NOT ({self.yes_outcome_text()})"


@dataclass
class MatchGroup:
    """
    Two markets from different platforms judged to be the same real-world event.
    Includes outcome alignment: poly_yes_maps_to_kalshi_yes = True means
    YES on Polymarket and YES on Kalshi resolve on the same condition.
    """
    group_id: str                    # Stable hash of (poly_id, kalshi_id)
    poly_market: UnifiedMarket
    kalshi_market: UnifiedMarket
    similarity_score: float          # BGE-M3 cosine similarity
    outcomes_aligned: bool           # True = YES↔YES, False = YES↔NO (inverted)
    matched_at: float = field(default_factory=time.time)

    def get_comparable_prices(self) -> tuple[float, float]:
        """
        Return (poly_yes_equivalent, kalshi_yes_equivalent) — both referring
        to the same real-world outcome so we can directly compare them.
        """
        kalshi_equiv = (
            self.kalshi_market.yes_price
            if self.outcomes_aligned
            else self.kalshi_market.no_price
        )
        return self.poly_market.yes_price, kalshi_equiv


@dataclass
class ArbOpportunity:
    """
    A detected arbitrage: buy YES-equivalent on the cheap side,
    buy NO-equivalent on the expensive side.
    """
    group_id: str
    poly_market: UnifiedMarket
    kalshi_market: UnifiedMarket

    # Which platform has the cheap YES
    buy_platform: Platform
    buy_side: str                  # "YES" or "NO" contract side on buy_platform
    buy_price: float                 # Cost of cheap YES leg
    sell_platform: Platform
    hedge_side: str                # "YES" or "NO" contract side on sell_platform
    sell_price: float                # Cost of corresponding NO leg (1 - expensive_yes)

    gross_spread: float              # Before fees
    total_fees: float                # Sum of both platform fee rates × $1 payout
    net_roi: float                   # gross_spread - total_fees (as fraction)

    similarity_score: float
    detected_at: float = field(default_factory=time.time)

    @property
    def net_roi_pct(self) -> float:
        return round(self.net_roi * 100, 3)

    @property
    def gross_spread_pct(self) -> float:
        return round(self.gross_spread * 100, 3)

    @property
    def dedup_key(self) -> str:
        return f"{self.group_id}:{self.buy_platform}"

    @property
    def hedge_price(self) -> float:
        return round(1.0 - self.sell_price, 4)

    def outcome_text_for(self, platform: Platform, side: str) -> str:
        market = self.poly_market if platform == Platform.POLYMARKET else self.kalshi_market
        return market.yes_outcome_text() if side.upper() == "YES" else market.no_outcome_text()
