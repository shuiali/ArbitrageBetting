"""
arbitrage/engine.py

Pure maths layer — no I/O, no side effects.

Takes MatchGroup objects and decides if a real arbitrage opportunity exists.

────────────────────────────────────────────────────────────────────────────
Maths
────────────────────────────────────────────────────────────────────────────
For a binary market, YES resolves to $1 if the event happens and $0 if not.

Given two platforms with comparable YES prices p_A and p_B (both in [0,1]):

  If p_A < p_B:
    Buy YES on A for p_A
    Buy NO  on B for (1 - p_B)
    Total cost     = p_A + (1 - p_B)
    Guaranteed pay = $1.00   (one leg always wins)
    Gross profit   = 1 - cost = p_B - p_A = gross_spread

  gross_spread = |p_A - p_B|
  total_fees   = poly_fee_rate + kalshi_fee_rate
  net_roi      = gross_spread - total_fees

We require net_roi ≥ min_roi (from config or caller override) to surface
an opportunity.

Why filter binary markets?
  Multi-outcome markets (e.g. "who wins the nomination?" with 5 candidates)
  have different pricing logic — YES on one candidate ≠ NO on all others.
  The simple two-leg arbitrage above only works for binary (YES/NO) markets.
  We define a market as binary when its outcome_count == 2, i.e. the event
  has exactly one market within it (no sibling "candidate" markets).
  Markets in events with multiple siblings are also accepted as binary when
  their individual outcome is still YES/NO — we check has_price instead.
────────────────────────────────────────────────────────────────────────────
"""

import logging
from typing import List, Optional

from config import settings
from models.market import ArbOpportunity, MatchGroup, Platform

logger = logging.getLogger(__name__)


class ArbEngine:

    def __init__(self) -> None:
        self._poly_fee   = settings.polymarket_fee_rate
        self._kalshi_fee = settings.kalshi_fee_rate
        self._total_fees = self._poly_fee + self._kalshi_fee
        self._min_roi    = settings.min_net_roi
        logger.info(
            "ArbEngine: poly_fee=%.1f%%  kalshi_fee=%.1f%%  "
            "total_fees=%.1f%%  min_roi=%.1f%%",
            self._poly_fee * 100, self._kalshi_fee * 100,
            self._total_fees * 100, self._min_roi * 100,
        )

    def find_opportunities(
        self,
        groups: List[MatchGroup],
        min_roi: Optional[float] = None,
    ) -> List[ArbOpportunity]:
        """
        Scan match groups and return those with profitable spreads.
        Results are sorted by net_roi descending.

        Args:
            groups:  List of MatchGroups from the Matcher.
            min_roi: Optional override for minimum net ROI threshold.
                     Useful in scan_once.py for "show all spreads" mode.
        """
        _min_roi = min_roi if min_roi is not None else self._min_roi
        opportunities: List[ArbOpportunity] = []
        
        # Diagnostic counters
        rejected_no_price = 0
        rejected_price_range = 0
        rejected_low_roi = 0
        rejected_samples = []

        for group in groups:
            opp, rejection_reason = self._evaluate(group, _min_roi)
            if opp is not None:
                opportunities.append(opp)
            elif rejection_reason and len(rejected_samples) < 5:
                rejected_samples.append(rejection_reason)
            
            # Count rejections
            if rejection_reason:
                if "no_price" in rejection_reason:
                    rejected_no_price += 1
                elif "price_range" in rejection_reason:
                    rejected_price_range += 1
                elif "low_roi" in rejection_reason:
                    rejected_low_roi += 1

        opportunities.sort(key=lambda o: o.net_roi, reverse=True)

        if opportunities:
            logger.info(
                "ArbEngine: %d / %d groups profitable (best net ROI: %.2f%%)",
                len(opportunities), len(groups),
                opportunities[0].net_roi_pct,
            )
        else:
            logger.warning("ArbEngine: no opportunities in %d groups.", len(groups))
            logger.warning(
                "Rejection breakdown: no_price=%d, price_range=%d, low_roi=%d",
                rejected_no_price, rejected_price_range, rejected_low_roi,
            )
            if rejected_samples:
                logger.warning("Sample rejections:")
                for i, reason in enumerate(rejected_samples, 1):
                    logger.warning("  %d. %s", i, reason)

        return opportunities

    def _evaluate(self, group: MatchGroup, min_roi: float) -> tuple[Optional[ArbOpportunity], Optional[str]]:
        """
        Evaluate a single MatchGroup. Returns (ArbOpportunity, None) or (None, rejection_reason).
        """
        poly_m   = group.poly_market
        kalshi_m = group.kalshi_market

        # Skip markets that have no price data — can't trade them
        if not poly_m.has_price or not kalshi_m.has_price:
            reason = (
                f"no_price: poly({poly_m.yes_price:.4f}, {poly_m.no_price:.4f}) "
                f"kalshi({kalshi_m.yes_price:.4f}, {kalshi_m.no_price:.4f}) | "
                f"poly_has={poly_m.has_price} kalshi_has={kalshi_m.has_price} | "
                f"'{poly_m.title[:40]}' vs '{kalshi_m.title[:40]}'"
            )
            return None, reason

        poly_price, kalshi_price = group.get_comparable_prices()

        # Belt-and-suspenders price range check (connectors should already filter)
        if not (0.01 < poly_price < 0.99 and 0.01 < kalshi_price < 0.99):
            reason = (
                f"price_range: poly_comp={poly_price:.4f} kalshi_comp={kalshi_price:.4f} | "
                f"poly_raw({poly_m.yes_price:.4f}, {poly_m.no_price:.4f}) "
                f"kalshi_raw({kalshi_m.yes_price:.4f}, {kalshi_m.no_price:.4f}) | "
                f"aligned={group.outcomes_aligned} | "
                f"'{poly_m.title[:40]}' vs '{kalshi_m.title[:40]}'"
            )
            return None, reason

        gross_spread = abs(poly_price - kalshi_price)
        net_roi = gross_spread - self._total_fees

        if net_roi < min_roi:
            reason = (
                f"low_roi: spread={gross_spread:.4f} net_roi={net_roi:.4f} (need {min_roi:.4f}) | "
                f"poly={poly_price:.4f} kalshi={kalshi_price:.4f} | "
                f"'{poly_m.title[:40]}' vs '{kalshi_m.title[:40]}'"
            )
            return None, reason

        # Comparable side per platform for the SAME real-world outcome.
        # Polymarket comparable side is always YES.
        # Kalshi comparable side is YES when aligned, NO when inverted.
        poly_comp_side = "YES"
        kalshi_comp_side = "YES" if group.outcomes_aligned else "NO"

        # Determine which comparable side is cheap, then hedge with the opposite
        # contract on the expensive platform.
        if poly_price < kalshi_price:
            buy_platform  = Platform.POLYMARKET
            buy_side      = poly_comp_side
            buy_price     = poly_price
            sell_platform = Platform.KALSHI
            hedge_side    = "NO" if kalshi_comp_side == "YES" else "YES"
            sell_price    = kalshi_price
        else:
            buy_platform  = Platform.KALSHI
            buy_side      = kalshi_comp_side
            buy_price     = kalshi_price
            sell_platform = Platform.POLYMARKET
            hedge_side    = "NO" if poly_comp_side == "YES" else "YES"
            sell_price    = poly_price

        opportunity = ArbOpportunity(
            group_id      = group.group_id,
            poly_market   = poly_m,
            kalshi_market = kalshi_m,
            buy_platform  = buy_platform,
            buy_side      = buy_side,
            buy_price     = buy_price,
            sell_platform = sell_platform,
            hedge_side    = hedge_side,
            sell_price    = sell_price,
            gross_spread  = gross_spread,
            total_fees    = self._total_fees,
            net_roi       = net_roi,
            similarity_score = group.similarity_score,
        )
        return opportunity, None
