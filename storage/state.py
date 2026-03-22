"""
storage/state.py

In-memory state — no Redis, no DB for now.
This is intentionally simple. The future site will replace this with
a proper persistence layer; this module is the abstraction boundary.

Responsibilities:
  - Store the current list of MatchGroups (updated after each re-match).
  - Track which ArbOpportunity keys have been alerted and when,
    to enforce the cooldown window (avoid spamming the same opportunity).
"""

import logging
import time
from typing import Dict, List, Optional

from config import settings
from models.market import ArbOpportunity, MatchGroup

logger = logging.getLogger(__name__)


class StateManager:
    def __init__(self) -> None:
        self._match_groups: List[MatchGroup] = []
        # dedup_key → last alert timestamp
        self._alerted: Dict[str, float] = {}
        self._last_match_time: float = 0.0
        self._last_poly_count: int = 0
        self._last_kalshi_count: int = 0

    # ── Match groups ─────────────────────────────────────────────

    def update_match_groups(self, groups: List[MatchGroup]) -> None:
        self._match_groups = groups
        self._last_match_time = time.time()
        logger.info("State: updated to %d match groups.", len(groups))

    def get_match_groups(self) -> List[MatchGroup]:
        return self._match_groups

    def needs_rematch(self) -> bool:
        age = time.time() - self._last_match_time
        return age >= settings.rematch_interval_seconds

    # ── Opportunity dedup ────────────────────────────────────────

    def should_alert(self, opp: ArbOpportunity) -> bool:
        """
        Returns True if this opportunity has not been alerted,
        or was last alerted more than ALERT_COOLDOWN_SECONDS ago.
        """
        key = opp.dedup_key
        last = self._alerted.get(key)
        if last is None:
            return True
        return (time.time() - last) >= settings.alert_cooldown_seconds

    def mark_alerted(self, opp: ArbOpportunity) -> None:
        self._alerted[opp.dedup_key] = time.time()

    def purge_stale_alerts(self) -> None:
        """
        Remove alert records for opportunities that expired long ago.
        Prevents unbounded growth of the dict over multi-day runs.
        """
        cutoff = time.time() - (settings.alert_cooldown_seconds * 10)
        stale = [k for k, ts in self._alerted.items() if ts < cutoff]
        for k in stale:
            del self._alerted[k]

    # ── Stats ────────────────────────────────────────────────────

    def record_fetch_counts(self, poly_count: int, kalshi_count: int) -> None:
        self._last_poly_count = poly_count
        self._last_kalshi_count = kalshi_count

    def summary(self) -> str:
        return (
            f"poly={self._last_poly_count} kalshi={self._last_kalshi_count} "
            f"groups={len(self._match_groups)} alerted_keys={len(self._alerted)}"
        )
