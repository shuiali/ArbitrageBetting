"""
matching/matcher.py

Cross-platform semantic matching using BAAI/bge-m3 with three-tier thresholding.

Algorithm (hierarchical event→market matching with three-tier thresholding)
──────────────────────────────────────────────────────────────────────────────
1.  Group markets by their parent event on each platform.
2.  Embed ALL event titles (batch processing for speed).
3.  Build (N_poly_events × N_kalshi_events) similarity matrix (~30M comparisons, ~3 sec).
4.  **Three-tier thresholding** on event pairs:
    - Auto-accept (≥0.88): High confidence, feed directly to market matching
    - Review zone (0.70-0.88): Log to CSV for human inspection
    - Reject (<0.70): Discard as genuinely unrelated
5.  **Many-to-many at event level**: Same event can match multiple events on other platform.
6.  For each auto-accepted event pair, embed markets within those events only.
7.  **One-to-one at market level**: Greedy assignment within event pairs (≥0.82 threshold).
8.  Infer outcome alignment per market pair and return MatchGroup objects.

Why three tiers?
  - Single threshold (0.70 or 0.82) is too simplistic: either misses matches or floods with noise
  - ≥0.88: BGE-M3 is certain (trust it completely)
  - 0.70-0.88: BGE-M3 is uncertain (human review needed)
  - <0.70: Genuinely different events

Why many-to-many events?
  - Platforms structure events differently:
    - Polymarket: "2024 US Election" (1 event, 12 markets)  
    - Kalshi: "Rep nominee wins", "Dem nominee wins", "Third party wins" (3 events, 4 markets each)
  - All three Kalshi events should match the one Poly event
  - Enforcing one-to-one at event level silently drops 2/3 of valid matches

Performance:
  - Event-level: 6,999 × 4,420 = ~30M comparisons (~3 seconds)
  - Market-level: Only markets from ~2K-8K matched event pairs (~6 markets each)
  - Total: ~25M comparisons vs 900M for flat matching
"""

import hashlib
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from config import settings
from matching.embedder import Embedder
from models.market import MatchGroup, UnifiedMarket

logger = logging.getLogger(__name__)

# ── Three-Tier Thresholds ────────────────────────────────────────────────────

# Event-level auto-accept threshold: ≥0.88 = high confidence, trust BGE-M3 completely
_EVENT_AUTO_ACCEPT = 0.88

# Event-level review threshold: 0.70-0.88 = uncertain, log to CSV for human inspection
_EVENT_REVIEW = 0.70

# Market-level threshold comes from settings.min_similarity (default 0.82)
# Applied within each auto-accepted event pair only

# Negation keywords used for outcome-alignment heuristic
_NEGATION_RE = re.compile(
    r"\b(not|no|won'?t|will\s+not|never|fail(?:s|ed)?|lose[sd]?|lost"
    r"|fall[s]?|below|under|miss(?:es|ed)?)\b",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOP_TOKENS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "by",
    "will", "be", "is", "are", "was", "were", "this", "that", "before", "after",
    "with", "from", "as", "who", "what", "when", "where", "how", "which",
    "vs", "v", "match", "game", "event", "market", "season", "week", "today",
}

_GENERIC_OUTCOME_TOKENS = {
    "yes", "no", "winner", "win", "wins", "won", "lose", "loses", "lost",
    "draw", "tie", "tied", "end", "ending", "result", "results", "finish",
    "top", "bottom", "over", "under", "above", "below", "between", "exactly",
}

_DRAW_TOKENS = {"draw", "tie", "tied"}
_WINNER_TOKENS = {"winner", "win", "wins", "won"}


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t]


def _specific_outcome_tokens(market: UnifiedMarket) -> Set[str]:
    """
    Extract outcome-specific tokens by removing event-level context words.
    This helps avoid matching different outcomes within the same event.
    """
    title_tokens = _tokenize(market.title)
    event_tokens = set(_tokenize(market.event_title)) if market.event_title else set()

    specific = {
        t for t in title_tokens
        if t not in _STOP_TOKENS and t not in event_tokens and len(t) > 1
    }

    if specific:
        return specific

    # Fallback: if event/title are near-identical, keep non-stop title tokens.
    return {
        t for t in title_tokens
        if t not in _STOP_TOKENS and len(t) > 1
    }


def _meaningful_tokens(tokens: Set[str]) -> Set[str]:
    return {t for t in tokens if t not in _GENERIC_OUTCOME_TOKENS}


def _outcome_pair_compatible(poly_market: UnifiedMarket, kalshi_market: UnifiedMarket) -> bool:
    """
    Gate candidate market pairs so multi-outcome events align by outcome side.
    Do not require exact wording, but reject clearly incompatible outcomes.
    """
    poly_specific = _specific_outcome_tokens(poly_market)
    kalshi_specific = _specific_outcome_tokens(kalshi_market)

    poly_draw = bool(poly_specific & _DRAW_TOKENS)
    kalshi_draw = bool(kalshi_specific & _DRAW_TOKENS)
    poly_winner = bool(poly_specific & _WINNER_TOKENS)
    kalshi_winner = bool(kalshi_specific & _WINNER_TOKENS)

    # Strong contradiction: draw-vs-winner markets are not the same outcome.
    if (poly_draw and kalshi_winner and not kalshi_draw) or (kalshi_draw and poly_winner and not poly_draw):
        return False

    return True


def _outcome_pair_weight(poly_market: UnifiedMarket, kalshi_market: UnifiedMarket) -> float:
    """
    Soft weighting for candidate scores within an event pair.
    This keeps events in play while nudging greedy assignment toward same-outcome matches.
    """
    poly_specific = _specific_outcome_tokens(poly_market)
    kalshi_specific = _specific_outcome_tokens(kalshi_market)
    poly_meaningful = _meaningful_tokens(poly_specific)
    kalshi_meaningful = _meaningful_tokens(kalshi_specific)

    if poly_meaningful and kalshi_meaningful:
        if poly_meaningful & kalshi_meaningful:
            return 1.08
        return 0.78

    # Unknown/underspecified outcome text: keep candidate mostly unchanged.
    return 0.98


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_review_csv(
    review_pairs: List[Tuple[str, str, str, str, float]],
    output_path: Optional[Path] = None,
) -> None:
    """
    Save review-zone event pairs to CSV for human inspection.
    
    Args:
        review_pairs: List of (poly_event_id, poly_title, kalshi_event_id, kalshi_title, score)
        output_path: Where to save (default: cache/event_matches_review.csv)
    """
    if not review_pairs:
        return
        
    if output_path is None:
        cache_dir = Path(__file__).resolve().parent.parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = cache_dir / "event_matches_review.csv"
    
    try:
        import csv
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['poly_event_id', 'poly_event_title', 'kalshi_event_id', 'kalshi_event_title', 'similarity_score'])
            for poly_id, poly_title, kalshi_id, kalshi_title, score in review_pairs:
                writer.writerow([poly_id, poly_title, kalshi_id, kalshi_title, f"{score:.4f}"])
        
        logger.info("📄 Saved %d review-zone event pairs to: %s", len(review_pairs), output_path)
        logger.info("   Human review recommended for scores 0.70-0.88 (uncertain matches)")
    except Exception as exc:
        logger.warning("Failed to save review CSV: %s", exc)


def _stable_group_id(poly_id: str, kalshi_id: str) -> str:
    """Deterministic 16-char group ID — stable across restarts."""
    return hashlib.sha256(f"{poly_id}:{kalshi_id}".encode()).hexdigest()[:16]


def _infer_outcome_alignment(title_a: str, title_b: str) -> bool:
    """
    Heuristic: if one title contains significantly more negation words than
    the other (odd vs even count), the outcomes are probably inverted.
    When uncertain, returns True (treats as aligned) — safer to miss an
    opportunity than to trade backwards.
    """
    neg_a = len(_NEGATION_RE.findall(title_a))
    neg_b = len(_NEGATION_RE.findall(title_b))
    return (neg_a % 2) == (neg_b % 2)


def _get_device() -> str:
    """Detect best available device for matrix computation."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_matrix(
    row_vecs: List[np.ndarray],
    col_vecs: List[np.ndarray],
) -> np.ndarray:
    """
    GPU-accelerated similarity matrix using torch.
    Vectors must already be L2-normalised (cosine sim = dot product).
    
    Keeps all computation on GPU to avoid repeated CPU↔GPU transfers.
    """
    if not row_vecs or not col_vecs:
        return np.empty((0, 0), dtype=np.float32)

    row_np = np.asarray(row_vecs, dtype=np.float32)
    col_np = np.asarray(col_vecs, dtype=np.float32)

    device = _get_device()

    if device == "cpu":
        return row_np @ col_np.T

    try:
        with torch.inference_mode():
            row_t = torch.from_numpy(row_np).to(device)
            col_t = torch.from_numpy(col_np).to(device)
            sim_t = torch.mm(row_t, col_t.t())
            return sim_t.float().cpu().numpy()
    except Exception as exc:
        logger.debug("GPU matrix build failed, using CPU: %s", exc)
        return row_np @ col_np.T


def _greedy_assign(
    sim_matrix: np.ndarray,
    threshold: float,
    row_ids: List[str],
    col_ids: List[str],
    used_rows: Optional[Set[str]] = None,
    used_cols: Optional[Set[str]] = None,
) -> List[Tuple[str, str, float]]:
    """
    Greedy one-to-one assignment by descending similarity score.
    Returns list of (row_id, col_id, score).
    """
    used_rows = used_rows or set()
    used_cols = used_cols or set()
    
    # All candidate (row, col) pairs above threshold
    candidates = np.argwhere(sim_matrix >= threshold)
    if len(candidates) == 0:
        return []

    scores = sim_matrix[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(-scores)
    candidates = candidates[order]
    scores = scores[order]

    results: List[Tuple[str, str, float]] = []
    local_used_rows: Set[int] = set()
    local_used_cols: Set[int] = set()

    for (ri, ci), score in zip(candidates, scores):
        rid = row_ids[ri]
        cid = col_ids[ci]
        
        # Check both global and local usage
        if rid in used_rows or ri in local_used_rows:
            continue
        if cid in used_cols or ci in local_used_cols:
            continue
            
        results.append((rid, cid, float(score)))
        local_used_rows.add(ri)
        local_used_cols.add(ci)

    return results


# ── Main class ────────────────────────────────────────────────────────────────

class Matcher:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def match(
        self,
        poly_markets: List[UnifiedMarket],
        kalshi_markets: List[UnifiedMarket],
        min_similarity: Optional[float] = None,
    ) -> List[MatchGroup]:
        """
        Run hierarchical event→market matching.
        
        Step 1: Match events by their titles
        Step 2: Match markets within matched event pairs
        
        Returns a list of MatchGroup objects sorted by similarity desc.
        """
        if not poly_markets or not kalshi_markets:
            logger.warning(
                "Matcher: called with empty lists (poly=%d, kalshi=%d)",
                len(poly_markets) if poly_markets else 0,
                len(kalshi_markets) if kalshi_markets else 0,
            )
            return []

        t0 = time.time()
        market_threshold = settings.min_similarity if min_similarity is None else min_similarity

        logger.info(
            "Matcher: starting hierarchical match with %d Polymarket × %d Kalshi markets",
            len(poly_markets), len(kalshi_markets),
        )

        # ══════════════════════════════════════════════════════════════════
        # STEP 1: Group markets by event
        # ══════════════════════════════════════════════════════════════════
        poly_events: Dict[str, List[UnifiedMarket]] = defaultdict(list)
        for m in poly_markets:
            # Use event_id if available, otherwise use market_id as event_id
            event_key = m.event_id or m.market_id
            poly_events[event_key].append(m)

        kalshi_events: Dict[str, List[UnifiedMarket]] = defaultdict(list)
        for m in kalshi_markets:
            event_key = m.event_id or m.market_id
            kalshi_events[event_key].append(m)

        poly_event_ids = list(poly_events.keys())
        kalshi_event_ids = list(kalshi_events.keys())

        logger.info(
            "Grouped into events: %d Polymarket events (%d markets) | %d Kalshi events (%d markets)",
            len(poly_event_ids), len(poly_markets),
            len(kalshi_event_ids), len(kalshi_markets),
        )

        # ══════════════════════════════════════════════════════════════════
        # STEP 2: Embed event titles
        # ══════════════════════════════════════════════════════════════════
        def _get_event_title(event_id: str, event_map: Dict[str, List[UnifiedMarket]]) -> str:
            """Get best title for an event (prefer event_title, fall back to first market title)."""
            markets = event_map[event_id]
            # Use event_title if available and non-empty
            if markets[0].event_title:
                return markets[0].event_title
            # Fall back to market title
            return markets[0].title

        poly_event_titles = [_get_event_title(eid, poly_events) for eid in poly_event_ids]
        kalshi_event_titles = [_get_event_title(eid, kalshi_events) for eid in kalshi_event_ids]
        
        # Deduplicate titles for embedding (same title = same embedding)
        all_event_titles = list(dict.fromkeys(poly_event_titles + kalshi_event_titles))

        logger.info("Embedding %d unique event titles…", len(all_event_titles))
        event_title_embeddings = self._embedder.embed_strings(all_event_titles)

        if not event_title_embeddings:
            logger.error("Failed to embed event titles! Check embedder.")
            return []

        # Build lists of (event_id, title, vector) for events that have embeddings
        poly_events_with_vec = [
            (eid, title, event_title_embeddings.get(title))
            for eid, title in zip(poly_event_ids, poly_event_titles)
            if title in event_title_embeddings
        ]
        
        kalshi_events_with_vec = [
            (eid, title, event_title_embeddings.get(title))
            for eid, title in zip(kalshi_event_ids, kalshi_event_titles)
            if title in event_title_embeddings
        ]

        if not poly_events_with_vec or not kalshi_events_with_vec:
            logger.error(
                "No valid event embeddings (poly=%d, kalshi=%d)",
                len(poly_events_with_vec), len(kalshi_events_with_vec),
            )
            return []

        # ══════════════════════════════════════════════════════════════════
        # STEP 3: Build event similarity matrix and apply three-tier thresholding
        # ══════════════════════════════════════════════════════════════════
        poly_event_vecs = [vec for _, _, vec in poly_events_with_vec]
        kalshi_event_vecs = [vec for _, _, vec in kalshi_events_with_vec]
        poly_event_ids_ok = [eid for eid, _, _ in poly_events_with_vec]
        kalshi_event_ids_ok = [eid for eid, _, _ in kalshi_events_with_vec]

        logger.info(
            "Building event similarity matrix (%d × %d = %s comparisons)…",
            len(poly_event_vecs), len(kalshi_event_vecs),
            f"{len(poly_event_vecs) * len(kalshi_event_vecs):,}",
        )

        event_sim_matrix = _build_matrix(poly_event_vecs, kalshi_event_vecs)

        # Log event similarity stats
        if event_sim_matrix.size > 0:
            event_sim_flat = event_sim_matrix.flatten()
            above_review = np.sum(event_sim_flat >= _EVENT_REVIEW)
            above_auto = np.sum(event_sim_flat >= _EVENT_AUTO_ACCEPT)
            logger.info(
                "Event similarity: min=%.3f, max=%.3f, mean=%.3f",
                float(np.min(event_sim_flat)),
                float(np.max(event_sim_flat)),
                float(np.mean(event_sim_flat)),
            )
            logger.info(
                "Three-tier distribution: %d total | %d above review (≥%.2f) | %d auto-accept (≥%.2f)",
                len(event_sim_flat),
                above_review,
                _EVENT_REVIEW,
                above_auto,
                _EVENT_AUTO_ACCEPT,
            )

        # ══════════════════════════════════════════════════════════════════
        # STEP 4: Apply three-tier thresholding (NO one-to-one at event level)
        # ══════════════════════════════════════════════════════════════════
        # Find all pairs above review threshold
        rows, cols = np.where(event_sim_matrix >= _EVENT_REVIEW)
        
        auto_accept_pairs: List[Tuple[str, str, float]] = []
        review_pairs: List[Tuple[str, str, str, str, float]] = []  # (poly_id, poly_title, kalshi_id, kalshi_title, score)
        
        for i, j in zip(rows, cols):
            score = float(event_sim_matrix[i, j])
            poly_event_id = poly_event_ids_ok[i]
            kalshi_event_id = kalshi_event_ids_ok[j]
            
            if score >= _EVENT_AUTO_ACCEPT:
                # High confidence — auto-accept
                auto_accept_pairs.append((poly_event_id, kalshi_event_id, score))
            else:
                # Review zone (0.70-0.88) — log for human inspection
                poly_title = poly_event_titles[poly_event_ids.index(poly_event_id)]
                kalshi_title = kalshi_event_titles[kalshi_event_ids.index(kalshi_event_id)]
                review_pairs.append((poly_event_id, poly_title, kalshi_event_id, kalshi_title, score))

        logger.info(
            "✓ Three-tier event matching: %d auto-accept (≥%.2f) | %d review (%.2f-%.2f) | %d reject (<%.2f)",
            len(auto_accept_pairs), _EVENT_AUTO_ACCEPT,
            len(review_pairs), _EVENT_REVIEW, _EVENT_AUTO_ACCEPT,
            len(event_sim_flat) - above_review if event_sim_matrix.size > 0 else 0,
            _EVENT_REVIEW,
        )

        # Save review-zone pairs to CSV
        if review_pairs:
            _save_review_csv(review_pairs)

        if not auto_accept_pairs:
            logger.warning(
                "No event pairs auto-accepted! Event threshold %.2f may be too high. "
                "Review the CSV file for potential matches in the review zone (%.2f-%.2f).",
                _EVENT_AUTO_ACCEPT, _EVENT_REVIEW, _EVENT_AUTO_ACCEPT,
            )
            return []

        # ══════════════════════════════════════════════════════════════════
        # STEP 5: Embed markets from auto-accepted event pairs
        # ══════════════════════════════════════════════════════════════════
        target_markets: List[UnifiedMarket] = []
        seen_market_ids: Set[str] = set()
        
        for poly_event_id, kalshi_event_id, _event_score in auto_accept_pairs:
            for market in poly_events[poly_event_id]:
                if market.market_id not in seen_market_ids:
                    target_markets.append(market)
                    seen_market_ids.add(market.market_id)
            for market in kalshi_events[kalshi_event_id]:
                if market.market_id not in seen_market_ids:
                    target_markets.append(market)
                    seen_market_ids.add(market.market_id)

        logger.info(
            "Embedding %d markets from %d auto-accepted event pairs…",
            len(target_markets), len(auto_accept_pairs),
        )

        market_embeddings = self._embedder.embed_markets(target_markets)

        if not market_embeddings:
            logger.error("Failed to embed markets! Check embedder.")
            return []

        logger.info(
            "Generated embeddings for %d/%d markets",
            len(market_embeddings), len(target_markets),
        )

        # ══════════════════════════════════════════════════════════════════
        # STEP 6: Match markets within each auto-accepted event pair (one-to-one)
        # ══════════════════════════════════════════════════════════════════
        groups: List[MatchGroup] = []
        used_poly_market_ids: Set[str] = set()
        used_kalshi_market_ids: Set[str] = set()
        outcome_filtered_pairs = 0

        for poly_event_id, kalshi_event_id, event_score in auto_accept_pairs:
            poly_markets_in_event = poly_events[poly_event_id]
            kalshi_markets_in_event = kalshi_events[kalshi_event_id]

            # Get markets with valid embeddings
            poly_with_vec = [
                (m, market_embeddings.get(m.market_id))
                for m in poly_markets_in_event
                if m.market_id in market_embeddings
            ]
            kalshi_with_vec = [
                (m, market_embeddings.get(m.market_id))
                for m in kalshi_markets_in_event
                if m.market_id in market_embeddings
            ]

            if not poly_with_vec or not kalshi_with_vec:
                continue

            poly_market_vecs = [vec for _, vec in poly_with_vec]
            kalshi_market_vecs = [vec for _, vec in kalshi_with_vec]
            poly_market_ids = [m.market_id for m, _ in poly_with_vec]
            kalshi_market_ids = [m.market_id for m, _ in kalshi_with_vec]

            # Build market similarity matrix for this event pair
            market_sim_matrix = _build_matrix(poly_market_vecs, kalshi_market_vecs)

            # Outcome-side compatibility filter:
            # keep event pair, but prevent assigning obviously different outcomes.
            for ri, (poly_market, _) in enumerate(poly_with_vec):
                for ci, (kalshi_market, _) in enumerate(kalshi_with_vec):
                    if not _outcome_pair_compatible(poly_market, kalshi_market):
                        market_sim_matrix[ri, ci] = -1.0
                        outcome_filtered_pairs += 1
                    else:
                        weighted = market_sim_matrix[ri, ci] * _outcome_pair_weight(poly_market, kalshi_market)
                        market_sim_matrix[ri, ci] = min(0.999, float(weighted))

            # Greedy assign markets within this event pair (one-to-one at market level)
            market_assignments = _greedy_assign(
                market_sim_matrix,
                market_threshold,
                poly_market_ids,
                kalshi_market_ids,
                used_rows=used_poly_market_ids,
                used_cols=used_kalshi_market_ids,
            )

            # Create MatchGroup objects
            poly_markets_by_id = {m.market_id: m for m, _ in poly_with_vec}
            kalshi_markets_by_id = {m.market_id: m for m, _ in kalshi_with_vec}

            for poly_mid, kalshi_mid, market_score in market_assignments:
                poly_market = poly_markets_by_id[poly_mid]
                kalshi_market = kalshi_markets_by_id[kalshi_mid]

                aligned = _infer_outcome_alignment(poly_market.title, kalshi_market.title)

                groups.append(MatchGroup(
                    group_id=_stable_group_id(poly_market.market_id, kalshi_market.market_id),
                    poly_market=poly_market,
                    kalshi_market=kalshi_market,
                    similarity_score=market_score,
                    outcomes_aligned=aligned,
                ))

                used_poly_market_ids.add(poly_mid)
                used_kalshi_market_ids.add(kalshi_mid)

        # ══════════════════════════════════════════════════════════════════
        # FINAL: Sort and return
        # ══════════════════════════════════════════════════════════════════
        groups.sort(key=lambda g: g.similarity_score, reverse=True)
        elapsed = time.time() - t0

        logger.info(
            "✓ Three-tier hierarchical matching complete in %.1fs:",
            elapsed,
        )
        logger.info(
            "  Event level: %d auto-accept (≥%.2f) | %d review (%.2f-%.2f)",
            len(auto_accept_pairs), _EVENT_AUTO_ACCEPT,
            len(review_pairs), _EVENT_REVIEW, _EVENT_AUTO_ACCEPT,
        )
        logger.info(
            "  Market level: %d match groups (threshold=%.2f, one-to-one within event pairs)",
            len(groups), market_threshold,
        )
        logger.info(
            "  Outcome compatibility: filtered %d incompatible market-pair candidates",
            outcome_filtered_pairs,
        )
        
        # Log sample match groups for diagnostics
        if groups:
            sample_count = min(5, len(groups))
            import random
            samples = random.sample(groups, sample_count) if len(groups) > sample_count else groups
            logger.info("Sample match groups (for diagnostics):")
            for i, g in enumerate(samples, 1):
                poly_p, kalshi_p = g.get_comparable_prices()
                logger.info(
                    "  %d. sim=%.3f aligned=%s | poly=%.4f kalshi=%.4f spread=%.4f",
                    i, g.similarity_score, g.outcomes_aligned, poly_p, kalshi_p, abs(poly_p - kalshi_p),
                )
                logger.info(
                    "     poly: '%s' (yes=%.4f no=%.4f has_price=%s)",
                    g.poly_market.title[:60], g.poly_market.yes_price, g.poly_market.no_price, g.poly_market.has_price,
                )
                logger.info(
                    "     kalshi: '%s' (yes=%.4f no=%.4f has_price=%s)",
                    g.kalshi_market.title[:60], g.kalshi_market.yes_price, g.kalshi_market.no_price, g.kalshi_market.has_price,
                )

        return groups
