"""
matching/embedder.py

Wraps BAAI/bge-large-en-v1.5 via sentence-transformers.

Cache design
────────────
Two complementary dicts keep everything consistent:

  _vec_cache: Dict[str, np.ndarray]
      key = title string → embedding vector
      Single source of truth for vectors.

  _id_map: Dict[str, str]
      key = market_id → title string
      Lets us look up a vector by market ID without storing duplicates.

This means embed_strings and embed_markets share the same model cache
transparently — if a market title was already embedded (regardless of how
it entered the cache), we never re-embed it.

GPU support
───────────
Priority order:  CUDA (Nvidia) → MPS (Apple Silicon) → CPU

If CUDA is reported unavailable even though you have an Nvidia GPU, the
most likely cause is that PyTorch was installed without CUDA support:

    pip install torch --index-url https://download.pytorch.org/whl/cu124

Replace cu124 with your CUDA version (e.g. cu118 for CUDA 11.8).
Run `nvidia-smi` to check your driver version → find matching wheel at
https://pytorch.org/get-started/locally/
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import settings
from models.market import UnifiedMarket

logger = logging.getLogger(__name__)

# Multilingual model with 1024-dim embeddings
# Trained specifically for semantic equivalence across different phrasings
_MODEL_NAME = "BAAI/bge-m3"


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_title(title: str) -> str:
    return " ".join(title.strip().lower().split())


def _cache_path() -> Path:
    configured = Path(settings.embedding_cache_path)
    if configured.is_absolute():
        return configured
    return Path(__file__).resolve().parent.parent / configured


def _detect_device() -> Tuple[str, int]:
    """
    Detect the best available device and return (device_str, batch_size).

    Emits detailed diagnostics so GPU problems are easy to spot in logs.
    """
    # ── CUDA (Nvidia) ────────────────────────────────────────────────────
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        if settings.embedding_batch_size_override > 0:
            batch_size = settings.embedding_batch_size_override
        else:
            batch_size = max(64, int(vram_gb * settings.embedding_batch_vram_factor))
        logger.info(
            "GPU ✓  CUDA device: %s  VRAM: %.1f GB  batch_size: %d",
            props.name, vram_gb, batch_size,
        )
        return "cuda", batch_size

    # ── MPS (Apple Silicon) ──────────────────────────────────────────────
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        batch_size = settings.embedding_batch_size_override or 128
        logger.info("GPU ✓  MPS device (Apple Silicon)  batch_size: %d", batch_size)
        return "mps", batch_size

    # ── CPU fallback — explain why GPU is absent ─────────────────────────
    cpu_batch = settings.embedding_batch_size_override or 32
    logger.warning(
        "GPU ✗  No CUDA or MPS device found — running on CPU (slow for large batches).\n"
        "  • If you have an Nvidia GPU: make sure PyTorch was installed with CUDA support:\n"
        "      pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
        "    (Replace cu124 with your CUDA version. Check with `nvidia-smi`.)\n"
        "  • CUDA available according to torch: %s\n"
        "  • torch version: %s",
        torch.cuda.is_available(),
        torch.__version__,
    )
    return "cpu", cpu_batch


class Embedder:
    def __init__(self) -> None:
        device, batch_size = _detect_device()

        logger.info("Loading %s on %s — ~10 s on first run…", _MODEL_NAME, device)
        self._model = SentenceTransformer(_MODEL_NAME, device=device)
        self._device = device
        self._batch_size = batch_size

        # normalized_title_str → np.ndarray (L2-normalised)
        self._vec_cache: Dict[str, np.ndarray] = {}
        # market_id → normalized_title_str
        self._id_map: Dict[str, str] = {}
        self._persistent_enabled = settings.embedding_cache_enabled
        self._persistent_path = _cache_path()
        self._dirty = False
        self._last_persist = 0.0

        # Verify model is on correct device (some sentence-transformers versions need explicit move)
        if device == "cuda" and torch.cuda.is_available():
            self._model = self._model.cuda()
            logger.info("✓ Model explicitly moved to CUDA.")
        elif device == "mps" and hasattr(torch.backends, "mps"):
            self._model = self._model.to("mps")
            logger.info("✓ Model explicitly moved to MPS.")

        self._load_persistent_cache()
        logger.info("Embedder ready on %s.", device)

    def _cache_meta(self) -> Dict[str, Any]:
        return {
            "model_name": _MODEL_NAME,
            "vector_dim": int(getattr(self._model, "get_sentence_embedding_dimension")()),
        }

    def _load_persistent_cache(self) -> None:
        if not self._persistent_enabled:
            return
        try:
            if not self._persistent_path.exists():
                return
            with open(self._persistent_path, "rb") as f:
                payload = pickle.load(f)

            if not isinstance(payload, dict):
                logger.warning("Embedding cache payload is invalid; ignoring.")
                return

            meta = payload.get("meta", {})
            vectors = payload.get("vectors", {})
            if not isinstance(meta, dict) or not isinstance(vectors, dict):
                logger.warning("Embedding cache structure is invalid; ignoring.")
                return

            current_meta = self._cache_meta()
            if meta.get("model_name") != current_meta["model_name"]:
                logger.warning(
                    "Embedding cache model mismatch: cached=%s, current=%s. "
                    "Cache will be rebuilt from scratch.",
                    meta.get("model_name"), current_meta["model_name"],
                )
                return
            if int(meta.get("vector_dim", 0)) != current_meta["vector_dim"]:
                logger.warning(
                    "Embedding cache vector dim mismatch: cached=%d, current=%d. "
                    "Cache will be rebuilt.",
                    int(meta.get("vector_dim", 0)), current_meta["vector_dim"],
                )
                return

            restored = 0
            for key, vec in vectors.items():
                if not isinstance(key, str):
                    continue
                if isinstance(vec, np.ndarray):
                    arr = vec.astype(np.float32, copy=False)
                else:
                    arr = np.asarray(vec, dtype=np.float32)
                if arr.ndim != 1:
                    continue
                self._vec_cache[key] = arr
                restored += 1

            logger.info("Loaded %d embeddings from disk cache.", restored)
        except Exception as exc:
            logger.warning("Failed to load embedding cache: %s", exc)

    def _persist_cache(self, force: bool = False) -> None:
        if not self._persistent_enabled or not self._dirty:
            return

        now = time.time()
        if not force and (now - self._last_persist) < settings.embedding_cache_save_interval_seconds:
            return

        try:
            self._persistent_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "meta": self._cache_meta(),
                "timestamp": now,
                "count": len(self._vec_cache),
                "vectors": self._vec_cache,
            }
            tmp = self._persistent_path.with_suffix(self._persistent_path.suffix + ".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(self._persistent_path)
            self._last_persist = now
            self._dirty = False
        except Exception as exc:
            logger.warning("Failed to persist embedding cache: %s", exc)

    # ── Public embedding methods ─────────────────────────────────────────

    def embed_strings(self, strings: List[str]) -> Dict[str, np.ndarray]:
        """
        Embed a list of raw title strings.
        Returns {title_str: vector} for every string in the input.
        Skips strings that are already cached.
        """
        pairs = [(s, _normalize_title(s)) for s in strings if s]
        pairs = [(raw, norm) for raw, norm in pairs if norm]

        unique_new: Dict[str, str] = {}
        for raw, norm in pairs:
            if norm not in self._vec_cache and norm not in unique_new:
                unique_new[norm] = raw

        if unique_new:
            norms = list(unique_new.keys())
            texts = _dedupe_keep_order([unique_new[n] for n in norms])
            vecs: np.ndarray = self._model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 200,
            )
            if len(texts) == len(norms):
                for norm, vec in zip(norms, vecs):
                    self._vec_cache[norm] = np.asarray(vec, dtype=np.float32)
            else:
                text_to_vec = {
                    text: np.asarray(vec, dtype=np.float32) for text, vec in zip(texts, vecs)
                }
                for norm in norms:
                    text = unique_new[norm]
                    vec = text_to_vec.get(text)
                    if vec is not None:
                        self._vec_cache[norm] = vec
            self._dirty = True
            self._persist_cache()

        out: Dict[str, np.ndarray] = {}
        for raw, norm in pairs:
            vec = self._vec_cache.get(norm)
            if vec is not None:
                out[raw] = vec
        return out

    def embed_markets(self, markets: List[UnifiedMarket]) -> Dict[str, np.ndarray]:
        """
        Embed a list of UnifiedMarket objects.
        Returns {market_id: vector} for every market in the input.
        Re-embeds a market only if its title has changed since last time.
        """
        if not markets:
            return {}

        new_markets = [
            m for m in markets
            if m.market_id not in self._id_map
            or self._id_map[m.market_id] != _normalize_title(m.title)
        ]

        if new_markets:
            unique_new: Dict[str, str] = {}
            for market in new_markets:
                norm = _normalize_title(market.title)
                if norm and norm not in self._vec_cache and norm not in unique_new:
                    unique_new[norm] = market.title

            if unique_new:
                norms = list(unique_new.keys())
                texts = _dedupe_keep_order([unique_new[n] for n in norms])
                vecs: np.ndarray = self._model.encode(
                    texts,
                    batch_size=self._batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=len(texts) > 200,
                )
                if len(texts) == len(norms):
                    for norm, vec in zip(norms, vecs):
                        self._vec_cache[norm] = np.asarray(vec, dtype=np.float32)
                else:
                    text_to_vec = {
                        text: np.asarray(vec, dtype=np.float32) for text, vec in zip(texts, vecs)
                    }
                    for norm in norms:
                        text = unique_new[norm]
                        vec = text_to_vec.get(text)
                        if vec is not None:
                            self._vec_cache[norm] = vec
                self._dirty = True
                self._persist_cache()

            for market in new_markets:
                self._id_map[market.market_id] = _normalize_title(market.title)

        # CRITICAL: Always update _id_map for ALL markets, not just new ones.
        # This ensures cached markets can still be looked up by market_id.
        for market in markets:
            norm = _normalize_title(market.title)
            if norm:
                self._id_map[market.market_id] = norm

        cached_new = len(markets) - len(new_markets)
        logger.debug(
            "embed_markets: %d total, %d new, %d cached, _id_map size: %d",
            len(markets), len(new_markets), cached_new, len(self._id_map),
        )

        out: Dict[str, np.ndarray] = {}
        for market in markets:
            norm = _normalize_title(market.title)
            vec = self._vec_cache.get(norm)
            if vec is not None:
                out[market.market_id] = vec
        return out

    # ── Lookup helpers ───────────────────────────────────────────────────

    def get_vector_by_title(self, title: str) -> Optional[np.ndarray]:
        """Return the cached vector for a title string, or None."""
        return self._vec_cache.get(_normalize_title(title))

    def get_vector_by_market_id(self, market_id: str) -> Optional[np.ndarray]:
        """Return the cached vector for a market ID, or None."""
        title = self._id_map.get(market_id)
        if title is None:
            return None
        return self._vec_cache.get(title)

    def get_vector_for_market(self, market: UnifiedMarket) -> Optional[np.ndarray]:
        """Return cached vector for a market (looks up by both id and title)."""
        # Try by market_id first (most specific), then by title
        vec = self.get_vector_by_market_id(market.market_id)
        if vec is None:
            vec = self._vec_cache.get(_normalize_title(market.title))
        return vec

    # ── Cache management ─────────────────────────────────────────────────

    def cache_size(self) -> int:
        return len(self._vec_cache)

    def flush(self) -> None:
        self._persist_cache(force=True)

    def evict(self, market_ids: List[str]) -> None:
        """Remove stale market IDs from the id_map (vectors stay for title reuse)."""
        for mid in market_ids:
            self._id_map.pop(mid, None)
