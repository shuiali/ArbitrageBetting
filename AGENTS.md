# AGENTS.md — AI Handoff Document
# Prediction Market Arbitrage Bot

This document is written for an AI agent that will continue developing this
project. It explains what the project is, why each decision was made, what
has been built, what is missing, and how to extend it correctly.

Read this entire document before touching any code.

---

## 1. What this project is

This is a **backend system that detects arbitrage opportunities across
prediction markets**. It compares prices for the same real-world event
listed on two platforms — Polymarket and Kalshi — and alerts a human
trader via Telegram when a profitable spread exists.

### What "arbitrage" means here

In a binary prediction market, a contract pays $1 if an event happens
(YES) and $0 if it doesn't (NO). Prices are probabilities: a YES price
of 0.42 means the market thinks there is a 42% chance of the event.

If the same event is listed on two platforms at different prices, you can
buy YES on the cheap platform and buy NO on the expensive platform. One of
them must win. The profit is the price difference.

Example:
  - Polymarket: YES at 0.40
  - Kalshi:     YES at 0.52
  - Buy YES on Polymarket for $0.40
  - Buy NO on Kalshi for $0.48 (= $1 − $0.52)
  - Total cost: $0.88
  - Guaranteed payout: $1.00 (one leg always wins)
  - Gross profit: $0.12 (12%)

The core challenge is not the maths — it is finding which markets on two
platforms describe the same event, because they are worded differently.

---

## 2. Core design decisions (do not reverse these)

**Python with asyncio.** All I/O is async. The main loop uses
`asyncio.gather` to fetch both platforms concurrently. Do not introduce
synchronous blocking calls into the fetch path.

**BGE-M3 for semantic matching with three-tier thresholding.** The model
`BAAI/bge-m3` from HuggingFace is used via `sentence-transformers`. It produces
1024-dim L2-normalised dense vectors. Because they are L2-normalised, cosine
similarity equals the dot product, which is fast in numpy.

**Three-tier thresholding system:**
- **Event auto-accept** (≥0.88): High confidence matches, trusted completely
- **Event review zone** (0.70-0.88): Uncertain matches, logged to CSV for human review
- **Event reject** (<0.70): Genuinely unrelated events, discarded
- **Market threshold** (≥0.82): Applied within auto-accepted event pairs only

This system eliminates the single-threshold problem: too high misses matches,
too low floods with false positives. The review zone (typically ~3K-12K pairs)
can be inspected by humans in under an hour to catch edge cases.

**No WebSocket, no streaming.** The current design uses only REST polling.
WebSocket support is intentionally deferred for a future phase. Do not
add WebSocket logic unless explicitly asked to.

**No database.** State is held entirely in memory in `storage/state.py`.
The system is designed so that a future developer can swap `StateManager`
for a Redis or PostgreSQL-backed version by implementing the same interface.
Do not add a database dependency without replacing `StateManager` cleanly.

**All prices in probability space [0, 1].** Every connector normalises
prices to this range before returning a `UnifiedMarket`. The rest of the
system always works in [0, 1]. Kalshi uses cents [0, 100] internally —
this conversion happens only inside `connectors/kalshi.py`.

**Many-to-many at event level, one-to-one at market level.** Platforms
structure events differently. Polymarket might have one event "2024 US Election"
with 12 markets, while Kalshi splits this into three events ("Rep nominee wins",
"Dem nominee wins", "Third party wins") with 4 markets each. All three Kalshi
events should match the one Poly event.

The matcher allows **many-to-many matching at event level** (same event can
appear in multiple pairs) but enforces **one-to-one at market level** within
each event pair. This prevents missing valid opportunities while avoiding
duplicate market matches.

**Conservative outcome alignment.** The `_infer_outcome_alignment`
function in `matching/matcher.py` uses a negation-keyword heuristic. It
errs toward calling outcomes aligned (returning `True`) when uncertain.
This means it may miss an inverted pair rather than trade it backwards.
Missing an opportunity is safe. Trading backwards is catastrophic.

---

## 3. Complete file map with purpose

```
arb_bot/
│
├── main.py                     ENTRY POINT. The async run loop.
│                               Orchestrates all components.
│                               Cycle: fetch → match (if due) → arb scan → alert.
│
├── config.py                   All runtime settings. Loads from .env.
│                               Exposes a frozen Settings dataclass as `settings`.
│                               Import `settings` anywhere — do not read os.environ directly.
│
├── models/
│   └── market.py               THE ONLY TYPES THAT CROSS MODULE BOUNDARIES.
│                               UnifiedMarket   — normalised market from any platform
│                               MatchGroup      — a confirmed same-event pair
│                               ArbOpportunity  — a detected profitable spread
│                               Platform        — enum: POLYMARKET | KALSHI
│
├── connectors/
│   ├── base.py                 Abstract base class. One method: fetch_all() → List[UnifiedMarket].
│   ├── polymarket.py           Polymarket CLOB REST API. Public, no auth.
│   │                           Paginated via `next_cursor`. Last page cursor = "LTE=".
│   │                           Prices already in [0,1]. Parses `tokens` list for YES/NO.
│   └── kalshi.py               Kalshi Trade API v2. Requires RSA auth on production.
│                               Paginated via `cursor`. Last page = empty cursor.
│                               Prices in cents [0,100] → converted to [0,1].
│                               Uses KalshiSigner from utils/kalshi_auth.py.
│
├── utils/
│   ├── http.py                 Shared aiohttp helper. fetch_json() with exponential
│   │                           backoff retry (4 attempts, 1.5^n seconds).
│   │                           Both connectors use this — do not write raw aiohttp calls.
│   └── kalshi_auth.py          RSA-PS256 request signing for Kalshi.
│                               KalshiSigner loads the PEM key once and exposes
│                               headers_for(method, path). Signs: timestamp + method + path.
│                               If key not configured, returns empty headers (works on demo).
│
├── matching/
│   ├── embedder.py             Wraps BAAI/bge-m3 via sentence-transformers.
│   │                           Maintains an in-memory cache keyed by market_id.
│   │                           Only calls the model for titles not seen before.
│   │                           Returns Dict[market_id, np.ndarray].
│   └── matcher.py              Hierarchical event→market matching with three-tier thresholding.
│                               Stage 1: Group markets by event_id
│                               Stage 2: Embed all event titles (batch)
│                               Stage 3: Event similarity matrix (~30M comparisons, ~3 sec)
│                               Stage 4: Three-tier thresholding (auto/review/reject)
│                               Stage 5: Embed markets from auto-accepted event pairs only
│                               Stage 6: One-to-one market matching within event pairs
│                               Saves review-zone pairs to cache/event_matches_review.csv
│
├── arbitrage/
│   └── engine.py               Pure maths, no I/O.
│                               find_opportunities(groups) → List[ArbOpportunity].
│                               For each MatchGroup: get_comparable_prices() → compute
│                               gross_spread → check vs MIN_NET_ROI.
│                               Determines which platform to buy YES on (the cheaper one).
│
├── storage/
│   └── state.py                In-memory state manager.
│                               Stores current MatchGroup list.
│                               Tracks alert cooldowns: dedup_key → last alert timestamp.
│                               Exposes: update_match_groups, get_match_groups,
│                               needs_rematch, should_alert, mark_alerted.
│
├── alerts/
│   └── telegram_bot.py         Sends alerts via Telegram Bot API (raw HTTP, no library).
│                               Uses MarkdownV2 formatting. All special chars must be escaped
│                               with _esc() before inserting into the message template.
│                               send_alert(opp) is the primary method.
│
└── scripts/
    ├── test_connectors.py      Validates both APIs respond and data parses correctly.
    │                           Run this first before the bot. Prints sample markets.
    ├── inspect_matches.py      Runs BGE-M3 matching and prints/saves all match groups.
    │                           Use this to tune MIN_SIMILARITY. Outputs CSV.
    └── scan_once.py            Full end-to-end scan: fetch → match → arb → print.
                                No Telegram. Use this to verify maths before going live.
```

---

## 4. Data flow (exact sequence)

```
[Polymarket REST API]          [Kalshi REST API]
        │                              │
        └──────── asyncio.gather ──────┘
                        │
               List[UnifiedMarket]        (prices in [0,1], valid only)
                        │
          ┌─────────────┴──────────────┐
          │                            │
   (every 30 min)               (every cycle)
          │                            │
   1. Group by event_id        _refresh_prices()
   2. Embed event titles       Patch yes/no prices in
   3. Event similarity (~30M)  existing MatchGroup objects
   4. Three-tier threshold:           │
      - Auto-accept ≥0.88             │
      - Review 0.70-0.88 → CSV        │
      - Reject <0.70                  │
   5. Embed markets (auto only)       │
   6. Market matching (1-to-1)        │
   → List[MatchGroup]                 │
          │                            │
          └─────────────┬──────────────┘
                        │
              StateManager.get_match_groups()
                        │
              ArbEngine.find_opportunities()
              → List[ArbOpportunity]  sorted by net_roi desc
                        │
              StateManager.should_alert(opp)?
                        │
                   Yes  │  No (cooldown)
                        │
              TelegramBot.send_alert(opp)
              StateManager.mark_alerted(opp)
```

---

## 5. The `UnifiedMarket` contract

Every connector must produce `UnifiedMarket` objects satisfying these
invariants. If a parsed market fails `is_valid()`, it is silently dropped.

```python
platform:    Platform enum (POLYMARKET or KALSHI)
market_id:   str — native platform ID. Stable, unique per platform.
title:       str — the question text. Used as input to BGE-M3.
             Must be the actual question, not a category label.
yes_price:   float in [0.01, 0.99] — mid-price for YES outcome
no_price:    float in [0.01, 0.99] — mid-price for NO outcome
volume_usd:  float > 0 — total traded volume in USD
url:         str — direct link to the market page
fetched_at:  float — unix timestamp, set automatically
```

`is_valid()` rejects: prices outside [0.01, 0.99], zero volume.

`implied_sum` = yes_price + no_price. In an efficient market this is ~1.0.
Values >1.0 are normal (platform overround). Values <0.95 suggest stale
bid/ask spreads and should be treated with suspicion.

---

## 6. The `MatchGroup` contract

```python
group_id:        str — SHA256[:16] of "{poly_market_id}:{kalshi_market_id}"
                 Stable and deterministic. Used as dedup key in StateManager.
poly_market:     UnifiedMarket — the Polymarket leg
kalshi_market:   UnifiedMarket — the Kalshi leg
similarity_score: float in [0, 1] — BGE-M3 cosine similarity of titles
outcomes_aligned: bool — True means YES on both resolve on the same condition.
                  False means YES on Polymarket = NO on Kalshi.
matched_at:      float — unix timestamp of when the match was created
```

`get_comparable_prices()` returns `(poly_yes_equiv, kalshi_yes_equiv)` —
both floats representing the cost to back the SAME real-world outcome.
When `outcomes_aligned=False`, the kalshi equiv is `kalshi_market.no_price`.

---

## 7. The `ArbOpportunity` contract

```python
group_id:        str — inherited from MatchGroup
poly_market:     UnifiedMarket
kalshi_market:   UnifiedMarket
buy_platform:    Platform — where to buy YES (the cheaper side)
buy_price:       float — the YES price on the buy platform
sell_platform:   Platform — where to buy NO (the expensive side)
sell_price:      float — the YES price on the sell platform
                 (to buy NO you pay 1 - sell_price)
gross_spread:    float — abs(poly_yes_equiv - kalshi_yes_equiv)
detected_at:     float — unix timestamp
```

`dedup_key` = "{group_id}:{buy_platform}" — identifies a unique directional
opportunity. The same event can have TWO dedup keys (one per direction)
but in practice only one side can be profitable at a time.

---

## 8. Configuration reference

All settings live in `config.py` and are loaded from `.env`.
Never read `os.environ` directly in any module. Always import `settings`.

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | str | required | From @BotFather |
| `TELEGRAM_CHAT_ID` | str | required | Target chat or group ID |
| `KALSHI_API_KEY_ID` | str | "" | UUID from kalshi.com API Keys page |
| `KALSHI_PRIVATE_KEY_PATH` | str | "" | Path to RSA PEM private key file |
| `KALSHI_ENV` | str | "production" | "production" or "demo" |
| `MIN_NET_ROI` | float | 0.02 | Minimum net ROI (0.02 = 2%) |
| `MIN_SIMILARITY` | float | 0.82 | Market-level matching threshold (part of three-tier system) |
| `ALERT_COOLDOWN_SECONDS` | int | 300 | Re-alert same opportunity after N seconds |
| `POLL_INTERVAL_SECONDS` | int | 60 | How often to fetch prices |
| `REMATCH_INTERVAL_SECONDS` | int | 1800 | How often to re-run semantic matching |
| `LOG_LEVEL` | str | "INFO" | Python logging level |

**Three-tier thresholds** (hardcoded in `matching/matcher.py`):
- Event auto-accept: ≥0.88
- Event review zone: 0.70-0.88 (logged to `cache/event_matches_review.csv`)
- Event reject: <0.70
- Market matching: ≥0.82 (from `MIN_SIMILARITY` setting above)

---

## 9. API details for each platform

### Polymarket (CLOB API)

Base URL: `https://clob.polymarket.com`

Authentication: None required. Fully public.

Fetch all open markets:
```
GET /markets?limit=500&active=true&next_cursor=<cursor>
```

Response shape:
```json
{
  "data": [
    {
      "condition_id": "0xabc...",
      "question": "Will X happen?",
      "active": true,
      "closed": false,
      "tokens": [
        {"outcome": "Yes", "price": "0.65"},
        {"outcome": "No",  "price": "0.35"}
      ],
      "volume": "125000.50",
      "market_slug": "will-x-happen"
    }
  ],
  "next_cursor": "abc123"
}
```

Pagination ends when `next_cursor` equals `"LTE="` (base64 for "-1").

Prices: already in [0, 1] as strings. Parse to float.

URL template: `https://polymarket.com/event/{market_slug}`

### Kalshi (Trade API v2)

Base URL: `https://trading-api.kalshi.com/trade-api/v2` (production)
          `https://demo-api.kalshi.co/trade-api/v2` (demo)

Authentication (production): RSA-PS256 signed headers.
Three headers required on every request:
```
KALSHI-ACCESS-KEY:       <key_id UUID>
KALSHI-ACCESS-TIMESTAMP: <unix milliseconds as string>
KALSHI-ACCESS-SIGNATURE: <base64(RSA-PS256(timestamp + METHOD + path))>
```
The `path` for signing is only the URL path, not the full URL and not
including the query string. e.g. `/trade-api/v2/markets`.

Fetch all open markets:
```
GET /markets?limit=200&status=open&cursor=<cursor>
```

Response shape:
```json
{
  "markets": [
    {
      "ticker": "PRES-2024-DEM",
      "series_ticker": "PRES-2024",
      "title": "Democratic presidential nominee wins",
      "status": "open",
      "yes_bid": 42,
      "yes_ask": 44,
      "no_bid": 56,
      "no_ask": 58,
      "last_price": 43,
      "volume": 85000
    }
  ],
  "cursor": "next_page_token"
}
```

Pagination ends when `cursor` is empty string or absent.

Prices: in cents [0, 100]. Divide by 100 to get [0, 1].
Mid-price = (bid + ask) / 2. Fall back to `last_price` if bid/ask absent.

Volume: number of contracts traded. Each contract is worth $1 max.
Approximate USD: `volume_contracts × avg_price` where avg = yes_mid/100.

URL template: `https://kalshi.com/markets/{series_ticker}/{ticker}`

---

## 10. Known limitations and issues

**Outcome alignment is heuristic-based.** The `_infer_outcome_alignment`
function uses negation keyword counting, which is imperfect. There is no
guarantee it is correct for all market phrasings. Before trading real money,
run `scripts/inspect_matches.py --save matches.csv` and manually review the
`outcomes_aligned` column for every matched pair. Any row marked `False`
should be examined carefully.

**Kalshi volume calculation is approximate.** Kalshi returns a count of
contracts traded, not a USD volume figure. We approximate with
`contracts × avg_price`. This is fine for filtering (we use volume > 0
to eliminate illiquid markets) but should not be used for position sizing.

**`requirements.txt` is missing `cryptography`.** The `utils/kalshi_auth.py`
module imports `cryptography.hazmat.primitives`. This package must be added
to `requirements.txt` for Kalshi RSA signing to work:
```
cryptography>=42.0.0
```

**No persistence between restarts.** `StateManager` is entirely in-memory.
If the process restarts, all match groups are lost and must be recomputed.
The first cycle after restart will always trigger a full rematch.

**No position sizing.** The bot calculates ROI but not how much capital
to deploy. Actual arbitrage requires splitting capital between both legs
and accounting for order book depth (slippage). This is entirely absent.

**No execution.** The bot alerts a human but does not place trades. All
buying and selling must be done manually through the platform UIs.

**BGE-M3 is slow on first run.** The model is ~2GB and must be downloaded
from HuggingFace on first run. Subsequent runs use the local cache. In
Docker, mount a volume at `/app/.cache/huggingface` to persist this cache.

---

## 11. What has NOT been built yet (future work)

These features are known to be missing and are intended for future
development phases. They are listed in rough priority order.

### 11a. Persistence layer (high priority)

Replace `storage/state.py` with a proper backend. The `StateManager`
interface is the abstraction boundary — implement the same public methods
against Redis (for live state) and PostgreSQL (for history/logging).

Tables needed:
- `match_groups` — store matched pairs with their group_id and metadata
- `opportunities` — log every detected opportunity with timestamp and ROI
- `alerts_sent` — replace the in-memory `_alerted` dict with a DB table
- `market_snapshots` — price history per market_id for charting

### 11b. WebSocket / real-time price feeds (high priority)

The current design polls every `POLL_INTERVAL_SECONDS`. This is too slow
for fast-moving markets. Both Polymarket and Kalshi have WebSocket APIs
for live price streaming.

The recommended approach: use REST for the initial full market load on
startup, then switch to WebSocket for incremental price updates. The
connector interface in `connectors/base.py` would need a `subscribe()`
method returning an async generator of `UnifiedMarket` price updates.

### 11c. Web dashboard (medium priority)

The owner wants a website as the eventual product. The backend is
designed to support this. Build a REST API layer (FastAPI recommended)
in front of the existing components that exposes:

```
GET  /api/status              — bot health, last cycle time, market counts
GET  /api/opportunities       — current live opportunities
GET  /api/groups              — all active match groups
GET  /api/groups/{group_id}   — detail for one group with price history
GET  /api/history             — past opportunities log
```

The `storage/state.py` module is intentionally simple so it can be
replaced with DB-backed implementations without changing the API layer.

### 11d. Third exchange connector (medium priority)

The architecture explicitly supports more than two platforms. Adding a
third exchange requires:
1. A new file `connectors/manifold.py` (or whichever platform)
   implementing `BaseConnector.fetch_all()`.
2. Adding `Platform.MANIFOLD` to the `Platform` enum in `models/market.py`.
3. Extending `MatchGroup` to hold a third optional leg, or changing the
   matching strategy to allow N-way groups.
4. Extending `ArbEngine` to handle three-leg arbitrage.

Note: adding a third exchange significantly increases the complexity of
the matching step (N_poly × N_kalshi × N_manifold candidates).

### 11e. Position sizing and execution (low priority, phase 3)

Automated execution requires:
- Order book depth fetching to estimate slippage
- Capital allocation logic (Kelly criterion or fixed fraction)
- Execution via platform APIs (place order, check fill, handle partial fills)
- Circuit breaker: halt execution if unexpected losses exceed threshold
- A "paper trading" mode that simulates execution without real orders

Do NOT build this until the detection accuracy is validated over weeks of
real-market operation.

### 11f. Alert improvements

The Telegram bot sends one message format. Improvements:
- `/status` command: return current bot stats, last cycle time
- `/opportunities` command: return current open opportunities on demand
- `/groups` command: show top-N match groups by spread
- Inline keyboard buttons: "View on Polymarket", "View on Kalshi"
- Rate limiting per user if bot is shared with a group

---

## 12. How to run the project

### Prerequisites
- Python 3.12+
- ~2GB disk for BGE-M3 model (downloaded automatically)
- Kalshi account with API key (for production)
- Telegram bot token and chat ID

### Installation
```bash
pip install -r requirements.txt
pip install cryptography  # missing from requirements.txt, add it
```

### Configuration
```bash
cp .env.example .env
# Edit .env with your credentials
```

### Recommended first-run sequence
```bash
# 1. Verify connectors work
python scripts/test_connectors.py

# 2. Review match quality before trusting the bot
python scripts/inspect_matches.py --top 100 --save matches.csv
# Open matches.csv and check:
#   - Do matched titles describe the same event?
#   - Is outcomes_aligned correct?
#   - Adjust MIN_SIMILARITY in .env if needed

# 3. Full scan, no alerts, verify maths
python scripts/scan_once.py --min-roi 0.0

# 4. Run the bot
python main.py
```

### Docker
```bash
docker compose up -d
docker compose logs -f
```

---

## 13. Code conventions

- All modules use Python 3.12+ type hints.
- `settings` is the only allowed way to read configuration.
- No module imports from another module at the same or lower level
  except through the `models/market.py` types. The dependency graph
  is strictly layered: models ← connectors/matching/arbitrage/alerts ← main.
- Log with `logger = logging.getLogger(__name__)` in every module.
- Use `logger.debug` for per-item detail, `logger.info` for cycle-level
  events, `logger.warning` for recoverable errors, `logger.error` for
  failures that skip a cycle.
- All prices are floats in [0, 1]. Any deviation from this is a bug.
- `UnifiedMarket`, `MatchGroup`, and `ArbOpportunity` are dataclasses.
  Do not add methods that perform I/O to them — they are pure data.

---

## 14. Dependency graph

```
main.py
  ├── config.py                 (no internal deps)
  ├── models/market.py          (no internal deps)
  ├── utils/http.py             (no internal deps)
  ├── utils/kalshi_auth.py      (no internal deps)
  ├── connectors/polymarket.py  → models, utils/http
  ├── connectors/kalshi.py      → models, utils/http, utils/kalshi_auth
  ├── matching/embedder.py      → models
  ├── matching/matcher.py       → models, matching/embedder, config
  ├── arbitrage/engine.py       → models, config
  ├── storage/state.py          → models, config
  └── alerts/telegram_bot.py   → models, config
```

No circular dependencies. Every import is one-directional.
`models/market.py` and `config.py` have zero internal imports.

---

## 15. External dependencies (requirements.txt)

```
aiohttp==3.9.5               # Async HTTP for all API calls
sentence-transformers==3.0.1 # BGE-M3 wrapper
numpy==1.26.4                # Similarity matrix computation
scikit-learn==1.5.0          # (available but not currently used — reserved)
python-dotenv==1.0.1         # .env loading
FlagEmbedding==1.2.11        # BGE-M3 model support
torch==2.3.1                 # Required by sentence-transformers
transformers==4.41.2         # Required by sentence-transformers
cryptography>=42.0.0         # RSA signing for Kalshi (MISSING — add this)
```
