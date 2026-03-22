# Prediction Market Arbitrage Bot

Finds arbitrage opportunities between **Polymarket** and **Kalshi** using
BGE-M3 semantic matching and fee-adjusted spread calculation.
Alerts via Telegram.

---

## Project structure

```
arb_bot/
├── main.py                  # Main loop entry point
├── config.py                # All settings (loaded from .env)
├── connectors/
│   ├── polymarket.py        # Polymarket CLOB REST connector
│   └── kalshi.py            # Kalshi v2 REST connector (RSA auth)
├── models/
│   └── market.py            # UnifiedMarket, MatchGroup, ArbOpportunity
├── matching/
│   ├── embedder.py          # BGE-M3 wrapper with title cache
│   └── matcher.py           # Cosine similarity cross-platform matching
├── arbitrage/
│   └── engine.py            # Spread calc + fee-adjusted ROI
├── alerts/
│   └── telegram_bot.py      # Telegram Bot API integration
├── storage/
│   └── state.py             # In-memory state + alert cooldown tracking
├── utils/
│   ├── http.py              # Shared aiohttp with exponential backoff retry
│   └── kalshi_auth.py       # RSA-PS256 request signing for Kalshi
└── scripts/
    ├── test_connectors.py   # Validate API access before running bot
    ├── inspect_matches.py   # Review semantic match quality (outputs CSV)
    └── scan_once.py         # One-shot scan — no Telegram, print to terminal
```

---

## Setup

### 1. Clone and install

```bash
pip install -r requirements.txt
```

> BGE-M3 is ~2GB. It downloads automatically on first run and caches locally.

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | From @BotFather on Telegram |
| `TELEGRAM_CHAT_ID` | Your chat or group ID. Get it via @userinfobot |
| `KALSHI_API_KEY_ID` | UUID from kalshi.com → Account → API Keys |
| `KALSHI_PRIVATE_KEY_PATH` | Path to your downloaded PEM private key |
| `KALSHI_ENV` | `production` or `demo` |
| `MIN_SIMILARITY` | BGE-M3 cosine threshold (default `0.82`) |
| `POLL_INTERVAL_SECONDS` | How often to re-fetch prices (default `60`) |
| `REMATCH_INTERVAL_SECONDS` | How often to re-embed + re-match (default `1800`) |
| `MARKET_CACHE_REJECT_EMPTY` | Reject and never reuse 0-market cache snapshots (default `true`) |
| `MARKET_CACHE_TTL_SECONDS` | Fresh cache TTL (default `3600`) |
| `MARKET_CACHE_STALE_TTL_SECONDS` | Max stale cache age for fallback on API errors (default `21600`) |
| `POLYMARKET_EVENTS_RATE_LIMIT_PER_10S` | Gamma `/events` request budget per 10 seconds (default `290`) |
| `POLYMARKET_EVENTS_PARALLEL_PAGES` | Parallel pagination workers for Polymarket events (default `24`) |
| `EMBEDDING_CACHE_ENABLED` | Persist title embeddings to disk across runs (default `true`) |
| `EMBEDDING_BATCH_VRAM_FACTOR` | GPU batch scaling factor for embeddings (default `24`) |

### 3. Kalshi API key setup

1. Go to [kalshi.com](https://kalshi.com) → Account → API Keys → **Create API Key**
2. Download the private key PEM file — save it somewhere safe, e.g. `~/.kalshi_key.pem`
3. Copy the **Key ID** UUID shown in the UI
4. Set in `.env`:
   ```
   KALSHI_API_KEY_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
   KALSHI_PRIVATE_KEY_PATH=/home/you/.kalshi_key.pem
   ```

> For demo/testing: set `KALSHI_ENV=demo` — no credentials needed.

---

## Usage

### Validate connectors first

```bash
python scripts/test_connectors.py
```

Should show sample markets from both platforms.

### Check match quality before going live

```bash
python scripts/inspect_matches.py --top 50 --save matches.csv
```

Open `matches.csv` and review:
- Do the matched titles actually describe the same event?
- Is the outcome alignment correct? (`YES↔YES` or `YES↔NO`)
- Adjust `MIN_SIMILARITY` in `.env` if you see false positives (raise it) or missed pairs (lower it).

### One-shot scan (no Telegram)

```bash
python scripts/scan_once.py
python scripts/scan_once.py --min-roi 0.0    # show all spreads
```

The scan now prints cache status details when a connector returns zero markets,
so you can quickly tell whether it is a cache issue, proxy issue, or API/auth issue.

### Run the bot

```bash
python main.py
```

### Docker

```bash
docker compose up -d
docker compose logs -f
```

---

## How the maths work

For a binary market where YES pays $1:

```
gross_spread = |poly_yes_price - kalshi_yes_price|
total_fees   = polymarket_fee_rate + kalshi_fee_rate   (default 2% + 4% = 6%)
net_roi      = gross_spread - total_fees
```

You need a **gross spread > 6%** to profit. Example:

| Platform | YES price | Action | Cost |
|---|---|---|---|
| Polymarket | 0.40 | Buy YES | $0.40 |
| Kalshi | 0.52 | Buy NO (= 1 - 0.52) | $0.48 |
| **Total** | | | **$0.88** |
| **Payout** | | One leg always wins | **$1.00** |
| **Gross profit** | | | **$0.12 = 12%** |
| **Fees** | | 2% + 4% | **$0.06** |
| **Net ROI** | | | **6%** |

---

## Tuning

**Too many false positives** (wrong event pairs matched):
- Raise `MIN_SIMILARITY` to `0.88` or higher

**Too few matches** (clearly same events not found):
- Lower `MIN_SIMILARITY` to `0.78`
- Check that both platforms list the event as "open"

**No alerts firing despite visible spreads**:
- Check `MIN_NET_ROI` — default 2% requires 8% gross spread
- Run `scan_once.py --min-roi 0.0` to see all spreads
- Verify fee rates in `.env` are correct for your account tier

**Alert spam on the same opportunity**:
- Raise `ALERT_COOLDOWN_SECONDS` (default 300 = 5 min)
