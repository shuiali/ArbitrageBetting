"""
alerts/telegram_bot.py

Sends arbitrage alerts to a Telegram chat via the Bot API.
Uses raw HTTP (aiohttp) — no python-telegram-bot dependency needed.

Message format (MarkdownV2):
  🔔 ARB FOUND — net +X.XX%
  ──────────────────────────
  📌 Polymarket: "Will X happen?"
  📌 Kalshi:     "Will X happen?"
  Similarity: 0.93

  ✅ BUY YES  on Polymarket @ 0.42 ($0.42)
  🔴 BUY NO   on Kalshi     @ 0.64 (= $0.36)

  Gross spread:  6.0%
  Fees:          6.0% (poly 2% + kalshi 4%)
  Net ROI:       +0.00%  ← example at threshold

  [Polymarket ↗] [Kalshi ↗]

Also supports a /status command that returns current bot stats.
"""

import logging
import time
from typing import Optional

import aiohttp

from config import settings
from models.market import ArbOpportunity, Platform

logger = logging.getLogger(__name__)

_BASE = "https://api.telegram.org"
_TIMEOUT = aiohttp.ClientTimeout(total=10)


def _esc(text: str) -> str:
    """Escape special chars for Telegram MarkdownV2."""
    special = r"\_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{c}" if c in special else c for c in str(text))


def _platform_emoji(platform: Platform) -> str:
    return "🟣" if platform == Platform.POLYMARKET else "🔵"


def _format_opportunity(opp: ArbOpportunity) -> str:
    buy_emoji = _platform_emoji(opp.buy_platform)
    sell_emoji = _platform_emoji(opp.sell_platform)

    buy_name = opp.buy_platform.value.capitalize()
    sell_name = opp.sell_platform.value.capitalize()

    # Truncate long titles
    def trunc(s: str, n: int = 80) -> str:
        return s if len(s) <= n else s[:n - 1] + "…"

    poly_title = trunc(opp.poly_market.title)
    kalshi_title = trunc(opp.kalshi_market.title)

    buy_outcome = opp.outcome_text_for(opp.buy_platform, opp.buy_side)
    hedge_outcome = opp.outcome_text_for(opp.sell_platform, opp.hedge_side)

    lines = [
        f"🔔 *ARB FOUND — net \\+{_esc(f'{opp.net_roi_pct:.2f}')}%*",
        "─" * 28,
        f"🟣 *Polymarket:* {_esc(poly_title)}",
        f"🔵 *Kalshi:*     {_esc(kalshi_title)}",
        f"_Similarity: {_esc(f'{opp.similarity_score:.3f}')}_",
        "",
        f"✅ *BUY {_esc(opp.buy_side)}* on {_esc(buy_name)} @ {_esc(str(round(opp.buy_price, 4)))}",
        f"   ↳ {_esc(buy_outcome[:120])}",
        f"🔴 *BUY {_esc(opp.hedge_side)}* on {_esc(sell_name)} @ {_esc(str(opp.hedge_price))}  \\(\\= 1 \\− {_esc(str(round(opp.sell_price, 4)))}\\)",
        f"   ↳ {_esc(hedge_outcome[:120])}",
        "",
        f"Gross spread:  {_esc(f'{opp.gross_spread_pct:.2f}')}%",
        f"Fees:          {_esc(f'{opp.total_fees * 100:.1f}')}% \\(poly {_esc(f'{settings.polymarket_fee_rate * 100:.0f}')}% \\+ kalshi {_esc(f'{settings.kalshi_fee_rate * 100:.0f}')}%\\)",
        f"*Net ROI:*      *\\+{_esc(f'{opp.net_roi_pct:.2f}')}%*",
        "",
        f"[Polymarket ↗]({_esc(opp.poly_market.url)})  \\|  [Kalshi ↗]({_esc(opp.kalshi_market.url)})",
    ]
    return "\n".join(lines)


class TelegramBot:
    def __init__(self) -> None:
        self._token = settings.telegram_bot_token
        self._chat_id = settings.telegram_chat_id
        self._base_url = f"{_BASE}/bot{self._token}"

    async def send_alert(self, opp: ArbOpportunity) -> bool:
        """Send a formatted arbitrage alert. Returns True on success."""
        text = _format_opportunity(opp)
        return await self._send_message(text)

    async def send_text(self, message: str) -> bool:
        """Send a plain text message (used for status, errors, startup notice)."""
        return await self._send_message(_esc(message), parse_mode="MarkdownV2")

    async def send_startup_message(self, poly_count: int, kalshi_count: int) -> None:
        msg = (
            f"🤖 *Arb bot started*\n"
            f"Polymarket: {_esc(str(poly_count))} markets\n"
            f"Kalshi: {_esc(str(kalshi_count))} markets\n"
            f"Min ROI threshold: {_esc(f'{settings.min_net_roi * 100:.1f}')}%\n"
            f"Similarity threshold: {_esc(str(settings.min_similarity))}"
        )
        await self._send_message(msg)

    async def send_cycle_summary(
        self, groups: int, opps: int, cycle: int
    ) -> None:
        """Optional periodic heartbeat — sent every N cycles."""
        ts = time.strftime("%H:%M:%S")
        msg = (
            f"💓 *Heartbeat* \\[{_esc(ts)}\\]\n"
            f"Cycle: {_esc(str(cycle))} \\| "
            f"Groups: {_esc(str(groups))} \\| "
            f"Opps found: {_esc(str(opps))}"
        )
        await self._send_message(msg)

    async def _send_message(
        self,
        text: str,
        parse_mode: str = "MarkdownV2",
        disable_preview: bool = True,
    ) -> bool:
        url = f"{self._base_url}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_preview,
        }
        try:
            async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
                async with session.post(url, json=payload) as resp:
                    data = await resp.json()
                    if not data.get("ok"):
                        logger.warning(
                            "Telegram send failed: %s", data.get("description")
                        )
                        return False
                    return True
        except Exception as exc:
            logger.warning("Telegram error: %s", exc)
            return False
