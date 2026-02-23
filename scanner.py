"""
Daily Stock Scanner + Discord Alerts
======================================
Reads tickers from stocks.txt, runs the LinReg + AVWAP indicator on each,
and sends a Discord alert whenever a BUY signal fires within the last 5 candles.

Runs once per day at 3:30 PM ET (market close) via GitHub Actions,
or on a local scheduler if running manually.

Requirements:
    pip install yfinance pandas numpy requests schedule

Setup:
    1. Create a Discord Webhook (instructions below in CONFIG)
    2. Add/remove tickers in stocks.txt
    3. Run:  python scanner.py
"""

import os
import time
import logging
import schedule
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

# ── Import your indicator ────────────────────────────────────────────────────
from indicator import compute_signals

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

# Discord Webhook URL — read from environment variable (GitHub Secret)
# For local use, set it in your terminal:
#   Windows: set DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
#   Mac/Linux: export DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
# Or replace the fallback string below with your URL for local testing only.
DISCORD_WEBHOOK_URL = os.environ.get(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1463019737297453242/2iKe5IqTAoy0FeQeXOxrXG1YeQQqRRhlOusNz8Py0w_3wjk7nkfoxsknjDdNpUk-sbx5"
)

# Path to your watchlist file (one ticker per line)
STOCKS_FILE = "stocks.txt"

# How many days of daily data to fetch (needs enough for EMA-480 to warm up)
HISTORY_DAYS = "2y"

# Scan time — 9:35 AM gives market 5 min to open and settle
SCAN_TIME = "15:30"  # 3:30 PM ET (matches GitHub Actions cron: 20:30 UTC)

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        # Force UTF-8 on Windows console so emojis don't crash logging
        logging.StreamHandler(
            open(os.devnull, 'w', encoding='utf-8')
            if False else
            __import__('sys').stdout
        ),
        logging.FileHandler("scanner.log", encoding="utf-8"),
    ],
)
# Override stream handler to force UTF-8 encoding on Windows
import sys
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Load watchlist from file
# ─────────────────────────────────────────────

def load_watchlist(filepath: str) -> list[str]:
    """
    Read tickers from a text file.
    - One ticker per line
    - Lines starting with # are treated as comments and skipped
    - Blank lines are skipped
    """
    if not os.path.exists(filepath):
        log.error(f"Watchlist file not found: {filepath}")
        return []

    tickers = []
    with open(filepath, "r") as f:
        for line in f:
            ticker = line.strip()
            if ticker and not ticker.startswith("#"):
                tickers.append(ticker.upper())

    log.info(f"📋 Loaded {len(tickers)} tickers from {filepath}: {tickers}")
    return tickers


# ─────────────────────────────────────────────
#  Fetch daily OHLCV data
# ─────────────────────────────────────────────

def fetch_data(ticker: str) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance."""
    try:
        df = yf.download(ticker, period=HISTORY_DAYS, interval="1d", progress=False)
        if df.empty:
            log.warning(f"No data for {ticker}")
            return pd.DataFrame()
        # Fix yfinance MultiIndex columns (e.g. ('Close', 'AAPL') -> 'close')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        log.error(f"Failed to fetch {ticker}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
#  Discord alert sender
# ─────────────────────────────────────────────

def send_discord_alert(embeds: list[dict]):
    """
    Post one or more Discord embed cards to the webhook.
    Batches up to 10 embeds per request (Discord limit).
    """
    if not embeds:
        return

    # Discord allows max 10 embeds per message — chunk if needed
    for i in range(0, len(embeds), 10):
        chunk = embeds[i : i + 10]
        payload = {"embeds": chunk}
        try:
            resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
            if resp.status_code not in (200, 204):
                log.error(f"Discord error {resp.status_code}: {resp.text}")
            else:
                log.info(f"📨 Sent {len(chunk)} Discord alert(s)")
        except Exception as e:
            log.error(f"Discord send failed: {e}")
        time.sleep(0.5)  # small pause between batches


def build_embed(ticker: str, signal: str, row: pd.Series,
                signal_date=None, bars_ago: int = 0) -> dict:
    """
    Build a Discord embed card for a signal.

    Args:
        ticker:      Stock symbol
        signal:      'BUY', 'SELL', or 'BULL_ARROW'
        row:         Indicator row from the bar where the signal fired
        signal_date: The date/index of the bar that fired the signal
        bars_ago:    How many bars ago the signal fired (0 = latest bar)
    """

    # Colours (Discord uses decimal integers for embed colours)
    COLORS = {
        "BUY":        0x00FF7F,   # Spring green
        "SELL":       0xFF3333,   # Red
        "BULL_ARROW": 0x00CC44,   # Dark green
    }

    # Emoji + titles
    TITLES = {
        "BUY":        "🟢 BUY Signal",
        "SELL":       "🔴 SELL Signal",
        "BULL_ARROW": "🏹 Bull Arrow",
    }

    trend_emoji = {"bull": "🐂", "bear": "🐻", "neutral": "⚪"}
    trend       = row.get("trend", "neutral")

    # Recency label
    if bars_ago == 0:
        recency = "🔴 LIVE — Today's candle"
    elif bars_ago == 1:
        recency = "🟠 1 candle ago"
    else:
        recency = f"🟡 {bars_ago} candles ago"

    # Format signal date
    date_str = str(signal_date)[:10] if signal_date is not None else "N/A"

    # Safe value formatting
    def fmt(val, decimals=2):
        try:
            return f"{float(val):.{decimals}f}"
        except Exception:
            return "N/A"

    embed = {
        "title": f"{TITLES[signal]}  —  ${ticker}",
        "color": COLORS[signal],
       
    }
    return embed


# ─────────────────────────────────────────────
#  Core scan logic
# ─────────────────────────────────────────────

def scan():
    """
    Main scan — loads watchlist, computes indicators for each ticker,
    and sends Discord alerts if any BUY / SELL / Bull Arrow signal
    fired within the last 5 candles.
    """
    log.info("=" * 55)
    log.info(f"🔍 Starting daily scan  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")

    tickers = load_watchlist(STOCKS_FILE)
    if not tickers:
        log.warning("No tickers to scan. Check stocks.txt")
        return

    alerts = []      # Collect all embeds, send at the end in one batch

    for ticker in tickers:
        try:
            log.info(f"   Scanning {ticker}...")
            df = fetch_data(ticker)
            if df.empty or len(df) < 50:
                log.warning(f"   {ticker}: Not enough data, skipping")
                continue

            df = compute_signals(df)

            # ── Exact same logic as chart.py BUY bubble ───────────────────
            # chart.py does: buys = df[df["buy"] == True]
            # "buy" = buySignal AND NOT buySignal[1]  (first bar of new signal)
            #
            # We look at the last 5 bars of the FULL dataset (same as chart
            # showing last N bars — the bubble position never changes based
            # on the display window).
            # A bubble appears on bar X only when:
            #   - buySignal[X]     is True
            #   - buySignal[X-1]   is False  (it just turned on)
            # We replicate this exactly by checking df["buy"] which is
            # pre-computed with this exact edge logic in indicator.py.
            # ─────────────────────────────────────────────────────────────

            lookback = 1
            # Slice last 5 bars — includes the bar BEFORE so edge is valid
            recent   = df.iloc[-lookback:]
            fired    = False

            # Find all bars in the last 5 where buy bubble appears
            buy_bars = recent[recent["buy"] == True]

            for idx, row in buy_bars.iterrows():
                # bars_ago: 0 = today, 1 = yesterday, etc.
                bars_ago = len(df) - 1 - df.index.get_loc(idx)
                log.info(
                    f"   BUY bubble: {ticker}  "
                    f"{'(TODAY)' if bars_ago == 0 else f'{bars_ago} candle(s) ago'}  "
                    f"— {str(idx)[:10]}  close={row['close']:.2f}  "
                    f"RS={row.get('rs', 0):.0f}"
                )
                alerts.append(build_embed(ticker, "BUY", row,
                                          signal_date=idx, bars_ago=bars_ago))
                fired = True

            if not fired:
                log.info(
                    f"   {ticker}: No BUY bubble in last {lookback} candles  "
                    f"(trend={df.iloc[-1].get('trend','?')}, "
                    f"RS={df.iloc[-1].get('rs', 0):.0f})"
                )

        except Exception as e:
            log.error(f"   Error scanning {ticker}: {e}")

        time.sleep(0.3)  # Be polite to Yahoo Finance rate limits

    # ── Send all alerts ──────────────────────────────────────────────────
    if alerts:
        log.info(f"\n📨 Sending {len(alerts)} alert(s) to Discord...")
        send_discord_alert(alerts)
    else:
        # Send a quiet "no signals" summary so you know the scan ran
        summary = {
            "embeds": [{
                "title": "📋 Daily Scan Complete — No BUY Signals",
                "color": 0x888888,
                "description": (
                    f"Scanned **{len(tickers)} tickers** on the daily timeframe.\n"
                    f"No BUY signals found in the **last 5 candles** for any ticker."
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "footer": {"text": "LinReg + AVWAP Scanner  •  Last 5 Candles"},
            }]
        }
        try:
            requests.post(DISCORD_WEBHOOK_URL, json=summary, timeout=10)
        except Exception:
            pass
        log.info("   No signals today.")

    log.info("✅ Scan complete.\n")


# ─────────────────────────────────────────────
#  Entry point + scheduler
# ─────────────────────────────────────────────

if __name__ == "__main__":
    log.info("🚀 Daily Stock Scanner started")
    log.info(f"   Watchlist file : {STOCKS_FILE}")
    log.info(f"   Discord webhook: {'✅ Set' if 'YOUR_WEBHOOK' not in DISCORD_WEBHOOK_URL else '⚠️  NOT SET'}")

    # Detect GitHub Actions environment — run once and exit (no infinite loop needed
    # because GitHub Actions cron handles the scheduling)
    running_in_ci = os.environ.get("GITHUB_ACTIONS") == "true"

    if running_in_ci:
        log.info("   Mode: GitHub Actions (run once and exit)")
        scan()
    else:
        # Local mode — run immediately then schedule daily
        log.info(f"   Mode: Local scheduler — running at {SCAN_TIME} ET every day")
        scan()
        schedule.every().day.at(SCAN_TIME).do(scan)
        while True:
            schedule.run_pending()
            time.sleep(60)