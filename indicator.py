"""
LinReg Entry + Rolling AVWAP Indicator
=======================================
Python port of the ThinkScript strategy:
  - Ripster Trend EMAs + EMA Clouds
  - Linear Regression Divergence
  - Rolling Anchored VWAP (EWMA-based)
  - Entry Logic: EMA Stack + AVWAP Break
  - Relative Strength (0–100, 120-bar window)
  - Bull Arrow condition

Usage:
    from indicator import compute_signals
    signals = compute_signals(df)   # df must have: open, high, low, close, volume

    signals["buy"]       → True/False per bar  (ValidBuy)
    signals["sell"]      → True/False per bar  (ValidSell)
    signals["bull_arrow"]→ True/False per bar  (BullArrow)
    signals["trend"]     → "bull" / "bear" / "neutral" per bar (candle color)
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
#  Low-level helpers
# ─────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average — mirrors ThinkScript ExpAverage()."""
    return series.ewm(span=period, adjust=False).mean()


def inertia(series: pd.Series, period: int) -> pd.Series:
    """
    ThinkScript Inertia() = linear regression value (end-point of linreg line).
    Equivalent to pandas rolling linear regression fitted value at each bar.
    """
    result = series.copy() * np.nan
    for i in range(period - 1, len(series)):
        y = series.iloc[i - period + 1 : i + 1].values
        x = np.arange(period)
        slope, intercept = np.polyfit(x, y, 1)
        result.iloc[i] = slope * (period - 1) + intercept
    return result


# ─────────────────────────────────────────────
#  Main function
# ─────────────────────────────────────────────

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all indicator values and signals for a price DataFrame.

    Args:
        df: Must contain columns — open, high, low, close, volume
            Index should be datetime or integer (oldest bar first).

    Returns:
        The same DataFrame with indicator + signal columns added.
    """
    df = df.copy()
    df["close"]  = pd.to_numeric(df["close"])
    df["high"]   = pd.to_numeric(df["high"])
    df["low"]    = pd.to_numeric(df["low"])
    df["volume"] = pd.to_numeric(df["volume"])

    # ── A. Ripster / Trend EMAs ──────────────────────────────────────────
    df["ema5"]   = ema(df["close"], 5)
    df["ema13"]  = ema(df["close"], 12)   # len12 = 12 in original
    df["ema34"]  = ema(df["close"], 34)
    df["ema50"]  = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 480)  # len200 = 480 in original

    # EMA Clouds (directional flag — True = bullish cloud colour)
    df["cloud1_bull"] = df["ema5"]  > df["ema13"]   # green cloud  (5 vs 13)
    df["cloud2_bull"] = df["ema34"] > df["ema50"]   # lime cloud   (34 vs 50)

    # ── B. Linear Regression Divergence ─────────────────────────────────
    print("  Computing linear regression (this may take a moment)...")
    df["linReg"] = inertia(df["close"], 80)
    df["emaLR"]  = ema(df["linReg"], 20)

    # ── C. Rolling Anchored VWAP (EWMA-based) ───────────────────────────
    df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3

    # Short AVWAP (120-bar EWMA of typical price * volume / volume)
    df["avwap"] = (
        ema(df["hlc3"] * df["volume"], 120) /
        ema(df["volume"], 120)
    )

    # Long AVWAP (240-bar)
    df["avwap2"] = (
        ema(df["hlc3"] * df["volume"], 240) /
        ema(df["volume"], 240)
    )

    # ── D. Entry Logic ───────────────────────────────────────────────────

    # EMA stacks
    df["emaStackBull"] = (df["ema13"] > df["ema34"]) & (df["ema34"] > df["ema50"])
    df["emaStackBear"] = (df["ema13"] < df["ema34"]) & (df["ema34"] < df["ema50"])

    # Raw buy/sell conditions
    df["buySignal"]  = (
        df["emaStackBull"] &
        (df["close"] > df["avwap"]) &
        (df["close"] > df["linReg"])
    )
    df["sellSignal"] = (
        df["emaStackBear"] &
        (df["close"] < df["avwap"]) &
        (df["close"] < df["linReg"])
    )

    # Valid only on the FIRST bar the signal fires (edge detection)
    df["buy"]  = df["buySignal"]  & ~df["buySignal"].shift(1).fillna(False)
    df["sell"] = df["sellSignal"] & ~df["sellSignal"].shift(1).fillna(False)

    # ── Trend / Candle Colour ────────────────────────────────────────────
    bull_trend = (
        (df["avwap"]  > df["avwap2"]) &
        (df["close"]  > df["ema200"]) &
        df["emaStackBull"] &
        (df["avwap2"] > df["ema200"])
    )
    bear_trend = (
        (df["avwap"]  < df["avwap2"]) &
        (df["close"]  < df["ema200"]) &
        df["emaStackBear"] &
        (df["avwap2"] < df["ema200"])
    )
    df["trend"] = np.where(bull_trend, "bull", np.where(bear_trend, "bear", "neutral"))

    # ── Relative Strength (0–100, 120-bar window) ────────────────────────
    rolling_high = df["high"].rolling(120).max()
    rolling_low  = df["low"].rolling(120).min()
    df["rs"] = ((df["close"] - rolling_low) / (rolling_high - rolling_low) * 100).round(0)

    # RS bubble condition (same logic as ThinkScript)
    df["rs_bubble"] = (
        ((df["rs"] > 69) & (df["close"] > df["ema50"]) & (df["ema34"] > df["ema50"])) |
        ((df["rs"] < 20) & (df["close"] < df["ema50"]) & (df["ema34"] < df["ema50"]))
    )

    # ── Bull Arrow ───────────────────────────────────────────────────────
    close_between_34_and_50 = (df["low"] < df["ema34"]) & (df["close"] > df["ema50"])

    df["bull_arrow"] = (
        (df["close"]  > df["ema200"]) &
        (df["ema34"]  > df["ema50"])  &
        (df["ema50"]  > df["ema200"]) &
        (df["avwap"]  > df["avwap2"]) &
        (df["close"]  > df["avwap"])  &
        (df["close"]  > df["avwap2"]) &
        (df["rs"]     > 70)           &
        close_between_34_and_50
    )

    # ── LinReg state label ───────────────────────────────────────────────
    df["linreg_state"] = np.where(df["linReg"] > df["emaLR"], "Bullish", "Bearish")

    return df


# ─────────────────────────────────────────────
#  Quick summary printer (mirrors ThinkScript labels)
# ─────────────────────────────────────────────

def print_latest(df: pd.DataFrame):
    """Print a summary of the latest bar — mirrors the chart labels."""
    row = df.iloc[-1]
    print("\n── Latest Bar Summary ─────────────────────────")
    print(f"  Close         : {row['close']:.2f}")
    print(f"  EMA200        : {row['ema200']:.2f}")
    print(f"  AVWAP (120)   : {row['avwap']:.2f}")
    print(f"  AVWAP2 (240)  : {row['avwap2']:.2f}")
    print(f"  LinReg        : {row['linReg']:.2f}  ({row['linreg_state']})")
    print(f"  RS (0-100)    : {row['rs']:.0f}")
    print(f"  Trend         : {row['trend'].upper()}")
    print(f"  EMA Stack Bull: {row['emaStackBull']}  Bear: {row['emaStackBear']}")
    print(f"  ── Signals ──")
    print(f"  BUY Signal    : {'✅ YES' if row['buy']  else '❌ No'}")
    print(f"  SELL Signal   : {'🔴 YES' if row['sell'] else '❌ No'}")
    print(f"  Bull Arrow    : {'🟢 YES' if row['bull_arrow'] else '❌ No'}")
    print(f"  RS Bubble     : {'👁  YES' if row['rs_bubble'] else '❌ No'}")
    print("────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────
#  Integration with the Webull bot
# ─────────────────────────────────────────────

def get_signal_for_bot(df: pd.DataFrame) -> str:
    """
    Convenience wrapper for use in webull_trading_bot.py.
    Replaces the old check_signal() function.

    Returns:
        'BUY'  — ValidBuy fired on the latest bar
        'SELL' — ValidSell fired on the latest bar
        'HOLD' — No actionable signal
    """
    signals = compute_signals(df)
    latest  = signals.iloc[-1]

    if latest["buy"]:
        return "BUY"
    elif latest["sell"]:
        return "SELL"
    else:
        return "HOLD"


# ─────────────────────────────────────────────
#  Standalone demo (run directly to test)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf  # pip install yfinance

    ticker = "AAPL"
    print(f"Fetching 1y of daily data for {ticker}...")
    raw = yf.download(ticker, period="2y", interval="1d", progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    print("Computing indicators...")
    result = compute_signals(raw)
    print_latest(result)

    # Show all bars where a BUY or SELL fired
    buys  = result[result["buy"]]
    sells = result[result["sell"]]
    arrows = result[result["bull_arrow"]]

    print(f"BUY  signals  : {len(buys)}   most recent → {buys.index[-1]  if len(buys)  else 'none'}")
    print(f"SELL signals  : {len(sells)}  most recent → {sells.index[-1] if len(sells) else 'none'}")
    print(f"Bull arrows   : {len(arrows)} most recent → {arrows.index[-1] if len(arrows) else 'none'}")