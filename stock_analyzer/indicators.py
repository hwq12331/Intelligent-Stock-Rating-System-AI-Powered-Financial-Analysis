"""Pure-pandas technical indicator fallbacks."""

from __future__ import annotations

import pandas as pd


def rsi_series(close: pd.Series, length: int = 14) -> pd.Series:
    """Calculate RSI with pure pandas."""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / length, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - 100 / (1 + rs)


def macd_series(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD with pure pandas."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def bbands_series(close: pd.Series, n: int = 20, stds: int = 2):
    """Calculate Bollinger Bands."""
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    upper = ma + stds * sd
    lower = ma - stds * sd
    pctb = (close - lower) / ((upper - lower) + 1e-12)
    width = (upper - lower) / (ma.abs() + 1e-12)
    return pctb, width


def atr_pct_series(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Calculate ATR as a percentage of price."""
    prev = close.shift()
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    return atr / (close.abs() + 1e-12)


def stoch_series(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
):
    """Calculate the stochastic oscillator."""
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    k_raw = 100 * (close - ll) / ((hh - ll) + 1e-12)
    k_smooth = k_raw.rolling(smooth_k).mean()
    d_line = k_smooth.rolling(d).mean()
    return k_smooth, d_line
