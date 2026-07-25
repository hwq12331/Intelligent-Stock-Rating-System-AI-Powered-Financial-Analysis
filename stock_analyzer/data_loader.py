"""Market data loading helpers."""

from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

from .config import ENABLE_DEBUG, TIMEFRAMES


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize OHLCV data from yfinance."""
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(col) for col in group if col]) for group in df.columns]

    rename_map: dict[object, str] = {}
    for column in df.columns:
        lowered = str(column).lower()
        if "open" in lowered and "adj" not in lowered:
            rename_map[column] = "Open"
        elif "high" in lowered:
            rename_map[column] = "High"
        elif "low" in lowered:
            rename_map[column] = "Low"
        elif "close" in lowered:
            rename_map[column] = "Close"
        elif "volume" in lowered:
            rename_map[column] = "Volume"
    df = df.rename(columns=rename_map)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not set(needed).issubset(df.columns):
        return pd.DataFrame()

    for column in needed:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=["Close"])


def download_multi_timeframe_data(ticker: str) -> dict[str, pd.DataFrame]:
    """Download data for all configured timeframes with retry handling."""
    results: dict[str, pd.DataFrame] = {}

    for timeframe_name, timeframe_config in TIMEFRAMES.items():
        interval = timeframe_config["interval"]
        period = timeframe_config["period"]
        min_bars = timeframe_config["min_bars"]

        for attempt in range(1, 3):
            try:
                df = yf.download(
                    ticker,
                    interval=interval,
                    period=period,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    group_by="column",
                )
                df = clean_ohlcv(df)

                if not df.empty and len(df) >= min_bars:
                    results[timeframe_name] = df
                    if ENABLE_DEBUG:
                        print(f"    ✓ {timeframe_name:10s}: {len(df):4d} bars ({interval})")
                    break
            except Exception as exc:
                if ENABLE_DEBUG and attempt == 2:
                    print(f"    ✗ {timeframe_name:10s}: {str(exc)[:50]}")

            time.sleep(0.3 * attempt)

    return results
