"""Project configuration and optional dependency discovery."""

from __future__ import annotations

import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "2.1.0-COMPLETE"
CURRENT_USER = "hwq12331"
BUILD_TIMESTAMP = "2025-10-24 06:27:27"
RUN_TIMESTAMP = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

TIMEFRAMES = {
    "yearly": {
        "interval": "1mo",
        "period": "5y",
        "horizon_days": 252,
        "min_bars": 20,
        "label": "Yearly",
    },
    "quarterly": {
        "interval": "1wk",
        "period": "3y",
        "horizon_days": 63,
        "min_bars": 60,
        "label": "Quarterly",
    },
    "weekly": {
        "interval": "1d",
        "period": "2y",
        "horizon_days": 21,
        "min_bars": 100,
        "label": "Weekly",
    },
    "daily": {
        "interval": "1d",
        "period": "1y",
        "horizon_days": 10,
        "min_bars": 80,
        "label": "Daily",
    },
    "intraday": {
        "interval": "1h",
        "period": "60d",
        "horizon_hours": 16,
        "min_bars": 100,
        "label": "4-Hour",
    },
}

MIN_TRAINING_PERIODS = 25
MIN_WARMUP_PERIODS = 15
WF_N_SPLITS = 3
WF_TEST_SIZE = 8

MIN_WF_CORRELATION = 0.01
MAX_UNCERTAINTY = 3.0
MIN_WF_PREDICTIONS = 10
MIN_TIMEFRAMES_REQUIRED = 1
TRANSACTION_COST = 0.002

BATCH_SIZE = 100
ENABLE_CHECKPOINTS = True
CHECKPOINT_INTERVAL = 20
ENABLE_DEBUG = True
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
TICKER_FILES = ("tickers_sr.txt", "symbols.txt")

try:
    import pandas_ta as ta  # noqa: F401

    HAS_TA = True
    print("✓ pandas_ta available")
except ImportError:
    HAS_TA = False
    print("⚠ pandas_ta not available - using fallbacks")

try:
    import xgboost as xgb  # type: ignore

    HAS_XGB = True
    print("✓ XGBoost available")
except ImportError:
    xgb = None
    HAS_XGB = False
    print("⚠ XGBoost not available")

try:
    import lightgbm as lgb  # type: ignore

    HAS_LGB = True
    print("✓ LightGBM available")
except ImportError:
    lgb = None
    HAS_LGB = False
    print("⚠ LightGBM not available")
