"""Technical feature engineering and ML training."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import StandardScaler

from .config import (
    HAS_LGB,
    HAS_XGB,
    MIN_TRAINING_PERIODS,
    MIN_WARMUP_PERIODS,
    MIN_WF_PREDICTIONS,
    TIMEFRAMES,
    WF_N_SPLITS,
    WF_TEST_SIZE,
    lgb,
    xgb,
)
from .indicators import atr_pct_series, bbands_series, macd_series, rsi_series, stoch_series


class WalkForwardValidation(BaseCrossValidator):
    """Walk-forward cross-validation for time series."""

    def __init__(self, n_splits: int = 3, test_size: int = 10, gap: int = 0):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        idx = np.arange(n_samples)
        min_train = max(MIN_TRAINING_PERIODS, n_samples // 4)
        test_starts = np.linspace(min_train, n_samples - self.test_size, self.n_splits, dtype=int)

        for t0 in test_starts:
            train_end = t0 - self.gap
            t1 = min(t0 + self.test_size, n_samples)
            if train_end < MIN_TRAINING_PERIODS:
                continue
            yield (idx[:train_end], idx[t0:t1])

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def build_technical_features(ohlcv: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Create technical features adapted to the requested timeframe."""
    tf_config = TIMEFRAMES.get(timeframe, TIMEFRAMES["daily"])

    close = ohlcv["Close"].astype(float)
    high = ohlcv["High"].astype(float)
    low = ohlcv["Low"].astype(float)
    volume = ohlcv["Volume"].astype(float)

    df = pd.DataFrame(index=close.index)
    df["price"] = close
    df["returns"] = close.pct_change()

    if timeframe == "yearly":
        lags = [1, 2, 3, 6, 12]
        ma_periods = [6, 12, 24]
        vol_periods = [6, 12]
    elif timeframe == "quarterly":
        lags = [1, 2, 4, 8, 13]
        ma_periods = [8, 13, 26]
        vol_periods = [8, 13]
    elif timeframe == "weekly":
        lags = [1, 2, 4, 8, 13]
        ma_periods = [10, 20, 50]
        vol_periods = [10, 20]
    elif timeframe == "intraday":
        lags = [1, 2, 4, 8, 16]
        ma_periods = [8, 16, 32]
        vol_periods = [8, 16]
    else:
        lags = [1, 2, 3, 5, 10, 20]
        ma_periods = [20, 50, 100]
        vol_periods = [10, 20]

    for lag in lags:
        df[f"return_lag_{lag}"] = df["returns"].shift(lag)

    for period in [lags[2], lags[3], lags[4]]:
        df[f"momentum_{period}"] = (close / close.shift(period) - 1).shift(1)

    for period in vol_periods:
        df[f"volatility_{period}"] = df["returns"].rolling(period).std().shift(1)

    ma_short = close.rolling(ma_periods[0]).mean()
    ma_mid = close.rolling(ma_periods[1]).mean()
    ma_long = close.rolling(ma_periods[2]).mean() if len(close) >= ma_periods[2] else ma_mid

    df["price_vs_ma_short"] = (close / (ma_short + 1e-12) - 1).shift(1)
    df["price_vs_ma_mid"] = (close / (ma_mid + 1e-12) - 1).shift(1)
    df["price_vs_ma_long"] = (close / (ma_long + 1e-12) - 1).shift(1)
    df["ma_cross"] = ((ma_short - ma_mid) / (close + 1e-12)).shift(1)

    df["rsi"] = rsi_series(close).shift(1)
    macd, macd_sig, macd_hist = macd_series(close)
    df["macd"] = macd.shift(1)
    df["macd_signal"] = macd_sig.shift(1)
    df["macd_hist"] = macd_hist.shift(1)

    bb_pctb, bb_width = bbands_series(close, n=ma_periods[0])
    df["bb_pctb"] = bb_pctb.shift(1)
    df["bb_width"] = bb_width.shift(1)
    df["atr_pct"] = atr_pct_series(high, low, close).shift(1)

    stoch_k, stoch_d = stoch_series(high, low, close)
    df["stoch_k"] = stoch_k.shift(1)
    df["stoch_d"] = stoch_d.shift(1)

    if (volume > 0).mean() > 0.3:
        vol_ma = volume.rolling(ma_periods[0]).mean()
        df["volume_ratio"] = (volume / (vol_ma + 1e-12)).shift(1)
    else:
        df["volume_ratio"] = 1.0

    if timeframe == "intraday":
        horizon = tf_config["horizon_hours"]
        df["target"] = close.pct_change(horizon).shift(-horizon)
    else:
        horizon = tf_config["horizon_days"]
        df["target"] = close.pct_change(horizon).shift(-horizon)

    warmup = min(MIN_WARMUP_PERIODS, len(df) // 4)
    return df.iloc[warmup:].copy()


def train_ml_model_timeframe(ticker: str, data: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    """Train the multi-model ensemble for one timeframe."""
    try:
        features_df = build_technical_features(data, timeframe)
        tf_config = TIMEFRAMES[timeframe]

        if timeframe in ["yearly", "quarterly"]:
            min_required = max(15, tf_config["min_bars"] // 4)
        else:
            min_required = max(20, tf_config["min_bars"] // 3)

        if len(features_df) < min_required:
            return {"success": False, "reason": f"insufficient_{timeframe}_data"}

        train_df = features_df[features_df["target"].notna()].copy()
        live_df = features_df.iloc[[-1]].copy()
        if len(train_df) < MIN_TRAINING_PERIODS:
            return {"success": False, "reason": "insufficient_training_data"}

        feature_cols = [column for column in train_df.columns if column not in ["target", "price", "returns"]]
        X = train_df[feature_cols].values
        y = train_df["target"].values
        X_live = live_df[feature_cols].values

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]
        if len(X) < MIN_TRAINING_PERIODS:
            return {"success": False, "reason": f"only_{len(X)}_clean_rows"}

        models: list[Any] = [
            Ridge(alpha=1.0, random_state=42),
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=3,
                subsample=0.8,
                random_state=42,
            ),
        ]

        if HAS_XGB and xgb is not None:
            try:
                models.append(
                    xgb.XGBRegressor(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.8,
                        random_state=42,
                        n_jobs=1,
                        verbosity=0,
                    )
                )
            except Exception:
                pass

        if HAS_LGB and lgb is not None:
            try:
                models.append(
                    lgb.LGBMRegressor(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=4,
                        random_state=42,
                        verbose=-1,
                    )
                )
            except Exception:
                pass

        wfv = WalkForwardValidation(WF_N_SPLITS, WF_TEST_SIZE, 0)
        wf_preds: list[float] = []
        wf_acts: list[float] = []

        for tr_idx, te_idx in wfv.split(X):
            if len(tr_idx) < MIN_TRAINING_PERIODS:
                continue

            Xtr, Xte = X[tr_idx], X[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]

            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xte_s = scaler.transform(Xte)

            fold_preds = []
            for model in models:
                model_instance = type(model)(**model.get_params())
                model_instance.fit(Xtr_s, ytr)
                fold_preds.append(model_instance.predict(Xte_s))

            wf_preds.extend(np.mean(fold_preds, axis=0).tolist())
            wf_acts.extend(yte.tolist())

        if len(wf_preds) < MIN_WF_PREDICTIONS:
            return {"success": False, "reason": f"only_{len(wf_preds)}_wf_samples"}

        wf_preds_array = np.array(wf_preds)
        wf_acts_array = np.array(wf_acts)
        wf_mae = float(mean_absolute_error(wf_acts_array, wf_preds_array))
        wf_corr = float(np.corrcoef(wf_acts_array, wf_preds_array)[0, 1])
        wf_dir = float(np.mean(np.sign(wf_preds_array) == np.sign(wf_acts_array)))

        wf_returns = wf_preds_array * wf_acts_array
        wf_sharpe = wf_returns.mean() / (wf_returns.std() + 1e-6)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        X_live_s = scaler.transform(X_live)

        final_preds = []
        for model in models:
            model_instance = type(model)(**model.get_params())
            model_instance.fit(Xs, y)
            final_preds.append(float(model_instance.predict(X_live_s)[0]))

        ensemble = float(np.mean(final_preds))
        prediction_std = float(np.std(final_preds))
        prediction_uncertainty = float(prediction_std / (abs(ensemble) + 1e-6))
        volatility = float(train_df["returns"].std())

        return {
            "success": True,
            "timeframe": timeframe,
            "prediction": ensemble,
            "prediction_std": prediction_std,
            "prediction_uncertainty": prediction_uncertainty,
            "volatility": volatility,
            "vol_adjusted_pred": ensemble / (volatility + 1e-6),
            "wf_mae": wf_mae,
            "wf_correlation": wf_corr,
            "wf_direction_accuracy": wf_dir,
            "wf_sharpe": float(wf_sharpe),
            "wf_n_predictions": len(wf_preds),
            "training_samples": len(X),
            "num_features": len(feature_cols),
            "models_used": len(models),
        }
    except Exception as exc:
        return {"success": False, "timeframe": timeframe, "reason": str(exc)[:100]}
