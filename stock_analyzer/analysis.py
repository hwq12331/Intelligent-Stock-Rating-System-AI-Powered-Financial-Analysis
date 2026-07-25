"""High-level stock analysis orchestration."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import ENABLE_DEBUG, MAX_UNCERTAINTY, MIN_TIMEFRAMES_REQUIRED, MIN_WF_CORRELATION, RUN_TIMESTAMP, TIMEFRAMES, TRANSACTION_COST
from .data_loader import download_multi_timeframe_data
from .fundamentals import FundamentalAnalyzer
from .ml import train_ml_model_timeframe
from .utils import safe_compare, safe_get


def analyze_stock_complete(ticker: str) -> dict[str, Any]:
    """Run the full multi-timeframe stock analysis workflow."""
    if ENABLE_DEBUG:
        print(f"\n{'=' * 80}")
        print(f"ANALYZING: {ticker}")
        print(f"{'=' * 80}")

    results: dict[str, Any] = {
        "ticker": ticker,
        "analysis_timestamp": RUN_TIMESTAMP,
        "success": False,
    }

    if ENABLE_DEBUG:
        print("  Downloading multi-timeframe data...")
    mtf_data = download_multi_timeframe_data(ticker)
    if not mtf_data:
        results["reason"] = "no_data_downloaded"
        return results

    if ENABLE_DEBUG:
        print("  Running fundamental analysis...")
    try:
        fundamental = FundamentalAnalyzer(ticker).get_complete_fundamental_analysis()
    except Exception as exc:
        if ENABLE_DEBUG:
            print(f"    ⚠ Fundamental analysis failed: {exc}")
        fundamental = {"success": False}

    if ENABLE_DEBUG:
        print("  Running multi-timeframe ML analysis...")
    timeframe_results: dict[str, dict[str, Any]] = {}

    for timeframe_name, timeframe_data in mtf_data.items():
        if ENABLE_DEBUG:
            print(f"    → {TIMEFRAMES[timeframe_name]['label']:10s}...", end=" ")

        ml_result = train_ml_model_timeframe(ticker, timeframe_data, timeframe_name)
        if ml_result["success"]:
            if safe_compare(ml_result["wf_correlation"], ">", MIN_WF_CORRELATION):
                if ENABLE_DEBUG:
                    print(
                        f"✓ (corr={ml_result['wf_correlation']:.3f}, pred={ml_result['prediction']:+.4f})"
                    )
                timeframe_results[timeframe_name] = ml_result
            elif ENABLE_DEBUG:
                print(f"⚠ (corr={ml_result['wf_correlation']:.3f} - skipped)")
        elif ENABLE_DEBUG:
            print(f"✗ ({ml_result.get('reason', 'unknown')})")

    if len(timeframe_results) < MIN_TIMEFRAMES_REQUIRED:
        results["reason"] = f"only_{len(timeframe_results)}_quality_timeframes"
        return results

    if ENABLE_DEBUG:
        print("  Aggregating scores...")

    tf_scores = []
    for tf_result in timeframe_results.values():
        weight = max(safe_get(tf_result["wf_correlation"], 0), 0)
        vol_adj = safe_get(tf_result["vol_adjusted_pred"], 0)
        sigmoid_score = 1.0 / (1.0 + np.exp(-np.clip(vol_adj, -10, 10)))
        tf_scores.append(sigmoid_score * 100 * weight)

    technical_score = float(np.clip(np.mean(tf_scores) if tf_scores else 50, 0, 100))
    avg_corr = np.mean([safe_get(result["wf_correlation"], 0) for result in timeframe_results.values()])
    avg_sharpe = np.mean([safe_get(result["wf_sharpe"], 0) for result in timeframe_results.values()])
    avg_corr_clipped = float(np.clip(avg_corr, -1, 1))
    avg_sharpe_clipped = float(np.clip(avg_sharpe, -1, 1))
    ml_quality_score = float(
        np.clip((avg_corr_clipped * 0.5 + avg_sharpe_clipped * 0.5 + 0.5) * 100, 0, 100)
    )
    fundamental_score = float(
        np.clip(safe_get(fundamental.get("fundamental_score"), 50) if fundamental.get("success") else 50, 0, 100)
    )
    composite_score = float(
        np.clip(technical_score * 0.15 + ml_quality_score * 0.35 + fundamental_score * 0.50, 0, 100)
    )

    daily_pred = safe_get(timeframe_results.get("daily", {}).get("prediction"), 0)
    daily_corr = safe_get(timeframe_results.get("daily", {}).get("wf_correlation"), 0)

    if safe_compare(timeframe_results.get("yearly", {}).get("prediction"), ">", 0.05):
        horizon = "LONG_TERM"
    elif safe_compare(timeframe_results.get("quarterly", {}).get("prediction"), ">", 0.03):
        horizon = "MEDIUM_TERM"
    elif "daily" in timeframe_results:
        horizon = "SHORT_TERM"
    else:
        horizon = "UNKNOWN"

    if (
        safe_compare(composite_score, ">", 75)
        and safe_compare(daily_pred, ">", TRANSACTION_COST * 3)
        and safe_compare(daily_corr, ">", 0.20)
        and safe_compare(fundamental_score, ">", 65)
    ):
        recommendation = "STRONG_BUY"
        confidence = "HIGH"
    elif (
        safe_compare(composite_score, ">", 60)
        and safe_compare(daily_pred, ">", TRANSACTION_COST * 2)
        and safe_compare(fundamental_score, ">", 55)
    ):
        recommendation = "BUY"
        confidence = "MEDIUM"
    elif safe_compare(composite_score, ">", 45):
        recommendation = "HOLD"
        confidence = "MEDIUM"
    elif safe_compare(composite_score, "<", 35) or safe_compare(
        fundamental.get("financial_health_score", 100), "<", 30
    ):
        recommendation = "SELL"
        confidence = "MEDIUM"
    else:
        recommendation = "AVOID"
        confidence = "LOW"

    warning_flags: list[str] = []
    if fundamental.get("success"):
        if safe_compare(fundamental.get("debt_to_equity"), ">", 150):
            warning_flags.append("HIGH_DEBT")
        if safe_compare(fundamental.get("current_ratio", 2), "<", 1):
            warning_flags.append("LIQUIDITY_RISK")
        if safe_compare(fundamental.get("revenue_growth_annual"), "<", -0.10):
            warning_flags.append("DECLINING_REVENUE")

    avg_uncertainty = np.mean([safe_get(result["prediction_uncertainty"], 0) for result in timeframe_results.values()])
    if safe_compare(avg_uncertainty, ">", MAX_UNCERTAINTY):
        warning_flags.append("HIGH_UNCERTAINTY")

    predictions = [safe_get(result["prediction"], 0) for result in timeframe_results.values()]
    if len(predictions) >= 2:
        pred_std = np.std(predictions)
        pred_mean = np.mean(np.abs(predictions))
        if safe_compare(pred_std, ">", pred_mean):
            warning_flags.append("TIMEFRAME_DIVERGENCE")

    win_prob = (avg_corr + 1) / 2
    kelly_fraction = (win_prob * abs(daily_pred) - (1 - win_prob)) / (abs(daily_pred) + 1e-6)
    position_size = np.clip(kelly_fraction * 0.25, 0, 0.10) * 100

    results.update(
        {
            "success": True,
            "composite_score": composite_score,
            "technical_score": technical_score,
            "ml_quality_score": ml_quality_score,
            "fundamental_score": fundamental_score,
            "recommendation": recommendation,
            "confidence": confidence,
            "investment_horizon": horizon,
            "position_size_pct": position_size,
            "warnings": ",".join(warning_flags) if warning_flags else "NONE",
            "daily_prediction": timeframe_results.get("daily", {}).get("prediction", np.nan),
            "intraday_prediction": timeframe_results.get("intraday", {}).get("prediction", np.nan),
            "weekly_prediction": timeframe_results.get("weekly", {}).get("prediction", np.nan),
            "quarterly_prediction": timeframe_results.get("quarterly", {}).get("prediction", np.nan),
            "yearly_prediction": timeframe_results.get("yearly", {}).get("prediction", np.nan),
            "daily_correlation": timeframe_results.get("daily", {}).get("wf_correlation", np.nan),
            "weekly_correlation": timeframe_results.get("weekly", {}).get("wf_correlation", np.nan),
            "quarterly_correlation": timeframe_results.get("quarterly", {}).get("wf_correlation", np.nan),
            "yearly_correlation": timeframe_results.get("yearly", {}).get("wf_correlation", np.nan),
            "avg_wf_correlation": avg_corr,
            "avg_wf_sharpe": avg_sharpe,
            **({key: value for key, value in fundamental.items() if key not in ["success", "ticker"]} if fundamental.get("success") else {}),
            "last_price": float(mtf_data.get("daily", mtf_data[list(mtf_data.keys())[0]])["Close"].iloc[-1]),
            "timeframes_analyzed": list(timeframe_results.keys()),
            "timeframes_count": len(timeframe_results),
        }
    )

    if ENABLE_DEBUG:
        print(f"  ✓ Analysis complete: {recommendation} (Score: {composite_score:.1f})")
    return results
