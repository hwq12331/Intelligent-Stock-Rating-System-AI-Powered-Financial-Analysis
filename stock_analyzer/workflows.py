"""User-facing workflows: batch runs, single stock analysis, screening, and exports."""

from __future__ import annotations

import os
import pickle
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .analysis import analyze_stock_complete
from .config import (
    BATCH_SIZE,
    BUILD_TIMESTAMP,
    CHECKPOINT_INTERVAL,
    CURRENT_USER,
    DEFAULT_TICKERS,
    ENABLE_CHECKPOINTS,
    SCRIPT_VERSION,
    TICKER_FILES,
    TIMEFRAMES,
    TRANSACTION_COST,
    MIN_WF_CORRELATION,
)
from .utils import safe_compare, safe_get


def load_tickers() -> tuple[list[str], str | None]:
    """Load tickers from the first supported input file in the current directory."""
    for filename in TICKER_FILES:
        path = Path(filename)
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                tickers = sorted(
                    {
                        line.split("#", 1)[0].strip().upper()
                        for line in handle
                        if line.split("#", 1)[0].strip()
                    }
                )
            return tickers, filename
    return DEFAULT_TICKERS.copy(), None


def run_batch_analysis() -> None:
    """Interactive batch analysis workflow."""
    print("=" * 80)
    print("ULTIMATE MULTI-TIMEFRAME STOCK ANALYZER - COMPLETE")
    print("=" * 80)
    print(f"Version:   {SCRIPT_VERSION}")
    print(f"User:      {CURRENT_USER}")
    print(f"Build:     {BUILD_TIMESTAMP} UTC")
    print(f"Timeframes: {', '.join(TIMEFRAMES[timeframe]['label'] for timeframe in TIMEFRAMES)}")
    print("=" * 80)

    all_tickers, source_file = load_tickers()
    if source_file:
        print(f"\n✓ Loaded {len(all_tickers)} tickers from {source_file}")
    else:
        print(f"\n⚠ Using {len(all_tickers)} default tickers")

    print(f"\n{'=' * 80}")
    print("BATCH CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"Total tickers available: {len(all_tickers)}")
    print(f"Default batch size: {BATCH_SIZE}")

    try:
        batch_start = int(input(f"\nStart from ticker # (0-{len(all_tickers) - 1}): ") or "0")
    except Exception:
        batch_start = 0

    try:
        batch_size = int(input(f"Batch size (default {BATCH_SIZE}): ") or str(BATCH_SIZE))
    except Exception:
        batch_size = BATCH_SIZE

    batch_end = min(batch_start + batch_size, len(all_tickers))
    tickers = all_tickers[batch_start:batch_end]

    print("\nProcessing batch:")
    print(f"  Start index: {batch_start}")
    print(f"  End index:   {batch_end}")
    print(f"  Count:       {len(tickers)}")
    print(f"{'=' * 80}")

    checkpoint_file = f"checkpoint_batch_{batch_start}_{batch_end}.pkl"
    if os.path.exists(checkpoint_file):
        print(f"\n⚠ Checkpoint file found: {checkpoint_file}")
        resume = input("Resume from checkpoint? (y/n): ")
        if resume.lower() == "y":
            try:
                with open(checkpoint_file, "rb") as handle:
                    all_results = pickle.load(handle)
                print(f"✓ Resumed: {len(all_results)} stocks already processed")
                tickers = tickers[len(all_results) :]
                start_offset = len(all_results)
            except Exception:
                print("⚠ Could not load checkpoint, starting fresh")
                all_results = []
                start_offset = 0
        else:
            all_results = []
            start_offset = 0
    else:
        all_results = []
        start_offset = 0

    print(f"\nProcessing {len(tickers)} stocks...")
    print(f"{'=' * 80}\n")

    for index, ticker in enumerate(tickers, start_offset + 1):
        print(f"[{index}/{batch_size}] {ticker}")
        try:
            all_results.append(analyze_stock_complete(ticker))
        except Exception as exc:
            print(f"  ✗ Critical error: {str(exc)[:100]}")
            traceback.print_exc()
            all_results.append({"ticker": ticker, "success": False, "reason": f"critical_error: {str(exc)[:100]}"})

        if ENABLE_CHECKPOINTS and index % CHECKPOINT_INTERVAL == 0:
            try:
                with open(checkpoint_file, "wb") as handle:
                    pickle.dump(all_results, handle)
                print(f"  💾 Checkpoint saved ({index} stocks processed)")
            except Exception as exc:
                print(f"  ⚠ Checkpoint save failed: {exc}")

        time.sleep(0.5)

    print(f"\n{'=' * 80}")
    print("PROCESSING RESULTS")
    print(f"{'=' * 80}")

    df = pd.DataFrame(all_results)
    successful = df[df["success"] == True].copy()
    failed = df[df["success"] == False].copy()

    print(f"Total analyzed: {len(df)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if len(successful) == 0:
        print("\n⚠ No successful analyses")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        failed_file = f"failed_batch_{batch_start}_{timestamp}.csv"
        failed[["ticker", "reason"]].to_csv(failed_file, index=False)
        print(f"Failed results saved to: {failed_file}")
        return

    print(f"\n{'=' * 80}")
    print("APPLYING QUALITY FILTERS")
    print(f"{'=' * 80}")
    initial = len(successful)

    successful = successful[successful["avg_wf_correlation"] > MIN_WF_CORRELATION]
    print(f"  Filter 1 (corr > {MIN_WF_CORRELATION}): {initial} → {len(successful)}")

    successful = successful[successful["composite_score"] > 40]
    print(f"  Filter 2 (score > 40): → {len(successful)}")

    pred_cols = ["daily_prediction", "intraday_prediction", "weekly_prediction", "quarterly_prediction"]
    available_pred_cols = [column for column in pred_cols if column in successful.columns]
    best_pred = successful[available_pred_cols].abs().max(axis=1)
    successful = successful[best_pred > TRANSACTION_COST * 2]
    print(f"  Filter 3 (best_pred > {TRANSACTION_COST * 2:.1%}): → {len(successful)}")

    if len(successful) == 0:
        print("\n⚠ All filtered out")
        return

    successful = successful.sort_values("composite_score", ascending=False)
    successful["rank"] = range(1, len(successful) + 1)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    main_file = f"predictions_batch_{batch_start}_{timestamp}.csv"
    detail_file = f"detailed_batch_{batch_start}_{timestamp}.csv"

    main_cols = [
        "rank",
        "ticker",
        "recommendation",
        "composite_score",
        "position_size_pct",
        "daily_prediction",
        "weekly_prediction",
        "quarterly_prediction",
        "yearly_prediction",
        "fundamental_score",
        "technical_score",
        "ml_quality_score",
        "investment_horizon",
        "confidence",
        "warnings",
        "last_price",
    ]
    successful[[column for column in main_cols if column in successful.columns]].to_csv(main_file, index=False)

    detail_cols = [
        "ticker",
        "composite_score",
        "recommendation",
        "daily_correlation",
        "weekly_correlation",
        "quarterly_correlation",
        "yearly_correlation",
        "pe_trailing",
        "peg_ratio",
        "roe",
        "revenue_growth_annual",
        "profit_margin",
        "debt_to_equity",
        "current_ratio",
        "investment_style",
        "sector",
        "market_cap",
    ]
    successful[[column for column in detail_cols if column in successful.columns]].to_csv(detail_file, index=False)

    if len(failed) > 0:
        failed_file = f"failed_batch_{batch_start}_{timestamp}.csv"
        failed[["ticker", "reason"]].to_csv(failed_file, index=False)
    else:
        failed_file = None

    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    print(f"Average Composite Score: {successful['composite_score'].mean():.1f}")
    print(f"Average Fundamental Score: {successful['fundamental_score'].mean():.1f}")
    print(f"Average ML Quality: {successful['ml_quality_score'].mean():.1f}")
    print(f"Average Daily Prediction: {successful['daily_prediction'].mean():+.2%}")
    print(f"Average Correlation: {successful['avg_wf_correlation'].mean():.3f}")

    print("\nRecommendation Distribution:")
    for recommendation, count in successful["recommendation"].value_counts().items():
        print(f"  {recommendation}: {count}")

    print("\nInvestment Horizon:")
    for horizon, count in successful["investment_horizon"].value_counts().items():
        print(f"  {horizon}: {count}")

    print(f"\n{'=' * 80}")
    print("TOP 20 RECOMMENDATIONS")
    print(f"{'=' * 80}")
    display_cols = [
        "rank",
        "ticker",
        "recommendation",
        "composite_score",
        "daily_prediction",
        "yearly_prediction",
        "position_size_pct",
        "last_price",
    ]
    print(successful[[column for column in display_cols if column in successful.columns]].head(20).to_string(index=False))

    high_conf = successful[
        (successful["recommendation"].isin(["STRONG_BUY", "BUY"])) & (successful["confidence"] == "HIGH")
    ]
    if len(high_conf) > 0:
        print(f"\n{'=' * 80}")
        print(f"HIGH CONFIDENCE OPPORTUNITIES ({len(high_conf)})")
        print(f"{'=' * 80}")
        buy_cols = [
            "ticker",
            "recommendation",
            "composite_score",
            "daily_prediction",
            "yearly_prediction",
            "position_size_pct",
            "investment_horizon",
        ]
        print(high_conf[[column for column in buy_cols if column in high_conf.columns]].to_string(index=False))

    print(f"\n{'=' * 80}")
    print("FILES SAVED")
    print(f"{'=' * 80}")
    print(f"  ✓ {main_file}")
    print(f"  ✓ {detail_file}")
    if failed_file:
        print(f"  ✓ {failed_file}")
    print(f"{'=' * 80}")

    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print(f"\n✓ Checkpoint file cleaned up: {checkpoint_file}")
        except Exception:
            print(f"\n⚠ Could not remove checkpoint: {checkpoint_file}")

    print(f"\n{'=' * 80}")
    print("✅ BATCH ANALYSIS COMPLETE!")
    print(f"{'=' * 80}")
    print(f"Processed: Batch {batch_start} to {batch_end}")
    print(f"Successful predictions: {len(successful)}")
    print(f"High confidence opportunities: {len(high_conf) if len(high_conf) > 0 else 0}")
    print("Ready for investment decisions!")
    print(f"{'=' * 80}\n")

    if batch_end < len(all_tickers):
        remaining = len(all_tickers) - batch_end
        print(f"📊 Progress: {batch_end}/{len(all_tickers)} tickers processed")
        print(f"📋 Remaining: {remaining} tickers")
        resume_range = min(batch_end + batch_size, len(all_tickers))
        response = input(f"\nProcess next batch ({batch_end} to {resume_range})? (y/n): ")
        if response.lower() == "y":
            print(f"\nPlease restart and use batch_start={batch_end}")
    else:
        print("\n✅ ALL TICKERS PROCESSED!")


def merge_batch_results(batch_files: list[str], output_file: str = "combined_results.csv"):
    """Merge multiple batch result CSV files into one ranked output."""
    print(f"Merging {len(batch_files)} batch files...")
    all_dfs = []
    for file in batch_files:
        try:
            df = pd.read_csv(file)
            all_dfs.append(df)
            print(f"  ✓ Loaded {file}: {len(df)} stocks")
        except Exception as exc:
            print(f"  ✗ Failed to load {file}: {exc}")

    if not all_dfs:
        print("No files loaded!")
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("composite_score", ascending=False)
    combined = combined.drop_duplicates(subset=["ticker"], keep="first")
    combined = combined.sort_values("composite_score", ascending=False)
    combined["rank"] = range(1, len(combined) + 1)
    combined.to_csv(output_file, index=False)

    print(f"\n✅ Merged results saved to: {output_file}")
    print(f"Total unique stocks: {len(combined)}")
    print(f"Top stock: {combined.iloc[0]['ticker']} (Score: {combined.iloc[0]['composite_score']:.1f})")
    return combined


def analyze_single_stock(ticker: str, save_report: bool = True) -> dict[str, Any]:
    """Analyze one ticker and optionally save a text report."""
    print(f"\n{'=' * 80}")
    print(f"SINGLE STOCK ANALYSIS: {ticker}")
    print(f"{'=' * 80}\n")

    result = analyze_stock_complete(ticker)
    if not result["success"]:
        print(f"❌ Analysis failed: {result.get('reason', 'unknown')}")
        return result

    print(f"\n{'=' * 80}")
    print(f"ANALYSIS RESULTS: {ticker}")
    print(f"{'=' * 80}")
    print("\n📊 SCORES:")
    print(f"  Composite Score:   {result['composite_score']:.1f}/100")
    print(f"  Fundamental Score: {result['fundamental_score']:.1f}/100")
    print(f"  Technical Score:   {result['technical_score']:.1f}/100")
    print(f"  ML Quality Score:  {result['ml_quality_score']:.1f}/100")

    print("\n🎯 RECOMMENDATION:")
    print(f"  Action:            {result['recommendation']}")
    print(f"  Confidence:        {result['confidence']}")
    print(f"  Investment Horizon: {result['investment_horizon']}")
    print(f"  Position Size:     {result['position_size_pct']:.1f}%")

    print("\n📈 PREDICTIONS:")
    daily = safe_get(result.get("daily_prediction"), 0)
    weekly = safe_get(result.get("weekly_prediction"), 0)
    quarterly = safe_get(result.get("quarterly_prediction"), 0)
    yearly = safe_get(result.get("yearly_prediction"), 0)
    print(f"  Daily (10d):       {daily:+.2%}")
    print(f"  Weekly (21d):      {weekly:+.2%}")
    print(f"  Quarterly (63d):   {quarterly:+.2%}")
    print(f"  Yearly (252d):     {yearly:+.2%}")

    print("\n📊 CORRELATIONS:")
    daily_corr = safe_get(result.get("daily_correlation"), 0)
    weekly_corr = safe_get(result.get("weekly_correlation"), 0)
    quarterly_corr = safe_get(result.get("quarterly_correlation"), 0)
    yearly_corr = safe_get(result.get("yearly_correlation"), 0)
    avg_corr = safe_get(result.get("avg_wf_correlation"), 0)
    print(f"  Daily:             {daily_corr:.3f}")
    print(f"  Weekly:            {weekly_corr:.3f}")
    print(f"  Quarterly:         {quarterly_corr:.3f}")
    print(f"  Yearly:            {yearly_corr:.3f}")
    print(f"  Average:           {avg_corr:.3f}")

    print("\n💰 FUNDAMENTALS:")
    pe = result.get("pe_trailing", "N/A")
    peg = result.get("peg_ratio", "N/A")
    roe = result.get("roe")
    rev_growth = result.get("revenue_growth_annual")
    profit_margin = result.get("profit_margin")
    debt_to_equity = result.get("debt_to_equity", "N/A")
    current_ratio = result.get("current_ratio", "N/A")
    print(f"  P/E Ratio:         {pe if pe == 'N/A' else f'{pe:.2f}'}")
    print(f"  PEG Ratio:         {peg if peg == 'N/A' else f'{peg:.2f}'}")
    print(f"  ROE:               {f'{roe * 100:.1f}%' if roe else 'N/A'}")
    print(f"  Revenue Growth:    {f'{rev_growth * 100:.1f}%' if rev_growth else 'N/A'}")
    print(f"  Profit Margin:     {f'{profit_margin * 100:.1f}%' if profit_margin else 'N/A'}")
    print(f"  Debt/Equity:       {debt_to_equity if debt_to_equity == 'N/A' else f'{debt_to_equity:.0f}'}")
    print(f"  Current Ratio:     {current_ratio if current_ratio == 'N/A' else f'{current_ratio:.2f}'}")

    print("\n🏢 COMPANY INFO:")
    print(f"  Sector:            {result.get('sector', 'Unknown')}")
    print(f"  Industry:          {result.get('industry', 'Unknown')}")
    market_cap = result.get("market_cap")
    if market_cap:
        print(f"  Market Cap:        ${market_cap / 1e9:.1f}B")
    else:
        print("  Market Cap:        N/A")
    print(f"  Investment Style:  {result.get('investment_style', 'Unknown')}")
    print(f"  Last Price:        ${result.get('last_price', 0):.2f}")

    print("\n⚠️  WARNINGS:")
    warnings = result.get("warnings", "NONE")
    if warnings == "NONE":
        print("  ✅ No warnings")
    else:
        for warning in warnings.split(","):
            print(f"  ⚠️  {warning}")

    print("\n🔍 METADATA:")
    print(f"  Timeframes Analyzed: {result.get('timeframes_count', 0)}")
    print(f"  Timeframes:          {', '.join(result.get('timeframes_analyzed', []))}")
    print(f"  Analysis Time:       {result['analysis_timestamp']}")
    print(f"\n{'=' * 80}")

    if save_report:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        report_file = f"report_{ticker}_{timestamp}.txt"
        with open(report_file, "w", encoding="utf-8") as handle:
            handle.write(f"{'=' * 80}\n")
            handle.write("DETAILED STOCK ANALYSIS REPORT\n")
            handle.write(f"{'=' * 80}\n")
            handle.write(f"Ticker: {ticker}\n")
            handle.write(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
            handle.write(f"User: {CURRENT_USER}\n")
            handle.write(f"Version: {SCRIPT_VERSION}\n")
            handle.write(f"{'=' * 80}\n\n")
            for key, value in result.items():
                handle.write(f"{key}: {value}\n")
        print(f"💾 Detailed report saved: {report_file}")

    return result


def quick_screen(tickers: list[str], criteria: dict[str, Any] | None = None):
    """Screen a small set of tickers against simple score thresholds."""
    if criteria is None:
        criteria = {"min_composite_score": 60, "min_fundamental_score": 50}

    print(f"\n{'=' * 80}")
    print(f"QUICK SCREENING: {len(tickers)} STOCKS")
    print(f"{'=' * 80}")
    print(f"Criteria: {criteria}")
    print(f"{'=' * 80}\n")

    passing: list[dict[str, Any]] = []
    failing: list[str] = []
    for index, ticker in enumerate(tickers, 1):
        print(f"[{index}/{len(tickers)}] Screening {ticker}...", end=" ")
        try:
            result = analyze_stock_complete(ticker)
            if not result["success"]:
                print(f"✗ Failed: {result.get('reason', 'unknown')}")
                failing.append(ticker)
                continue

            passes = True
            reasons = []
            min_comp = criteria.get("min_composite_score", 0)
            if safe_compare(result["composite_score"], "<", min_comp):
                passes = False
                reasons.append(f"score={result['composite_score']:.1f}")

            min_fund = criteria.get("min_fundamental_score", 0)
            if safe_compare(result["fundamental_score"], "<", min_fund):
                passes = False
                reasons.append(f"fund={result['fundamental_score']:.1f}")

            if "max_debt_to_equity" in criteria:
                max_debt_to_equity = criteria["max_debt_to_equity"]
                debt_to_equity = result.get("debt_to_equity")
                if debt_to_equity and safe_compare(debt_to_equity, ">", max_debt_to_equity):
                    passes = False
                    reasons.append(f"D/E={debt_to_equity:.0f}")

            if "min_roe" in criteria:
                min_roe = criteria["min_roe"]
                roe = result.get("roe")
                if roe and safe_compare(roe, "<", min_roe):
                    passes = False
                    reasons.append(f"ROE={roe * 100:.1f}%")

            if passes:
                print(f"✓ PASS (Score: {result['composite_score']:.1f})")
                passing.append(result)
            else:
                print(f"✗ FAIL ({', '.join(reasons)})")
                failing.append(ticker)
        except Exception as exc:
            print(f"✗ Error: {str(exc)[:50]}")
            failing.append(ticker)
        time.sleep(0.5)

    print(f"\n{'=' * 80}")
    print("SCREENING RESULTS")
    print(f"{'=' * 80}")
    print(f"Passed:  {len(passing)}/{len(tickers)}")
    print(f"Failed:  {len(failing)}/{len(tickers)}")

    if passing:
        print("\n✅ PASSING STOCKS:")
        for result in sorted(passing, key=lambda item: item["composite_score"], reverse=True):
            print(
                f"  {result['ticker']:6s} | Score: {result['composite_score']:5.1f} | "
                f"{result['recommendation']:12s} | ${result['last_price']:8.2f}"
            )

    return passing, failing


def export_to_excel(csv_file: str, excel_file: str | None = None) -> None:
    """Convert a result CSV file to a formatted multi-sheet Excel workbook."""
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("❌ openpyxl not installed. Install with: pip install openpyxl")
        return

    if excel_file is None:
        excel_file = csv_file.replace(".csv", ".xlsx")

    print(f"Converting {csv_file} to Excel...")
    df = pd.read_csv(csv_file)

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Stocks", index=False)

        strong_buy = df[df["recommendation"] == "STRONG_BUY"]
        if len(strong_buy) > 0:
            strong_buy.to_excel(writer, sheet_name="Strong Buy", index=False)

        buy = df[df["recommendation"] == "BUY"]
        if len(buy) > 0:
            buy.to_excel(writer, sheet_name="Buy", index=False)

        high_quality = df[df["composite_score"] > 70]
        if len(high_quality) > 0:
            high_quality.to_excel(writer, sheet_name="High Quality", index=False)

        summary = pd.DataFrame(
            {
                "Metric": [
                    "Total Stocks",
                    "Average Score",
                    "Average Fundamental",
                    "Average ML Quality",
                    "Strong Buy Count",
                    "Buy Count",
                    "High Quality (>70) Count",
                ],
                "Value": [
                    len(df),
                    f"{df['composite_score'].mean():.1f}",
                    f"{df['fundamental_score'].mean():.1f}",
                    f"{df['ml_quality_score'].mean():.1f}",
                    len(strong_buy),
                    len(buy),
                    len(high_quality),
                ],
            }
        )
        summary.to_excel(writer, sheet_name="Summary", index=False)

    print(f"✅ Excel file created: {excel_file}")
    print("   Sheets: All Stocks, Strong Buy, Buy, High Quality, Summary")
