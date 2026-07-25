"""Command-line entrypoint helpers."""

from __future__ import annotations

import sys
import traceback
from typing import Sequence

from .config import CURRENT_USER, SCRIPT_VERSION, TICKER_FILES
from .workflows import analyze_single_stock, export_to_excel, merge_batch_results, quick_screen, run_batch_analysis


def print_usage() -> None:
    """Print beginner-friendly usage instructions based on the current code."""
    print(f"\n{'=' * 80}")
    print("USAGE INSTRUCTIONS")
    print(f"{'=' * 80}")
    print("\n1. BATCH ANALYSIS (interactive)")
    print("   python main.py")
    print("   python -m stock_analyzer")
    print("   - Processes a batch of tickers")
    print("   - Supports checkpointing and resume")
    print("   - Writes ranked CSV output files")

    print("\n2. SINGLE STOCK ANALYSIS")
    print("   python main.py single AAPL")
    print("   from stock_analyzer import analyze_single_stock")
    print("   analyze_single_stock('2222.SR')  # Saudi stock example")

    print("\n3. QUICK SCREENING")
    print("   python main.py screen AAPL MSFT GOOGL")
    print("   from stock_analyzer import quick_screen")
    print("   quick_screen(['AAPL', 'MSFT', 'GOOGL'])")

    print("\n4. MERGE BATCH RESULTS")
    print("   python main.py merge predictions_batch_0.csv predictions_batch_100.csv")
    print("   from stock_analyzer import merge_batch_results")
    print("   merge_batch_results(['predictions_batch_0.csv', 'predictions_batch_100.csv'])")

    print("\n5. EXPORT TO EXCEL")
    print("   from stock_analyzer import export_to_excel")
    print("   export_to_excel('predictions_batch_0_timestamp.csv')")

    print(f"\n{'=' * 80}")
    print("REQUIREMENTS")
    print(f"{'=' * 80}")
    print("Required:")
    print("  - yfinance")
    print("  - pandas")
    print("  - numpy")
    print("  - scikit-learn")
    print("\nOptional:")
    print("  - xgboost")
    print("  - lightgbm")
    print("  - pandas_ta")
    print("  - openpyxl (for Excel export)")

    print(f"\n{'=' * 80}")
    print("INPUT FILES")
    print(f"{'=' * 80}")
    print(f"Create one of these files in the working directory: {', '.join(TICKER_FILES)}")
    print("Use one ticker per line, for example:")
    print("  AAPL")
    print("  MSFT")
    print("  2222.SR")

    print(f"\n{'=' * 80}")
    print("OUTPUT FILES")
    print(f"{'=' * 80}")
    print("  predictions_batch_X_TIMESTAMP.csv  - Main ranked results")
    print("  detailed_batch_X_TIMESTAMP.csv     - Fundamental/correlation details")
    print("  failed_batch_X_TIMESTAMP.csv       - Stocks that could not be analyzed")
    print("  checkpoint_batch_X_Y.pkl           - Resume checkpoint during batch runs")
    print("  report_TICKER_TIMESTAMP.txt        - Single-stock text report")
    print(f"{'=' * 80}\n")


def run_cli(argv: Sequence[str] | None = None) -> None:
    """Run the command-line interface."""
    args = list(argv if argv is not None else sys.argv[1:])

    if args:
        command = args[0]
        if command in ["-h", "--help", "help"]:
            print_usage()
            return
        if command in ["-v", "--version", "version"]:
            print(f"Stock Analyzer Version {SCRIPT_VERSION}")
            print(f"User: {CURRENT_USER}")
            return
        if command == "single" and len(args) > 1:
            analyze_single_stock(args[1].upper())
            return
        if command == "merge" and len(args) > 1:
            merge_batch_results(args[1:])
            return
        if command == "screen" and len(args) > 1:
            quick_screen([ticker.upper() for ticker in args[1:]])
            return
        if command == "excel" and len(args) > 1:
            export_to_excel(args[1], args[2] if len(args) > 2 else None)
            return

        print(f"Unknown command: {command}")
        print_usage()
        raise SystemExit(1)

    try:
        run_batch_analysis()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        print("Progress has been saved in checkpoint file")
        print("Restart the script to resume from checkpoint")
        raise SystemExit(0)
    except Exception as exc:
        print("\n\n❌ CRITICAL ERROR:")
        print(str(exc))
        traceback.print_exc()
        raise SystemExit(1)
