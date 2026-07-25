"""Compatibility entrypoint for the refactored stock analyzer package."""

from stock_analyzer import (
    analyze_single_stock,
    analyze_stock_complete,
    export_to_excel,
    merge_batch_results,
    print_usage,
    quick_screen,
    run_batch_analysis,
    run_cli,
)

main = run_batch_analysis

__all__ = [
    "analyze_single_stock",
    "analyze_stock_complete",
    "export_to_excel",
    "main",
    "merge_batch_results",
    "print_usage",
    "quick_screen",
    "run_batch_analysis",
    "run_cli",
]

if __name__ == "__main__":
    run_cli()
