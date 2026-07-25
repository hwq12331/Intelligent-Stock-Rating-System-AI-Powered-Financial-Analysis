# Intelligent Stock Rating System

A multi-timeframe stock analysis tool that combines Yahoo Finance market data, basic fundamental scoring, and simple machine-learning forecasts to rank stocks.

The current code can:
- download price history for yearly, quarterly, weekly, daily, and intraday views
- score fundamentals such as valuation, profitability, growth, and financial health
- train small forecasting ensembles per timeframe
- generate batch CSV rankings, single-stock reports, merged result files, and optional Excel exports

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional packages for better model variety or convenience:
- `xgboost`
- `lightgbm`
- `pandas_ta`

## How to run

### 1. Interactive batch analysis

```bash
python main.py
```

or:

```bash
python -m stock_analyzer
```

The script looks for a ticker list in the current working directory. It checks these filenames in order:
- `tickers_sr.txt`
- `symbols.txt`

If neither file exists, it falls back to a small built-in demo list: `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`.

Example ticker file:

```text
aapl
msft
2222.SR
```

During a batch run, the program asks for:
- start index
- batch size
- whether to resume from an existing checkpoint

### 2. Single stock analysis

```bash
python main.py single AAPL
```

Python API:

```python
from stock_analyzer import analyze_single_stock

analyze_single_stock("AAPL")
```

This prints a detailed console summary and writes a text report like:
- `report_AAPL_20260725_0152.txt`

### 3. Quick screening

```bash
python main.py screen AAPL MSFT GOOGL
```

Python API:

```python
from stock_analyzer import quick_screen

quick_screen(["AAPL", "MSFT", "GOOGL"])
```

### 4. Merge batch outputs

```bash
python main.py merge predictions_batch_0_20260725_0100.csv predictions_batch_100_20260725_0130.csv
```

Python API:

```python
from stock_analyzer import merge_batch_results

merge_batch_results([
    "predictions_batch_0_20260725_0100.csv",
    "predictions_batch_100_20260725_0130.csv",
])
```

### 5. Export a CSV result file to Excel

```python
from stock_analyzer import export_to_excel

export_to_excel("predictions_batch_0_20260725_0100.csv")
```

## Inputs and outputs

### Inputs
- ticker symbols supported by Yahoo Finance
- optional ticker list file (`tickers_sr.txt` or `symbols.txt`)

### Main outputs
- `predictions_batch_<start>_<timestamp>.csv`: ranked batch output
- `detailed_batch_<start>_<timestamp>.csv`: deeper metrics for successful stocks
- `failed_batch_<start>_<timestamp>.csv`: failed symbols and reasons
- `checkpoint_batch_<start>_<end>.pkl`: checkpoint for interrupted batch runs
- `report_<ticker>_<timestamp>.txt`: single-stock report
- `.xlsx` workbook exported from a batch CSV when `openpyxl` is installed

## Project structure

```text
.
├── main.py                  # compatibility entrypoint
├── requirements.txt
├── README.md
└── stock_analyzer/
    ├── __init__.py          # public import surface
    ├── __main__.py          # python -m stock_analyzer
    ├── analysis.py          # end-to-end analysis orchestration
    ├── cli.py               # command-line behavior and help text
    ├── config.py            # settings and optional dependency detection
    ├── data_loader.py       # Yahoo Finance downloads and OHLCV cleaning
    ├── fundamentals.py      # fundamental scoring logic
    ├── indicators.py        # pandas indicator fallbacks
    ├── ml.py                # feature engineering and model training
    ├── utils.py             # safe numeric helpers
    └── workflows.py         # batch, screen, merge, and export workflows
```

## Notes and limitations

- Data quality depends on Yahoo Finance availability and ticker support.
- The program uses simple heuristics plus lightweight ML models; it is a research tool, not investment advice.
- Optional libraries improve model variety, but the analyzer still runs without them.
- Batch mode is interactive rather than fully argument-driven.
