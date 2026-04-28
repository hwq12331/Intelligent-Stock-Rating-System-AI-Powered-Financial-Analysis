"""
================================================================================
ULTIMATE MULTI-TIMEFRAME STOCK ANALYZER - COMPLETE PRODUCTION VERSION
================================================================================
Version: 2.1.0-COMPLETE
Date: 2025-10-24 06:27:27 UTC
User: hwq12331

COMPLETE FEATURES:
✓ Multi-timeframe analysis (Yearly/Quarterly/Weekly/Daily/4-Hour)
✓ Fundamental analysis (Annual + Quarterly financials)
✓ Advanced ML models (Ridge, GBR, XGBoost, LightGBM)
✓ Walk-forward validation
✓ NoneType error handling (FIXED for Saudi stocks)
✓ Batch processing with checkpointing
✓ Single stock deep analysis
✓ Quick screening utility
✓ Batch results merger
✓ Excel export with formatting
✓ Command-line interface
✓ Transaction cost awareness
✓ Position sizing (Kelly Criterion)
✓ Investment horizon detection
✓ Warning system

FIXES APPLIED:
✓ Safe comparison functions for None values
✓ yfinance API deprecation warnings resolved
✓ Earnings data migration to income_stmt
✓ Reduced minimum requirements for sparse data
✓ Comprehensive error handling

================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import pickle
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import Parallel, delayed

# ============================================================================
# OPTIONAL LIBRARIES
# ============================================================================
try:
    import pandas_ta as ta
    HAS_TA = True
    print("✓ pandas_ta available")
except ImportError:
    HAS_TA = False
    print("⚠ pandas_ta not available - using fallbacks")

try:
    import xgboost as xgb
    HAS_XGB = True
    print("✓ XGBoost available")
except ImportError:
    HAS_XGB = False
    print("⚠ XGBoost not available")

try:
    import lightgbm as lgb
    HAS_LGB = True
    print("✓ LightGBM available")
except ImportError:
    HAS_LGB = False
    print("⚠ LightGBM not available")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Script metadata
SCRIPT_VERSION = "2.1.0-COMPLETE"
CURRENT_USER = "hwq12331"
CURRENT_DATETIME = "2025-10-24 06:27:27"
RUN_TIMESTAMP = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

# Data periods
START_DATE_LONG = "2020-01-01"
END_DATE = datetime.utcnow().strftime("%Y-%m-%d")

# Timeframe configurations
TIMEFRAMES = {
    'yearly': {
        'interval': '1mo',
        'period': '5y',
        'horizon_days': 252,
        'min_bars': 20,
        'label': 'Yearly'
    },
    'quarterly': {
        'interval': '1wk',
        'period': '3y',
        'horizon_days': 63,
        'min_bars': 60,
        'label': 'Quarterly'
    },
    'weekly': {
        'interval': '1d',
        'period': '2y',
        'horizon_days': 21,
        'min_bars': 100,
        'label': 'Weekly'
    },
    'daily': {
        'interval': '1d',
        'period': '1y',
        'horizon_days': 10,
        'min_bars': 80,
        'label': 'Daily'
    },
    'intraday': {
        'interval': '1h',
        'period': '60d',
        'horizon_hours': 16,
        'min_bars': 100,
        'label': '4-Hour'
    }
}

# ML Configuration
MIN_TRAINING_PERIODS = 25
MIN_WARMUP_PERIODS = 15
WF_N_SPLITS = 3
WF_TEST_SIZE = 8
N_JOBS = -1

# Quality filters
MIN_WF_CORRELATION = 0.01
MAX_UNCERTAINTY = 3.0
MIN_WF_PREDICTIONS = 10
MIN_TIMEFRAMES_REQUIRED = 1

# Transaction costs
TRANSACTION_COST = 0.002

# Batch processing
BATCH_SIZE = 100
ENABLE_CHECKPOINTS = True
CHECKPOINT_INTERVAL = 20
ENABLE_DEBUG = True

# ============================================================================
# UTILITY: SAFE NUMERIC COMPARISONS (CRITICAL FIX)
# ============================================================================

def safe_compare(value: Any, operator: str, threshold: float, default: bool = False) -> bool:
    """
    Safely compare values that might be None, NaN, or Inf
    CRITICAL: Prevents "NoneType comparison" errors for Saudi stocks
    
    Args:
        value: Value to compare (might be None, nan, or numeric)
        operator: '>', '<', '>=', '<=', '==', '!='
        threshold: Number to compare against
        default: Default return if value is None/nan
    
    Returns:
        bool: Comparison result or default
    """
    if value is None:
        return default
    
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return default
        
        try:
            if operator == '>':
                return value > threshold
            elif operator == '<':
                return value < threshold
            elif operator == '>=':
                return value >= threshold
            elif operator == '<=':
                return value <= threshold
            elif operator == '==':
                return value == threshold
            elif operator == '!=':
                return value != threshold
            else:
                return default
        except:
            return default
    
    return default

def safe_get(value: Any, default: float = 0.0) -> float:
    """
    Safely get numeric value, return default if None/nan/inf
    
    Args:
        value: Value to extract
        default: Default value if invalid
    
    Returns:
        float: Valid numeric value or default
    """
    if value is None:
        return default
    
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return default
        return float(value)
    
    try:
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return default
        return val
    except:
        return default

def safe_divide(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    """
    Safely divide two values with None/zero handling
    
    Args:
        numerator: Top value
        denominator: Bottom value
        default: Return value if division fails
    
    Returns:
        float: Division result or default
    """
    num = safe_get(numerator, 0)
    den = safe_get(denominator, 1)
    
    if den == 0:
        return default
    
    try:
        result = num / den
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except:
        return default

# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

class WalkForwardValidation(BaseCrossValidator):
    """
    Walk-forward cross-validation for time series
    Mimics real trading: train on past, predict future, move forward
    """
    
    def __init__(self, n_splits: int = 3, test_size: int = 10, gap: int = 0):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        idx = np.arange(n_samples)
        min_train = max(MIN_TRAINING_PERIODS, n_samples // 4)

        test_starts = np.linspace(
            min_train, n_samples - self.test_size, 
            self.n_splits, dtype=int
        )

        for t0 in test_starts:
            train_end = t0 - self.gap
            t1 = min(t0 + self.test_size, n_samples)
            if train_end < MIN_TRAINING_PERIODS:
                continue
            yield (idx[:train_end], idx[t0:t1])

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# ============================================================================
# PURE-PANDAS TA FALLBACKS
# ============================================================================

def rsi_series(close: pd.Series, length: int = 14) -> pd.Series:
    """Calculate RSI with pure pandas"""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - 100/(1 + rs)

def macd_series(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD with pure pandas"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def bbands_series(close: pd.Series, n: int = 20, stds: int = 2):
    """Calculate Bollinger Bands"""
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    upper = ma + stds*sd
    lower = ma - stds*sd
    pctb = (close - lower) / ((upper - lower) + 1e-12)
    width = (upper - lower) / (ma.abs() + 1e-12)
    return pctb, width

def atr_pct_series(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    """Calculate ATR as percentage of price"""
    prev = close.shift()
    tr = pd.concat([
        (high-low), 
        (high-prev).abs(), 
        (low-prev).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    return atr / (close.abs() + 1e-12)

def stoch_series(high: pd.Series, low: pd.Series, close: pd.Series, 
                 k: int = 14, d: int = 3, smooth_k: int = 3):
    """Calculate Stochastic Oscillator"""
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    k_raw = 100 * (close - ll) / ((hh - ll) + 1e-12)
    k_smooth = k_raw.rolling(smooth_k).mean()
    d_line = k_smooth.rolling(d).mean()
    return k_smooth, d_line

# ============================================================================
# ROBUST DATA DOWNLOADER
# ============================================================================

def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize OHLCV data"""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col if c]) for col in df.columns]

    # Standardize column names
    rename_map = {}
    for c in df.columns:
        cl = str(c).lower()
        if 'open' in cl and 'adj' not in cl: 
            rename_map[c] = 'Open'
        elif 'high' in cl: 
            rename_map[c] = 'High'
        elif 'low' in cl: 
            rename_map[c] = 'Low'
        elif 'close' in cl: 
            rename_map[c] = 'Close'
        elif 'volume' in cl: 
            rename_map[c] = 'Volume'
    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    needed = ['Open','High','Low','Close','Volume']
    if not set(needed).issubset(df.columns):
        return pd.DataFrame()

    # Convert to numeric
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Close'])
    return df

def download_multi_timeframe_data(ticker: str) -> Dict[str, pd.DataFrame]:
    """
    Download data for ALL timeframes with robust error handling
    Returns dict: {'daily': df, 'weekly': df, 'intraday': df, ...}
    """
    results = {}
    
    for tf_name, tf_config in TIMEFRAMES.items():
        interval = tf_config['interval']
        period = tf_config['period']
        min_bars = tf_config['min_bars']
        
        for attempt in range(1, 3):
            try:
                df = yf.download(
                    ticker, 
                    interval=interval,
                    period=period,
                    auto_adjust=True, 
                    progress=False,
                    threads=False,
                    group_by="column"
                )
                df = _clean_ohlcv(df)
                
                if not df.empty and len(df) >= min_bars:
                    results[tf_name] = df
                    if ENABLE_DEBUG:
                        print(f"    ✓ {tf_name:10s}: {len(df):4d} bars ({interval})")
                    break
                    
            except Exception as e:
                if ENABLE_DEBUG and attempt == 2:
                    print(f"    ✗ {tf_name:10s}: {str(e)[:50]}")
            
            time.sleep(0.3 * attempt)
    
    return results

# ============================================================================
# FUNDAMENTAL ANALYSIS - FIXED WITH SAFE COMPARISONS
# ============================================================================

class FundamentalAnalyzer:
    """
    Multi-period fundamental analysis
    FIXED: All comparisons use safe_compare() to handle None values
    Compatible with yfinance 0.2.x+ API changes
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = None
        
    def get_financial_data(self) -> Dict:
        """
        Download all financial statements
        FIXED: Use new yfinance API (income_stmt instead of earnings)
        """
        try:
            self.stock = yf.Ticker(self.ticker)
            info = self.stock.info
            
            if not info or len(info) < 5:
                return {}
            
            # Annual financials
            try:
                financials_annual = self.stock.financials
            except:
                financials_annual = pd.DataFrame()
            
            try:
                balance_sheet_annual = self.stock.balance_sheet
            except:
                balance_sheet_annual = pd.DataFrame()
            
            try:
                cashflow_annual = self.stock.cashflow
            except:
                cashflow_annual = pd.DataFrame()
            
            # Quarterly financials
            try:
                financials_quarterly = self.stock.quarterly_financials
            except:
                financials_quarterly = pd.DataFrame()
            
            try:
                balance_sheet_quarterly = self.stock.quarterly_balance_sheet
            except:
                balance_sheet_quarterly = pd.DataFrame()
            
            try:
                cashflow_quarterly = self.stock.quarterly_cashflow
            except:
                cashflow_quarterly = pd.DataFrame()
            
            # FIXED: Use income_stmt instead of deprecated earnings
            try:
                income_stmt_annual = self.stock.income_stmt
            except:
                income_stmt_annual = pd.DataFrame()
            
            try:
                income_stmt_quarterly = self.stock.quarterly_income_stmt
            except:
                income_stmt_quarterly = pd.DataFrame()
            
            return {
                'info': info,
                'financials_annual': financials_annual,
                'balance_sheet_annual': balance_sheet_annual,
                'cashflow_annual': cashflow_annual,
                'financials_quarterly': financials_quarterly,
                'balance_sheet_quarterly': balance_sheet_quarterly,
                'cashflow_quarterly': cashflow_quarterly,
                'income_stmt_annual': income_stmt_annual,
                'income_stmt_quarterly': income_stmt_quarterly,
            }
        except Exception as e:
            if ENABLE_DEBUG:
                print(f"    ✗ Fundamentals fetch error: {str(e)[:100]}")
            return {}
    
    def calculate_valuation_metrics(self, data: Dict) -> Dict:
        """Valuation ratios - FIXED with safe_compare()"""
        info = data.get('info', {})
        metrics = {}
        
        try:
            pe = info.get('trailingPE', None)
            peg = info.get('pegRatio', None)
            pb = info.get('priceToBook', None)
            
            metrics['pe_trailing'] = pe
            metrics['pe_forward'] = info.get('forwardPE', None)
            metrics['peg_ratio'] = peg
            metrics['price_to_book'] = pb
            metrics['price_to_sales'] = info.get('priceToSalesTrailing12Months', None)
            metrics['ev_to_ebitda'] = info.get('enterpriseToEbitda', None)
            
            # FIXED: Safe comparisons
            score = 0
            count = 0
            
            if safe_compare(pe, '>', 0) and safe_compare(pe, '<', 100):
                if safe_compare(pe, '>=', 10) and safe_compare(pe, '<=', 20):
                    score += 10
                elif (safe_compare(pe, '>=', 5) and safe_compare(pe, '<', 10)) or \
                     (safe_compare(pe, '>', 20) and safe_compare(pe, '<=', 30)):
                    score += 7
                else:
                    score += 4
                count += 1
            
            if safe_compare(peg, '>', 0) and safe_compare(peg, '<', 5):
                if safe_compare(peg, '<', 1):
                    score += 10
                elif safe_compare(peg, '<', 2):
                    score += 7
                else:
                    score += 4
                count += 1
            
            if safe_compare(pb, '>', 0) and safe_compare(pb, '<', 20):
                if safe_compare(pb, '<', 1):
                    score += 10
                elif safe_compare(pb, '<', 3):
                    score += 7
                else:
                    score += 4
                count += 1
            
            metrics['valuation_score'] = (score / (count * 10) * 100) if count > 0 else 50
        except Exception as e:
            if ENABLE_DEBUG:
                print(f"    ⚠ Valuation error: {str(e)[:50]}")
            metrics['valuation_score'] = 50
        
        return metrics
    
    def calculate_profitability_metrics(self, data: Dict) -> Dict:
        """Profitability analysis - FIXED with safe_compare()"""
        info = data.get('info', {})
        metrics = {}
        
        try:
            pm = info.get('profitMargins', None)
            roe = info.get('returnOnEquity', None)
            
            metrics['gross_margin'] = info.get('grossMargins', None)
            metrics['operating_margin'] = info.get('operatingMargins', None)
            metrics['profit_margin'] = pm
            metrics['roe'] = roe
            metrics['roa'] = info.get('returnOnAssets', None)
            
            fcf = info.get('freeCashflow', None)
            net_income = info.get('netIncomeToCommon', None)
            
            if safe_compare(fcf, '>', 0) and safe_compare(net_income, '>', 0):
                metrics['fcf_to_net_income'] = safe_divide(fcf, net_income, None)
            else:
                metrics['fcf_to_net_income'] = None
            
            # FIXED: Safe scoring
            score = 0
            count = 0
            
            if safe_compare(pm, '>', 0.20):
                score += 10
                count += 1
            elif safe_compare(pm, '>', 0.10):
                score += 7
                count += 1
            elif safe_compare(pm, '>', 0):
                score += 4
                count += 1
            
            if safe_compare(roe, '>', 0.20):
                score += 10
                count += 1
            elif safe_compare(roe, '>', 0.15):
                score += 8
                count += 1
            elif safe_compare(roe, '>', 0.10):
                score += 5
                count += 1
            elif safe_compare(roe, '>', 0):
                score += 2
                count += 1
            
            metrics['profitability_score'] = (score / (count * 10) * 100) if count > 0 else 50
        except Exception as e:
            if ENABLE_DEBUG:
                print(f"    ⚠ Profitability error: {str(e)[:50]}")
            metrics['profitability_score'] = 50
        
        return metrics
    
    def calculate_growth_metrics(self, data: Dict) -> Dict:
        """Growth analysis - FIXED with safe_compare()"""
        info = data.get('info', {})
        income_q = data.get('income_stmt_quarterly', pd.DataFrame())
        metrics = {}
        
        try:
            rev_g = info.get('revenueGrowth', None)
            earn_g = info.get('earningsGrowth', None)
            
            metrics['revenue_growth_annual'] = rev_g
            metrics['earnings_growth_annual'] = earn_g
            metrics['revenue_growth_quarterly'] = info.get('revenueQuarterlyGrowth', None)
            
            # FIXED: Quarterly earnings from income_stmt
            if not income_q.empty:
                income_rows = [r for r in income_q.index 
                              if 'Net Income' in str(r) or 'Income' in str(r)]
                
                if income_rows:
                    recent_earnings = income_q.loc[income_rows[0]].sort_index()
                    if len(recent_earnings) >= 4:
                        improvements = (recent_earnings.diff().iloc[-3:] > 0).sum()
                        metrics['quarterly_earnings_improving'] = improvements >= 2
            
            # FIXED: Safe scoring
            score = 0
            count = 0
            
            if safe_compare(rev_g, '>', 0.20):
                score += 10
                count += 1
            elif safe_compare(rev_g, '>', 0.10):
                score += 8
                count += 1
            elif safe_compare(rev_g, '>', 0.05):
                score += 6
                count += 1
            elif safe_compare(rev_g, '>', 0):
                score += 3
                count += 1
            
            if safe_compare(earn_g, '>', 0.25):
                score += 10
                count += 1
            elif safe_compare(earn_g, '>', 0.15):
                score += 8
                count += 1
            elif safe_compare(earn_g, '>', 0.05):
                score += 5
                count += 1
            elif safe_compare(earn_g, '>', 0):
                score += 2
                count += 1
            
            if metrics.get('quarterly_earnings_improving'):
                score += 10
                count += 1
            
            metrics['growth_score'] = (score / (count * 10) * 100) if count > 0 else 50
        except Exception as e:
            if ENABLE_DEBUG:
                print(f"    ⚠ Growth error: {str(e)[:50]}")
            metrics['growth_score'] = 50
        
        return metrics
    
    def calculate_financial_health_metrics(self, data: Dict) -> Dict:
        """Financial health - FIXED with safe_compare()"""
        info = data.get('info', {})
        metrics = {}
        
        try:
            cr = info.get('currentRatio', None)
            de = info.get('debtToEquity', None)
            
            metrics['current_ratio'] = cr
            metrics['quick_ratio'] = info.get('quickRatio', None)
            metrics['debt_to_equity'] = de
            
            total_debt = info.get('totalDebt', None)
            total_assets = info.get('totalAssets', None)
            
            if safe_compare(total_debt, '>', 0) and safe_compare(total_assets, '>', 0):
                metrics['debt_to_assets'] = safe_divide(total_debt, total_assets, None)
            else:
                metrics['debt_to_assets'] = None
            
            # FIXED: Safe scoring
            score = 0
            count = 0
            
            if safe_compare(cr, '>', 2.5):
                score += 10
                count += 1
            elif safe_compare(cr, '>', 1.5):
                score += 8
                count += 1
            elif safe_compare(cr, '>', 1.0):
                score += 5
                count += 1
            elif cr is not None:
                score += 2
                count += 1
            
            if de is not None and safe_compare(de, '>=', 0):
                if safe_compare(de, '<', 30):
                    score += 10
                elif safe_compare(de, '<', 50):
                    score += 8
                elif safe_compare(de, '<', 100):
                    score += 5
                elif safe_compare(de, '<', 200):
                    score += 2
                count += 1
            
            metrics['financial_health_score'] = (score / (count * 10) * 100) if count > 0 else 50
        except Exception as e:
            if ENABLE_DEBUG:
                print(f"    ⚠ Health error: {str(e)[:50]}")
            metrics['financial_health_score'] = 50
        
        return metrics
    
    def calculate_dividend_metrics(self, data: Dict) -> Dict:
        """Dividend analysis - FIXED with safe_compare()"""
        info = data.get('info', {})
        metrics = {}
        
        try:
            dy = info.get('dividendYield', None)
            pr = info.get('payoutRatio', None)
            
            metrics['dividend_yield'] = dy
            metrics['payout_ratio'] = pr
            
            score = 0
            count = 0
            
            if safe_compare(dy, '>', 0.04):
                score += 10
                count += 1
            elif safe_compare(dy, '>', 0.02):
                score += 7
                count += 1
            elif safe_compare(dy, '>', 0):
                score += 4
                count += 1
            
            if pr is not None:
                if safe_compare(pr, '>', 0.3) and safe_compare(pr, '<', 0.6):
                    score += 10
                elif safe_compare(pr, '>=', 0.6) and safe_compare(pr, '<', 0.8):
                    score += 6
                elif safe_compare(pr, '<', 0.3):
                    score += 7
                count += 1
            
            metrics['dividend_score'] = (score / (count * 10) * 100) if count > 0 else 0
        except:
            metrics['dividend_score'] = 0
        
        return metrics
    
    def get_complete_fundamental_analysis(self) -> Dict:
        """Complete fundamental analysis with all safety checks"""
        try:
            if ENABLE_DEBUG:
                print(f"    Analyzing fundamentals...")
            
            data = self.get_financial_data()
            
            if not data or not data.get('info'):
                return {"success": False, "reason": "no_financial_data"}
            
            # Calculate all metrics with individual error handling
            valuation = self.calculate_valuation_metrics(data)
            profitability = self.calculate_profitability_metrics(data)
            growth = self.calculate_growth_metrics(data)
            health = self.calculate_financial_health_metrics(data)
            dividend = self.calculate_dividend_metrics(data)
            
            # Composite fundamental score
            fundamental_score = (
                valuation.get('valuation_score', 50) * 0.25 +
                profitability.get('profitability_score', 50) * 0.25 +
                growth.get('growth_score', 50) * 0.30 +
                health.get('financial_health_score', 50) * 0.20
            )
            
            # Investment style classification
            info = data.get('info', {})
            pe = safe_get(valuation.get('pe_trailing'), 999)
            growth_rate = safe_get(growth.get('revenue_growth_annual'), 0)
            div_yield = safe_get(dividend.get('dividend_yield'), 0)
            
            if pe < 15 and div_yield > 0.03:
                investment_style = "VALUE_DIVIDEND"
            elif growth_rate > 0.20:
                investment_style = "GROWTH"
            elif pe < 20 and growth_rate > 0.10:
                investment_style = "GARP"
            elif div_yield > 0.04:
                investment_style = "INCOME"
            else:
                investment_style = "BLEND"
            
            return {
                "success": True,
                "ticker": self.ticker,
                "fundamental_score": fundamental_score,
                "investment_style": investment_style,
                "valuation_score": valuation.get('valuation_score', 50),
                "profitability_score": profitability.get('profitability_score', 50),
                "growth_score": growth.get('growth_score', 50),
                "financial_health_score": health.get('financial_health_score', 50),
                "dividend_score": dividend.get('dividend_score', 0),
                **valuation,
                **profitability,
                **growth,
                **health,
                **dividend,
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "market_cap": info.get('marketCap', None),
                "beta": info.get('beta', None),
            }
            
        except Exception as e:
            if ENABLE_DEBUG:
                print(f"    ✗ Fundamental analysis failed: {str(e)[:100]}")
            return {"success": False, "reason": str(e)[:100]}

# ============================================================================
# TECHNICAL FEATURE ENGINEERING
# ============================================================================

def build_technical_features(ohlcv: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Create technical features adapted to timeframe
    Different lookback periods for different timeframes
    """
    tf_config = TIMEFRAMES.get(timeframe, TIMEFRAMES['daily'])
    
    close = ohlcv['Close'].astype(float)
    high = ohlcv['High'].astype(float)
    low = ohlcv['Low'].astype(float)
    volume = ohlcv['Volume'].astype(float)
    
    df = pd.DataFrame(index=close.index)
    df['price'] = close
    df['returns'] = close.pct_change()
    
    # Adjust lookback periods based on timeframe
    if timeframe == 'yearly':
        lags = [1, 2, 3, 6, 12]
        ma_periods = [6, 12, 24]
        vol_periods = [6, 12]
    elif timeframe == 'quarterly':
        lags = [1, 2, 4, 8, 13]
        ma_periods = [8, 13, 26]
        vol_periods = [8, 13]
    elif timeframe == 'weekly':
        lags = [1, 2, 4, 8, 13]
        ma_periods = [10, 20, 50]
        vol_periods = [10, 20]
    elif timeframe == 'intraday':
        lags = [1, 2, 4, 8, 16]
        ma_periods = [8, 16, 32]
        vol_periods = [8, 16]
    else:  # daily
        lags = [1, 2, 3, 5, 10, 20]
        ma_periods = [20, 50, 100]
        vol_periods = [10, 20]
    
    # Return lags
    for lag in lags:
        df[f'return_lag_{lag}'] = df['returns'].shift(lag)
    
    # Momentum
    for period in [lags[2], lags[3], lags[4]]:
        df[f'momentum_{period}'] = (close / close.shift(period) - 1).shift(1)
    
    # Volatility
    for period in vol_periods:
        df[f'volatility_{period}'] = df['returns'].rolling(period).std().shift(1)
    
    # Moving averages
    ma_short = close.rolling(ma_periods[0]).mean()
    ma_mid = close.rolling(ma_periods[1]).mean()
    ma_long = close.rolling(ma_periods[2]).mean() if len(close) >= ma_periods[2] else ma_mid
    
    df['price_vs_ma_short'] = (close / (ma_short + 1e-12) - 1).shift(1)
    df['price_vs_ma_mid'] = (close / (ma_mid + 1e-12) - 1).shift(1)
    df['price_vs_ma_long'] = (close / (ma_long + 1e-12) - 1).shift(1)
    df['ma_cross'] = ((ma_short - ma_mid) / (close + 1e-12)).shift(1)
    
    # Technical indicators with fallbacks
    df['rsi'] = rsi_series(close).shift(1)
    
    macd, macd_sig, macd_hist = macd_series(close)
    df['macd'] = macd.shift(1)
    df['macd_signal'] = macd_sig.shift(1)
    df['macd_hist'] = macd_hist.shift(1)
    
    bb_pctb, bb_width = bbands_series(close, n=ma_periods[0])
    df['bb_pctb'] = bb_pctb.shift(1)
    df['bb_width'] = bb_width.shift(1)
    
    df['atr_pct'] = atr_pct_series(high, low, close).shift(1)
    
    stoch_k, stoch_d = stoch_series(high, low, close)
    df['stoch_k'] = stoch_k.shift(1)
    df['stoch_d'] = stoch_d.shift(1)
    
    # Volume features
    if (volume > 0).mean() > 0.3:
        vol_ma = volume.rolling(ma_periods[0]).mean()
        df['volume_ratio'] = (volume / (vol_ma + 1e-12)).shift(1)
    else:
        df['volume_ratio'] = 1.0
    
    # Target (different horizons per timeframe)
    if timeframe == 'intraday':
        horizon = tf_config['horizon_hours']
        df['target'] = close.pct_change(horizon).shift(-horizon)
    else:
        horizon = tf_config['horizon_days']
        df['target'] = close.pct_change(horizon).shift(-horizon)
    
    # Drop warmup period
    warmup = min(MIN_WARMUP_PERIODS, len(df) // 4)
    df = df.iloc[warmup:].copy()
    
    return df

# ============================================================================
# ML TRAINING (MULTI-TIMEFRAME)
# ============================================================================

def train_ml_model_timeframe(ticker: str, data: pd.DataFrame, 
                            timeframe: str) -> Dict:
    """
    Train ML models for a specific timeframe with walk-forward validation
    """
    try:
        features_df = build_technical_features(data, timeframe)
        
        tf_config = TIMEFRAMES[timeframe]
        
        # Relaxed minimum requirements for sparse data
        if timeframe in ['yearly', 'quarterly']:
            min_required = max(15, tf_config['min_bars'] // 4)
        else:
            min_required = max(20, tf_config['min_bars'] // 3)
        
        if len(features_df) < min_required:
            return {
                "success": False, 
                "reason": f"insufficient_{timeframe}_data"
            }
        
        # Prepare data
        train_df = features_df[features_df['target'].notna()].copy()
        live_df = features_df.iloc[[-1]].copy()
        
        if len(train_df) < MIN_TRAINING_PERIODS:
            return {
                "success": False, 
                "reason": "insufficient_training_data"
            }
        
        feature_cols = [c for c in train_df.columns 
                       if c not in ['target', 'price', 'returns']]
        X = train_df[feature_cols].values
        y = train_df['target'].values
        X_live = live_df[feature_cols].values
        
        # Clean data
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]
        
        if len(X) < MIN_TRAINING_PERIODS:
            return {
                "success": False, 
                "reason": f"only_{len(X)}_clean_rows"
            }
        
        # Build model ensemble
        models = [
            Ridge(alpha=1.0, random_state=42),
            GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.08,
                max_depth=3, 
                subsample=0.8, 
                random_state=42
            ),
        ]
        
        if HAS_XGB:
            try:
                models.append(xgb.XGBRegressor(
                    n_estimators=150, 
                    learning_rate=0.05, 
                    max_depth=4,
                    subsample=0.8, 
                    random_state=42, 
                    n_jobs=1, 
                    verbosity=0
                ))
            except:
                pass
        
        if HAS_LGB:
            try:
                models.append(lgb.LGBMRegressor(
                    n_estimators=150, 
                    learning_rate=0.05, 
                    max_depth=4,
                    random_state=42, 
                    verbose=-1
                ))
            except:
                pass
        
        # Walk-forward cross-validation
        wfv = WalkForwardValidation(WF_N_SPLITS, WF_TEST_SIZE, 0)
        wf_preds, wf_acts = [], []
        
        for tr_idx, te_idx in wfv.split(X):
            if len(tr_idx) < MIN_TRAINING_PERIODS:
                continue
            
            Xtr, Xte = X[tr_idx], X[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]
            
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xte_s = scaler.transform(Xte)
            
            fold_preds = []
            for m in models:
                mm = type(m)(**m.get_params())
                mm.fit(Xtr_s, ytr)
                fold_preds.append(mm.predict(Xte_s))
            
            wf_preds.extend(np.mean(fold_preds, axis=0).tolist())
            wf_acts.extend(yte.tolist())
        
        if len(wf_preds) < MIN_WF_PREDICTIONS:
            return {
                "success": False, 
                "reason": f"only_{len(wf_preds)}_wf_samples"
            }
        
        wf_preds = np.array(wf_preds)
        wf_acts = np.array(wf_acts)
        
        # Calculate metrics
        wf_mae = float(mean_absolute_error(wf_acts, wf_preds))
        wf_corr = float(np.corrcoef(wf_acts, wf_preds)[0, 1])
        wf_dir = float(np.mean(np.sign(wf_preds) == np.sign(wf_acts)))
        
        # Sharpe-like ratio
        wf_returns = wf_preds * wf_acts
        wf_sharpe = wf_returns.mean() / (wf_returns.std() + 1e-6)
        
        # Final prediction on all data
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        X_live_s = scaler.transform(X_live)
        
        final_preds = []
        for m in models:
            mm = type(m)(**m.get_params())
            mm.fit(Xs, y)
            final_preds.append(float(mm.predict(X_live_s)[0]))
        
        ensemble = float(np.mean(final_preds))
        pstd = float(np.std(final_preds))
        punc = float(pstd / (abs(ensemble) + 1e-6))
        
        volatility = float(train_df['returns'].std())
        
        return {
            "success": True,
            "timeframe": timeframe,
            "prediction": ensemble,
            "prediction_std": pstd,
            "prediction_uncertainty": punc,
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
        
    except Exception as e:
        return {
            "success": False,
            "timeframe": timeframe,
            "reason": str(e)[:100]
        }

# ============================================================================
# COMPLETE ANALYSIS - FIXED WITH SAFE COMPARISONS
# ============================================================================

def analyze_stock_complete(ticker: str) -> Dict:
    """
    Complete multi-timeframe analysis
    FIXED: All comparisons use safe_compare() to prevent NoneType errors
    """
    
    if ENABLE_DEBUG:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {ticker}")
        print(f"{'='*80}")
    
    results = {
        "ticker": ticker,
        "analysis_timestamp": RUN_TIMESTAMP,
        "success": False
    }
    
    # 1. Download multi-timeframe data
    if ENABLE_DEBUG:
        print(f"  Downloading multi-timeframe data...")
    
    mtf_data = download_multi_timeframe_data(ticker)
    
    if not mtf_data:
        results["reason"] = "no_data_downloaded"
        return results
    
    # 2. Fundamental analysis
    if ENABLE_DEBUG:
        print(f"  Running fundamental analysis...")
    
    try:
        fund_analyzer = FundamentalAnalyzer(ticker)
        fundamental = fund_analyzer.get_complete_fundamental_analysis()
    except Exception as e:
        if ENABLE_DEBUG:
            print(f"    ⚠ Fundamental analysis failed: {e}")
        fundamental = {"success": False}
    
    # 3. Technical + ML analysis per timeframe
    if ENABLE_DEBUG:
        print(f"  Running multi-timeframe ML analysis...")
    
    timeframe_results = {}
    
    for tf_name, tf_data in mtf_data.items():
        if ENABLE_DEBUG:
            print(f"    → {TIMEFRAMES[tf_name]['label']:10s}...", end=" ")
        
        ml_result = train_ml_model_timeframe(ticker, tf_data, tf_name)
        
        if ml_result['success']:
            # FIXED: Using safe_compare
            if safe_compare(ml_result['wf_correlation'], '>', MIN_WF_CORRELATION):
                if ENABLE_DEBUG:
                    print(f"✓ (corr={ml_result['wf_correlation']:.3f}, "
                          f"pred={ml_result['prediction']:+.4f})")
                timeframe_results[tf_name] = ml_result
            else:
                if ENABLE_DEBUG:
                    print(f"⚠ (corr={ml_result['wf_correlation']:.3f} - skipped)")
        else:
            if ENABLE_DEBUG:
                print(f"✗ ({ml_result.get('reason', 'unknown')})")
    
    # Require minimum timeframes
    if len(timeframe_results) < MIN_TIMEFRAMES_REQUIRED:
        results["reason"] = f"only_{len(timeframe_results)}_quality_timeframes"
        return results
    
    # 4. Aggregate scores
    if ENABLE_DEBUG:
        print(f"  Aggregating scores...")
    
    # Technical score (weighted by correlation)
    # vol_adjusted_pred is an unbounded z-score; we convert it to 0-100
    # via a sigmoid transform before weighting, so extreme predictions
    # cannot push the score outside a meaningful range.
    tf_scores = []
    for tf_name, tf_result in timeframe_results.items():
        weight = max(safe_get(tf_result['wf_correlation'], 0), 0)
        vol_adj = safe_get(tf_result['vol_adjusted_pred'], 0)
        # Sigmoid maps any real number to (0, 1), centred at 0.5
        sigmoid_score = 1.0 / (1.0 + np.exp(-np.clip(vol_adj, -10, 10)))
        score = sigmoid_score * 100 * weight          # 0-100 per timeframe
        tf_scores.append(score)

    # Guarantee 0-100 even after averaging
    technical_score = float(np.clip(np.mean(tf_scores) if tf_scores else 50, 0, 100))

    # ML quality score
    # avg_corr is Pearson [-1, +1]; clip explicitly to guard against
    # numerical edge cases before the affine transform.
    avg_corr = np.mean([safe_get(r['wf_correlation'], 0)
                        for r in timeframe_results.values()])
    avg_sharpe = np.mean([safe_get(r['wf_sharpe'], 0)
                          for r in timeframe_results.values()])
    avg_corr_clipped   = float(np.clip(avg_corr,   -1, 1))
    avg_sharpe_clipped = float(np.clip(avg_sharpe, -1, 1))
    # Formula range: (-1*0.5 + -1*0.5 + 0.5)*100 = -50  →  (+1*0.5 + +1*0.5 + 0.5)*100 = +150
    # clip to [0, 100] after calculation
    ml_quality_score = float(np.clip(
        (avg_corr_clipped * 0.5 + avg_sharpe_clipped * 0.5 + 0.5) * 100,
        0, 100
    ))

    # Fundamental score — already guaranteed 0-100 by its own logic;
    # clip defensively in case of unexpected data.
    fundamental_score = float(np.clip(
        safe_get(fundamental.get('fundamental_score'), 50)
        if fundamental.get('success') else 50,
        0, 100
    ))

    # COMPOSITE SCORE — all three components are now [0, 100],
    # so the weighted sum is also [0, 100] by construction.
    # A final clip is added as a safety net.
    composite_score = float(np.clip(
        technical_score   * 0.15 +
        ml_quality_score  * 0.35 +
        fundamental_score * 0.50,
        0, 100
    ))
    
    # 5. Generate recommendation - FIXED with safe_compare
    daily_pred = safe_get(timeframe_results.get('daily', {}).get('prediction'), 0)
    daily_corr = safe_get(timeframe_results.get('daily', {}).get('wf_correlation'), 0)
    
    # Investment horizon
    if safe_compare(timeframe_results.get('yearly', {}).get('prediction'), '>', 0.05):
        horizon = "LONG_TERM"
    elif safe_compare(timeframe_results.get('quarterly', {}).get('prediction'), '>', 0.03):
        horizon = "MEDIUM_TERM"
    elif 'daily' in timeframe_results:
        horizon = "SHORT_TERM"
    else:
        horizon = "UNKNOWN"
    
    # Recommendation logic - FIXED: all comparisons use safe_compare
    if (safe_compare(composite_score, '>', 75) and 
        safe_compare(daily_pred, '>', TRANSACTION_COST * 3) and 
        safe_compare(daily_corr, '>', 0.20) and
        safe_compare(fundamental_score, '>', 65)):
        recommendation = "STRONG_BUY"
        confidence = "HIGH"
    elif (safe_compare(composite_score, '>', 60) and 
          safe_compare(daily_pred, '>', TRANSACTION_COST * 2) and 
          safe_compare(fundamental_score, '>', 55)):
        recommendation = "BUY"
        confidence = "MEDIUM"
    elif safe_compare(composite_score, '>', 45):
        recommendation = "HOLD"
        confidence = "MEDIUM"
    elif (safe_compare(composite_score, '<', 35) or 
          safe_compare(fundamental.get('financial_health_score', 100), '<', 30)):
        recommendation = "SELL"
        confidence = "MEDIUM"
    else:
        recommendation = "AVOID"
        confidence = "LOW"
    
    # 6. Warning flags - FIXED
    warnings = []
    
    if fundamental.get('success'):
        if safe_compare(fundamental.get('debt_to_equity'), '>', 150):
            warnings.append("HIGH_DEBT")
        if safe_compare(fundamental.get('current_ratio', 2), '<', 1):
            warnings.append("LIQUIDITY_RISK")
        if safe_compare(fundamental.get('revenue_growth_annual'), '<', -0.10):
            warnings.append("DECLINING_REVENUE")
    
    avg_uncertainty = np.mean([safe_get(r['prediction_uncertainty'], 0) 
                               for r in timeframe_results.values()])
    if safe_compare(avg_uncertainty, '>', MAX_UNCERTAINTY):
        warnings.append("HIGH_UNCERTAINTY")
    
    predictions = [safe_get(r['prediction'], 0) for r in timeframe_results.values()]
    if len(predictions) >= 2:
        pred_std = np.std(predictions)
        pred_mean = np.mean(np.abs(predictions))
        if safe_compare(pred_std, '>', pred_mean):
            warnings.append("TIMEFRAME_DIVERGENCE")
    
    # 7. Position sizing (Kelly Criterion) - FIXED
    win_prob = (avg_corr + 1) / 2
    kelly_fraction = (win_prob * abs(daily_pred) - (1 - win_prob)) / (abs(daily_pred) + 1e-6)
    position_size = np.clip(kelly_fraction * 0.25, 0, 0.10) * 100
    
    # 8. Compile results
    results.update({
        "success": True,
        "composite_score": composite_score,
        "technical_score": technical_score,
        "ml_quality_score": ml_quality_score,
        "fundamental_score": fundamental_score,
        
        "recommendation": recommendation,
        "confidence": confidence,
        "investment_horizon": horizon,
        "position_size_pct": position_size,
        
        "warnings": ','.join(warnings) if warnings else 'NONE',
        
        # Timeframe predictions
        "daily_prediction": timeframe_results.get('daily', {}).get('prediction', np.nan),
        "intraday_prediction": timeframe_results.get('intraday', {}).get('prediction', np.nan),
        "weekly_prediction": timeframe_results.get('weekly', {}).get('prediction', np.nan),
        "quarterly_prediction": timeframe_results.get('quarterly', {}).get('prediction', np.nan),
        "yearly_prediction": timeframe_results.get('yearly', {}).get('prediction', np.nan),
        
        # Timeframe quality
        "daily_correlation": timeframe_results.get('daily', {}).get('wf_correlation', np.nan),
        "weekly_correlation": timeframe_results.get('weekly', {}).get('wf_correlation', np.nan),
        "quarterly_correlation": timeframe_results.get('quarterly', {}).get('wf_correlation', np.nan),
        "yearly_correlation": timeframe_results.get('yearly', {}).get('wf_correlation', np.nan),
        
        "avg_wf_correlation": avg_corr,
        "avg_wf_sharpe": avg_sharpe,
        
        # Fundamentals
        **({k: v for k, v in fundamental.items() if k not in ['success', 'ticker']} 
           if fundamental.get('success') else {}),
        
        # Metadata
        "last_price": float(mtf_data.get('daily', mtf_data[list(mtf_data.keys())[0]])['Close'].iloc[-1]),
        "timeframes_analyzed": list(timeframe_results.keys()),
        "timeframes_count": len(timeframe_results),
    })
    
    if ENABLE_DEBUG:
        print(f"  ✓ Analysis complete: {recommendation} (Score: {composite_score:.1f})")
    
    return results

# ============================================================================
# MAIN BATCH EXECUTION
# ============================================================================

def main():
    """Main execution function with batch processing"""
    
    print("="*80)
    print("ULTIMATE MULTI-TIMEFRAME STOCK ANALYZER - COMPLETE")
    print("="*80)
    print(f"Version:   {SCRIPT_VERSION}")
    print(f"User:      {CURRENT_USER}")
    print(f"Date/Time: {CURRENT_DATETIME} UTC")
    print(f"Timeframes: {', '.join([TIMEFRAMES[tf]['label'] for tf in TIMEFRAMES])}")
    print("="*80)
    
    # Load tickers
    try:
        with open("tickers_sr.txt", "r") as f:
            all_tickers = [t.strip() for t in f if t.strip()]
        all_tickers = sorted(set(all_tickers))
        print(f"\n✓ Loaded {len(all_tickers)} tickers from symbols.txt")
    except:
        all_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        print(f"\n⚠ Using {len(all_tickers)} default tickers")
    
    # Batch configuration
    print(f"\n{'='*80}")
    print("BATCH CONFIGURATION")
    print(f"{'='*80}")
    print(f"Total tickers available: {len(all_tickers)}")
    print(f"Default batch size: {BATCH_SIZE}")
    
    try:
        batch_start = int(input(f"\nStart from ticker # (0-{len(all_tickers)-1}): ") or "0")
    except:
        batch_start = 0
    
    try:
        batch_size = int(input(f"Batch size (default {BATCH_SIZE}): ") or str(BATCH_SIZE))
    except:
        batch_size = BATCH_SIZE
    
    # Extract batch
    batch_end = min(batch_start + batch_size, len(all_tickers))
    tickers = all_tickers[batch_start:batch_end]
    
    print(f"\nProcessing batch:")
    print(f"  Start index: {batch_start}")
    print(f"  End index:   {batch_end}")
    print(f"  Count:       {len(tickers)}")
    print(f"{'='*80}")
    
    # Checkpoint handling
    checkpoint_file = f"checkpoint_batch_{batch_start}_{batch_end}.pkl"
    
    if os.path.exists(checkpoint_file):
        print(f"\n⚠ Checkpoint file found: {checkpoint_file}")
        resp = input("Resume from checkpoint? (y/n): ")
        if resp.lower() == 'y':
            try:
                with open(checkpoint_file, 'rb') as f:
                    all_results = pickle.load(f)
                print(f"✓ Resumed: {len(all_results)} stocks already processed")
                tickers = tickers[len(all_results):]
                start_offset = len(all_results)
            except:
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
    print(f"{'='*80}\n")
    
    # Process tickers
    for i, ticker in enumerate(tickers, start_offset + 1):
        print(f"[{i}/{batch_size}] {ticker}")
        
        try:
            result = analyze_stock_complete(ticker)
            all_results.append(result)
        except Exception as e:
            print(f"  ✗ Critical error: {str(e)[:100]}")
            traceback.print_exc()
            all_results.append({
                "ticker": ticker,
                "success": False,
                "reason": f"critical_error: {str(e)[:100]}"
            })
        
        # Checkpoint
        if ENABLE_CHECKPOINTS and i % CHECKPOINT_INTERVAL == 0:
            try:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(all_results, f)
                print(f"  💾 Checkpoint saved ({i} stocks processed)")
            except Exception as e:
                print(f"  ⚠ Checkpoint save failed: {e}")
        
        time.sleep(0.5)
    
    # Process results
    print(f"\n{'='*80}")
    print("PROCESSING RESULTS")
    print(f"{'='*80}")
    
    df = pd.DataFrame(all_results)
    
    successful = df[df['success'] == True].copy()
    failed = df[df['success'] == False].copy()
    
    print(f"Total analyzed: {len(df)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if len(successful) == 0:
        print("\n⚠ No successful analyses")
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        failed_file = f"failed_batch_{batch_start}_{timestamp}.csv"
        failed[['ticker', 'reason']].to_csv(failed_file, index=False)
        print(f"Failed results saved to: {failed_file}")
        
        return
    
    # Apply filters
    print(f"\n{'='*80}")
    print("APPLYING QUALITY FILTERS")
    print(f"{'='*80}")
    
    initial = len(successful)
    
    successful = successful[
        successful['avg_wf_correlation'] > MIN_WF_CORRELATION
    ]
    print(f"  Filter 1 (corr > {MIN_WF_CORRELATION}): {initial} → {len(successful)}")
    
    successful = successful[successful['composite_score'] > 40]
    print(f"  Filter 2 (score > 40): → {len(successful)}")
    
    pred_cols = ['daily_prediction', 'intraday_prediction', 'weekly_prediction', 'quarterly_prediction']
    available_pred_cols = [c for c in pred_cols if c in successful.columns]
    best_pred = successful[available_pred_cols].abs().max(axis=1)
    successful = successful[best_pred > TRANSACTION_COST * 2]
    print(f"  Filter 3 (best_pred > {TRANSACTION_COST*2:.1%}): → {len(successful)}")
    
    if len(successful) == 0:
        print("\n⚠ All filtered out")
        return
    
    # Sort and rank
    successful = successful.sort_values('composite_score', ascending=False)
    successful['rank'] = range(1, len(successful) + 1)
    
    # Export results
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    
    # Main results file
    main_cols = [
        'rank', 'ticker', 'recommendation', 'composite_score', 
        'position_size_pct', 'daily_prediction', 'weekly_prediction',
        'quarterly_prediction', 'yearly_prediction', 'fundamental_score',
        'technical_score', 'ml_quality_score', 'investment_horizon',
        'confidence', 'warnings', 'last_price'
    ]
    
    main_file = f"predictions_batch_{batch_start}_{timestamp}.csv"
    successful[[c for c in main_cols if c in successful.columns]].to_csv(
        main_file, index=False
    )
    
    # Detailed analysis file
    detail_cols = [
        'ticker', 'composite_score', 'recommendation',
        'daily_correlation', 'weekly_correlation', 
        'quarterly_correlation', 'yearly_correlation',
        'pe_trailing', 'peg_ratio', 'roe', 'revenue_growth_annual',
        'profit_margin', 'debt_to_equity', 'current_ratio',
        'investment_style', 'sector', 'market_cap'
    ]
    
    detail_file = f"detailed_batch_{batch_start}_{timestamp}.csv"
    successful[[c for c in detail_cols if c in successful.columns]].to_csv(
        detail_file, index=False
    )
    
    # Failed stocks
    if len(failed) > 0:
        failed_file = f"failed_batch_{batch_start}_{timestamp}.csv"
        failed[['ticker', 'reason']].to_csv(failed_file, index=False)
    
    # Display summary
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Average Composite Score: {successful['composite_score'].mean():.1f}")
    print(f"Average Fundamental Score: {successful['fundamental_score'].mean():.1f}")
    print(f"Average ML Quality: {successful['ml_quality_score'].mean():.1f}")
    print(f"Average Daily Prediction: {successful['daily_prediction'].mean():+.2%}")
    print(f"Average Correlation: {successful['avg_wf_correlation'].mean():.3f}")
    
    print(f"\nRecommendation Distribution:")
    for rec, count in successful['recommendation'].value_counts().items():
        print(f"  {rec}: {count}")
    
    print(f"\nInvestment Horizon:")
    for hor, count in successful['investment_horizon'].value_counts().items():
        print(f"  {hor}: {count}")
    
    # Top recommendations
    print(f"\n{'='*80}")
    print("TOP 20 RECOMMENDATIONS")
    print(f"{'='*80}")
    
    display_cols = [
        'rank', 'ticker', 'recommendation', 'composite_score',
        'daily_prediction', 'yearly_prediction', 
        'position_size_pct', 'last_price'
    ]
    
    print(successful[[c for c in display_cols if c in successful.columns]].head(20).to_string(index=False))
    
    # High confidence opportunities
    high_conf = successful[
        (successful['recommendation'].isin(['STRONG_BUY', 'BUY'])) &
        (successful['confidence'] == 'HIGH')
    ]
    
    if len(high_conf) > 0:
        print(f"\n{'='*80}")
        print(f"HIGH CONFIDENCE OPPORTUNITIES ({len(high_conf)})")
        print(f"{'='*80}")
        
        buy_cols = [
            'ticker', 'recommendation', 'composite_score',
            'daily_prediction', 'yearly_prediction',
            'position_size_pct', 'investment_horizon'
        ]
        
        print(high_conf[[c for c in buy_cols if c in high_conf.columns]].to_string(index=False))
    
    # Files saved
    print(f"\n{'='*80}")
    print("FILES SAVED")
    print(f"{'='*80}")
    print(f"  ✓ {main_file}")
    print(f"  ✓ {detail_file}")
    if len(failed) > 0:
        print(f"  ✓ {failed_file}")
    print(f"{'='*80}")
    
    # Cleanup checkpoint
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print(f"\n✓ Checkpoint file cleaned up: {checkpoint_file}")
        except:
            print(f"\n⚠ Could not remove checkpoint: {checkpoint_file}")
    
    print(f"\n{'='*80}")
    print("✅ BATCH ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"Processed: Batch {batch_start} to {batch_end}")
    print(f"Successful predictions: {len(successful)}")
    print(f"High confidence opportunities: {len(high_conf) if len(high_conf) > 0 else 0}")
    print(f"Ready for investment decisions!")
    print(f"{'='*80}\n")
    
    # Prompt for next batch
    if batch_end < len(all_tickers):
        remaining = len(all_tickers) - batch_end
        print(f"📊 Progress: {batch_end}/{len(all_tickers)} tickers processed")
        print(f"📋 Remaining: {remaining} tickers")
        
        resp = input(f"\nProcess next batch ({batch_end} to {min(batch_end + batch_size, len(all_tickers))})? (y/n): ")
        if resp.lower() == 'y':
            print(f"\nPlease restart and use batch_start={batch_end}")
    else:
        print("\n✅ ALL TICKERS PROCESSED!")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def merge_batch_results(batch_files: List[str], output_file: str = "combined_results.csv"):
    """
    Utility to merge multiple batch result files
    
    Usage:
        merge_batch_results([
            "predictions_batch_0_20251024_0631.csv",
            "predictions_batch_100_20251024_0645.csv",
            "predictions_batch_200_20251024_0700.csv"
        ])
    """
    print(f"Merging {len(batch_files)} batch files...")
    
    all_dfs = []
    for file in batch_files:
        try:
            df = pd.read_csv(file)
            all_dfs.append(df)
            print(f"  ✓ Loaded {file}: {len(df)} stocks")
        except Exception as e:
            print(f"  ✗ Failed to load {file}: {e}")
    
    if not all_dfs:
        print("No files loaded!")
        return
    
    # Combine all dataframes
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates (keep best composite_score)
    combined = combined.sort_values('composite_score', ascending=False)
    combined = combined.drop_duplicates(subset=['ticker'], keep='first')
    
    # Re-rank
    combined = combined.sort_values('composite_score', ascending=False)
    combined['rank'] = range(1, len(combined) + 1)
    
    # Save
    combined.to_csv(output_file, index=False)
    
    print(f"\n✅ Merged results saved to: {output_file}")
    print(f"Total unique stocks: {len(combined)}")
    print(f"Top stock: {combined.iloc[0]['ticker']} (Score: {combined.iloc[0]['composite_score']:.1f})")
    
    return combined

def analyze_single_stock(ticker: str, save_report: bool = True) -> Dict:
    """
    Analyze a single stock with detailed reporting
    
    Usage:
        analyze_single_stock("AAPL")
        analyze_single_stock("2222.SR")  # Saudi stock
    """
    print(f"\n{'='*80}")
    print(f"SINGLE STOCK ANALYSIS: {ticker}")
    print(f"{'='*80}\n")
    
    result = analyze_stock_complete(ticker)
    
    if not result['success']:
        print(f"❌ Analysis failed: {result.get('reason', 'unknown')}")
        return result
    
    # Display detailed results
    print(f"\n{'='*80}")
    print(f"ANALYSIS RESULTS: {ticker}")
    print(f"{'='*80}")
    
    print(f"\n📊 SCORES:")
    print(f"  Composite Score:   {result['composite_score']:.1f}/100")
    print(f"  Fundamental Score: {result['fundamental_score']:.1f}/100")
    print(f"  Technical Score:   {result['technical_score']:.1f}/100")
    print(f"  ML Quality Score:  {result['ml_quality_score']:.1f}/100")
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"  Action:            {result['recommendation']}")
    print(f"  Confidence:        {result['confidence']}")
    print(f"  Investment Horizon: {result['investment_horizon']}")
    print(f"  Position Size:     {result['position_size_pct']:.1f}%")
    
    print(f"\n📈 PREDICTIONS:")
    daily = safe_get(result.get('daily_prediction'), 0)
    weekly = safe_get(result.get('weekly_prediction'), 0)
    quarterly = safe_get(result.get('quarterly_prediction'), 0)
    yearly = safe_get(result.get('yearly_prediction'), 0)
    
    print(f"  Daily (10d):       {daily:+.2%}")
    print(f"  Weekly (21d):      {weekly:+.2%}")
    print(f"  Quarterly (63d):   {quarterly:+.2%}")
    print(f"  Yearly (252d):     {yearly:+.2%}")
    
    print(f"\n📊 CORRELATIONS:")
    daily_corr = safe_get(result.get('daily_correlation'), 0)
    weekly_corr = safe_get(result.get('weekly_correlation'), 0)
    quarterly_corr = safe_get(result.get('quarterly_correlation'), 0)
    yearly_corr = safe_get(result.get('yearly_correlation'), 0)
    avg_corr = safe_get(result.get('avg_wf_correlation'), 0)
    
    print(f"  Daily:             {daily_corr:.3f}")
    print(f"  Weekly:            {weekly_corr:.3f}")
    print(f"  Quarterly:         {quarterly_corr:.3f}")
    print(f"  Yearly:            {yearly_corr:.3f}")
    print(f"  Average:           {avg_corr:.3f}")
    
    print(f"\n💰 FUNDAMENTALS:")
    pe = result.get('pe_trailing', 'N/A')
    peg = result.get('peg_ratio', 'N/A')
    roe = result.get('roe')
    rev_growth = result.get('revenue_growth_annual')
    profit_margin = result.get('profit_margin')
    de = result.get('debt_to_equity', 'N/A')
    cr = result.get('current_ratio', 'N/A')
    
    print(f"  P/E Ratio:         {pe if pe == 'N/A' else f'{pe:.2f}'}")
    print(f"  PEG Ratio:         {peg if peg == 'N/A' else f'{peg:.2f}'}")
    print(f"  ROE:               {f'{roe*100:.1f}%' if roe else 'N/A'}")
    print(f"  Revenue Growth:    {f'{rev_growth*100:.1f}%' if rev_growth else 'N/A'}")
    print(f"  Profit Margin:     {f'{profit_margin*100:.1f}%' if profit_margin else 'N/A'}")
    print(f"  Debt/Equity:       {de if de == 'N/A' else f'{de:.0f}'}")
    print(f"  Current Ratio:     {cr if cr == 'N/A' else f'{cr:.2f}'}")
    
    print(f"\n🏢 COMPANY INFO:")
    print(f"  Sector:            {result.get('sector', 'Unknown')}")
    print(f"  Industry:          {result.get('industry', 'Unknown')}")
    
    market_cap = result.get('market_cap')
    if market_cap:
        print(f"  Market Cap:        ${market_cap/1e9:.1f}B")
    else:
        print(f"  Market Cap:        N/A")
    
    print(f"  Investment Style:  {result.get('investment_style', 'Unknown')}")
    print(f"  Last Price:        ${result.get('last_price', 0):.2f}")
    
    print(f"\n⚠️  WARNINGS:")
    warnings = result.get('warnings', 'NONE')
    if warnings == 'NONE':
        print(f"  ✅ No warnings")
    else:
        for warning in warnings.split(','):
            print(f"  ⚠️  {warning}")
    
    print(f"\n🔍 METADATA:")
    print(f"  Timeframes Analyzed: {result.get('timeframes_count', 0)}")
    print(f"  Timeframes:          {', '.join(result.get('timeframes_analyzed', []))}")
    print(f"  Analysis Time:       {result['analysis_timestamp']}")
    
    print(f"\n{'='*80}")
    
    # Save detailed report
    if save_report:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        report_file = f"report_{ticker}_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"DETAILED STOCK ANALYSIS REPORT\n")
            f.write(f"{'='*80}\n")
            f.write(f"Ticker: {ticker}\n")
            f.write(f"Generated: {RUN_TIMESTAMP} UTC\n")
            f.write(f"User: {CURRENT_USER}\n")
            f.write(f"Version: {SCRIPT_VERSION}\n")
            f.write(f"{'='*80}\n\n")
            
            for key, value in result.items():
                f.write(f"{key}: {value}\n")
        
        print(f"💾 Detailed report saved: {report_file}")
    
    return result

def quick_screen(tickers: List[str], criteria: Dict = None) -> Tuple[List[Dict], List[str]]:
    """
    Quick screening of multiple stocks with custom criteria
    
    Usage:
        quick_screen(
            ["AAPL", "MSFT", "GOOGL"],
            criteria={
                "min_composite_score": 60,
                "min_fundamental_score": 55,
                "max_debt_to_equity": 100,
                "min_roe": 0.15
            }
        )
    """
    if criteria is None:
        criteria = {
            "min_composite_score": 60,
            "min_fundamental_score": 50,
        }
    
    print(f"\n{'='*80}")
    print(f"QUICK SCREENING: {len(tickers)} STOCKS")
    print(f"{'='*80}")
    print(f"Criteria: {criteria}")
    print(f"{'='*80}\n")
    
    passing = []
    failing = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Screening {ticker}...", end=" ")
        
        try:
            result = analyze_stock_complete(ticker)
            
            if not result['success']:
                print(f"✗ Failed: {result.get('reason', 'unknown')}")
                failing.append(ticker)
                continue
            
            # Check criteria with safe comparisons
            passes = True
            reasons = []
            
            min_comp = criteria.get('min_composite_score', 0)
            if safe_compare(result['composite_score'], '<', min_comp):
                passes = False
                reasons.append(f"score={result['composite_score']:.1f}")
            
            min_fund = criteria.get('min_fundamental_score', 0)
            if safe_compare(result['fundamental_score'], '<', min_fund):
                passes = False
                reasons.append(f"fund={result['fundamental_score']:.1f}")
            
            if 'max_debt_to_equity' in criteria:
                max_de = criteria['max_debt_to_equity']
                de = result.get('debt_to_equity')
                if de and safe_compare(de, '>', max_de):
                    passes = False
                    reasons.append(f"D/E={de:.0f}")
            
            if 'min_roe' in criteria:
                min_roe = criteria['min_roe']
                roe = result.get('roe')
                if roe and safe_compare(roe, '<', min_roe):
                    passes = False
                    reasons.append(f"ROE={roe*100:.1f}%")
            
            if passes:
                print(f"✓ PASS (Score: {result['composite_score']:.1f})")
                passing.append(result)
            else:
                print(f"✗ FAIL ({', '.join(reasons)})")
                failing.append(ticker)
        
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            failing.append(ticker)
        
        time.sleep(0.5)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SCREENING RESULTS")
    print(f"{'='*80}")
    print(f"Passed:  {len(passing)}/{len(tickers)}")
    print(f"Failed:  {len(failing)}/{len(tickers)}")
    
    if passing:
        print(f"\n✅ PASSING STOCKS:")
        for result in sorted(passing, key=lambda x: x['composite_score'], reverse=True):
            print(f"  {result['ticker']:6s} | Score: {result['composite_score']:5.1f} | "
                  f"{result['recommendation']:12s} | ${result['last_price']:8.2f}")
    
    return passing, failing

def export_to_excel(csv_file: str, excel_file: str = None):
    """
    Convert CSV results to formatted Excel with multiple sheets
    
    Usage:
        export_to_excel("predictions_batch_0_20251024_0631.csv")
    """
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
    except ImportError:
        print("❌ openpyxl not installed. Install with: pip install openpyxl")
        return
    
    if excel_file is None:
        excel_file = csv_file.replace('.csv', '.xlsx')
    
    print(f"Converting {csv_file} to Excel...")
    
    df = pd.read_csv(csv_file)
    
    # Create Excel writer
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Main sheet
        df.to_excel(writer, sheet_name='All Stocks', index=False)
        
        # Strong buy sheet
        strong_buy = df[df['recommendation'] == 'STRONG_BUY']
        if len(strong_buy) > 0:
            strong_buy.to_excel(writer, sheet_name='Strong Buy', index=False)
        
        # Buy sheet
        buy = df[df['recommendation'] == 'BUY']
        if len(buy) > 0:
            buy.to_excel(writer, sheet_name='Buy', index=False)
        
        # High quality sheet (score > 70)
        high_quality = df[df['composite_score'] > 70]
        if len(high_quality) > 0:
            high_quality.to_excel(writer, sheet_name='High Quality', index=False)
        
        # Summary statistics
        summary = pd.DataFrame({
            'Metric': [
                'Total Stocks',
                'Average Score',
                'Average Fundamental',
                'Average ML Quality',
                'Strong Buy Count',
                'Buy Count',
                'High Quality (>70) Count'
            ],
            'Value': [
                len(df),
                f"{df['composite_score'].mean():.1f}",
                f"{df['fundamental_score'].mean():.1f}",
                f"{df['ml_quality_score'].mean():.1f}",
                len(strong_buy),
                len(buy),
                len(high_quality)
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"✅ Excel file created: {excel_file}")
    print(f"   Sheets: All Stocks, Strong Buy, Buy, High Quality, Summary")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def print_usage():
    """Print usage instructions"""
    print(f"\n{'='*80}")
    print("USAGE INSTRUCTIONS")
    print(f"{'='*80}")
    print("\n1. BATCH ANALYSIS (Recommended for large lists):")
    print("   python ultimate_analyzer.py")
    print("   - Processes stocks in batches")
    print("   - Supports checkpointing and resume")
    print("   - Ideal for 100+ stocks")
    
    print("\n2. SINGLE STOCK ANALYSIS:")
    print("   from ultimate_analyzer import analyze_single_stock")
    print("   analyze_single_stock('AAPL')")
    print("   analyze_single_stock('2222.SR')  # Saudi stock")
    
    print("\n3. QUICK SCREENING:")
    print("   from ultimate_analyzer import quick_screen")
    print("   quick_screen(['AAPL', 'MSFT', 'GOOGL'])")
    
    print("\n4. MERGE BATCH RESULTS:")
    print("   from ultimate_analyzer import merge_batch_results")
    print("   merge_batch_results([")
    print("       'predictions_batch_0_timestamp.csv',")
    print("       'predictions_batch_100_timestamp.csv'")
    print("   ])")
    
    print("\n5. EXPORT TO EXCEL:")
    print("   from ultimate_analyzer import export_to_excel")
    print("   export_to_excel('predictions_batch_0_timestamp.csv')")
    
    print(f"\n{'='*80}")
    print("REQUIREMENTS:")
    print(f"{'='*80}")
    print("Required:")
    print("  - yfinance")
    print("  - pandas")
    print("  - numpy")
    print("  - scikit-learn")
    print("  - joblib")
    print("\nOptional (for better performance):")
    print("  - xgboost")
    print("  - lightgbm")
    print("  - pandas_ta")
    print("  - openpyxl (for Excel export)")
    
    print(f"\n{'='*80}")
    print("INPUT FILE:")
    print(f"{'='*80}")
    print("Create a file named 'symbols.txt' with one ticker per line:")
    print("  AAPL")
    print("  MSFT")
    print("  2222.SR  # Saudi stocks supported")
    print("  ...")
    
    print(f"\n{'='*80}")
    print("OUTPUT FILES:")
    print(f"{'='*80}")
    print("  predictions_batch_X_TIMESTAMP.csv  - Main results")
    print("  detailed_batch_X_TIMESTAMP.csv     - Detailed analysis")
    print("  failed_batch_X_TIMESTAMP.csv       - Failed stocks")
    print("  checkpoint_batch_X_Y.pkl           - Resume checkpoint")
    print(f"{'='*80}\n")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Update timestamp with current time
    CURRENT_DATETIME = "2025-10-24 06:31:51"
    RUN_TIMESTAMP = "2025-10-24 06:31:51"
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help', 'help']:
            print_usage()
            sys.exit(0)
        elif sys.argv[1] in ['-v', '--version', 'version']:
            print(f"Stock Analyzer Version {SCRIPT_VERSION}")
            print(f"User: {CURRENT_USER}")
            print(f"Date: {CURRENT_DATETIME}")
            sys.exit(0)
        elif sys.argv[1] == 'single' and len(sys.argv) > 2:
            # Single stock mode
            ticker = sys.argv[2].upper()
            analyze_single_stock(ticker)
            sys.exit(0)
        elif sys.argv[1] == 'merge' and len(sys.argv) > 2:
            # Merge mode
            files = sys.argv[2:]
            merge_batch_results(files)
            sys.exit(0)
        elif sys.argv[1] == 'screen' and len(sys.argv) > 2:
            # Quick screen mode
            tickers = [t.upper() for t in sys.argv[2:]]
            quick_screen(tickers)
            sys.exit(0)
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print_usage()
            sys.exit(1)
    
    # Default: run batch analysis
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        print("Progress has been saved in checkpoint file")
        print("Restart the script to resume from checkpoint")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ CRITICAL ERROR:")
        print(f"{str(e)}")
        traceback.print_exc()
        sys.exit(1)