"""Fundamental analysis logic."""

from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

from .config import ENABLE_DEBUG
from .utils import safe_compare, safe_divide, safe_get


class FundamentalAnalyzer:
    """Download and score multi-period company fundamentals."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock: Any | None = None

    def get_financial_data(self) -> dict[str, Any]:
        """Download financial statements using the current yfinance API."""
        try:
            self.stock = yf.Ticker(self.ticker)
            info = self.stock.info

            if not info or len(info) < 5:
                return {}

            def _safe_frame(attribute: str) -> pd.DataFrame:
                try:
                    return getattr(self.stock, attribute)
                except Exception:
                    return pd.DataFrame()

            return {
                "info": info,
                "financials_annual": _safe_frame("financials"),
                "balance_sheet_annual": _safe_frame("balance_sheet"),
                "cashflow_annual": _safe_frame("cashflow"),
                "financials_quarterly": _safe_frame("quarterly_financials"),
                "balance_sheet_quarterly": _safe_frame("quarterly_balance_sheet"),
                "cashflow_quarterly": _safe_frame("quarterly_cashflow"),
                "income_stmt_annual": _safe_frame("income_stmt"),
                "income_stmt_quarterly": _safe_frame("quarterly_income_stmt"),
            }
        except Exception as exc:
            if ENABLE_DEBUG:
                print(f"    ✗ Fundamentals fetch error: {str(exc)[:100]}")
            return {}

    def calculate_valuation_metrics(self, data: dict[str, Any]) -> dict[str, Any]:
        """Calculate valuation ratios and a normalized valuation score."""
        info = data.get("info", {})
        metrics: dict[str, Any] = {}

        try:
            pe = info.get("trailingPE")
            peg = info.get("pegRatio")
            pb = info.get("priceToBook")

            metrics["pe_trailing"] = pe
            metrics["pe_forward"] = info.get("forwardPE")
            metrics["peg_ratio"] = peg
            metrics["price_to_book"] = pb
            metrics["price_to_sales"] = info.get("priceToSalesTrailing12Months")
            metrics["ev_to_ebitda"] = info.get("enterpriseToEbitda")

            score = 0
            count = 0

            if safe_compare(pe, ">", 0) and safe_compare(pe, "<", 100):
                if safe_compare(pe, ">=", 10) and safe_compare(pe, "<=", 20):
                    score += 10
                elif (safe_compare(pe, ">=", 5) and safe_compare(pe, "<", 10)) or (
                    safe_compare(pe, ">", 20) and safe_compare(pe, "<=", 30)
                ):
                    score += 7
                else:
                    score += 4
                count += 1

            if safe_compare(peg, ">", 0) and safe_compare(peg, "<", 5):
                if safe_compare(peg, "<", 1):
                    score += 10
                elif safe_compare(peg, "<", 2):
                    score += 7
                else:
                    score += 4
                count += 1

            if safe_compare(pb, ">", 0) and safe_compare(pb, "<", 20):
                if safe_compare(pb, "<", 1):
                    score += 10
                elif safe_compare(pb, "<", 3):
                    score += 7
                else:
                    score += 4
                count += 1

            metrics["valuation_score"] = (score / (count * 10) * 100) if count > 0 else 50
        except Exception as exc:
            if ENABLE_DEBUG:
                print(f"    ⚠ Valuation error: {str(exc)[:50]}")
            metrics["valuation_score"] = 50

        return metrics

    def calculate_profitability_metrics(self, data: dict[str, Any]) -> dict[str, Any]:
        """Calculate profitability metrics and score."""
        info = data.get("info", {})
        metrics: dict[str, Any] = {}

        try:
            profit_margin = info.get("profitMargins")
            roe = info.get("returnOnEquity")

            metrics["gross_margin"] = info.get("grossMargins")
            metrics["operating_margin"] = info.get("operatingMargins")
            metrics["profit_margin"] = profit_margin
            metrics["roe"] = roe
            metrics["roa"] = info.get("returnOnAssets")

            fcf = info.get("freeCashflow")
            net_income = info.get("netIncomeToCommon")
            if safe_compare(fcf, ">", 0) and safe_compare(net_income, ">", 0):
                metrics["fcf_to_net_income"] = safe_divide(fcf, net_income, None)
            else:
                metrics["fcf_to_net_income"] = None

            score = 0
            count = 0
            if safe_compare(profit_margin, ">", 0.20):
                score += 10
                count += 1
            elif safe_compare(profit_margin, ">", 0.10):
                score += 7
                count += 1
            elif safe_compare(profit_margin, ">", 0):
                score += 4
                count += 1

            if safe_compare(roe, ">", 0.20):
                score += 10
                count += 1
            elif safe_compare(roe, ">", 0.15):
                score += 8
                count += 1
            elif safe_compare(roe, ">", 0.10):
                score += 5
                count += 1
            elif safe_compare(roe, ">", 0):
                score += 2
                count += 1

            metrics["profitability_score"] = (score / (count * 10) * 100) if count > 0 else 50
        except Exception as exc:
            if ENABLE_DEBUG:
                print(f"    ⚠ Profitability error: {str(exc)[:50]}")
            metrics["profitability_score"] = 50

        return metrics

    def calculate_growth_metrics(self, data: dict[str, Any]) -> dict[str, Any]:
        """Calculate growth metrics and score."""
        info = data.get("info", {})
        income_q = data.get("income_stmt_quarterly", pd.DataFrame())
        metrics: dict[str, Any] = {}

        try:
            revenue_growth = info.get("revenueGrowth")
            earnings_growth = info.get("earningsGrowth")

            metrics["revenue_growth_annual"] = revenue_growth
            metrics["earnings_growth_annual"] = earnings_growth
            metrics["revenue_growth_quarterly"] = info.get("revenueQuarterlyGrowth")

            if not income_q.empty:
                income_rows = [row for row in income_q.index if "Net Income" in str(row) or "Income" in str(row)]
                if income_rows:
                    recent_earnings = income_q.loc[income_rows[0]].sort_index()
                    if len(recent_earnings) >= 4:
                        improvements = (recent_earnings.diff().iloc[-3:] > 0).sum()
                        metrics["quarterly_earnings_improving"] = improvements >= 2

            score = 0
            count = 0
            if safe_compare(revenue_growth, ">", 0.20):
                score += 10
                count += 1
            elif safe_compare(revenue_growth, ">", 0.10):
                score += 8
                count += 1
            elif safe_compare(revenue_growth, ">", 0.05):
                score += 6
                count += 1
            elif safe_compare(revenue_growth, ">", 0):
                score += 3
                count += 1

            if safe_compare(earnings_growth, ">", 0.25):
                score += 10
                count += 1
            elif safe_compare(earnings_growth, ">", 0.15):
                score += 8
                count += 1
            elif safe_compare(earnings_growth, ">", 0.05):
                score += 5
                count += 1
            elif safe_compare(earnings_growth, ">", 0):
                score += 2
                count += 1

            if metrics.get("quarterly_earnings_improving"):
                score += 10
                count += 1

            metrics["growth_score"] = (score / (count * 10) * 100) if count > 0 else 50
        except Exception as exc:
            if ENABLE_DEBUG:
                print(f"    ⚠ Growth error: {str(exc)[:50]}")
            metrics["growth_score"] = 50

        return metrics

    def calculate_financial_health_metrics(self, data: dict[str, Any]) -> dict[str, Any]:
        """Calculate balance-sheet health metrics and score."""
        info = data.get("info", {})
        metrics: dict[str, Any] = {}

        try:
            current_ratio = info.get("currentRatio")
            debt_to_equity = info.get("debtToEquity")

            metrics["current_ratio"] = current_ratio
            metrics["quick_ratio"] = info.get("quickRatio")
            metrics["debt_to_equity"] = debt_to_equity

            total_debt = info.get("totalDebt")
            total_assets = info.get("totalAssets")
            if safe_compare(total_debt, ">", 0) and safe_compare(total_assets, ">", 0):
                metrics["debt_to_assets"] = safe_divide(total_debt, total_assets, None)
            else:
                metrics["debt_to_assets"] = None

            score = 0
            count = 0
            if safe_compare(current_ratio, ">", 2.5):
                score += 10
                count += 1
            elif safe_compare(current_ratio, ">", 1.5):
                score += 8
                count += 1
            elif safe_compare(current_ratio, ">", 1.0):
                score += 5
                count += 1
            elif current_ratio is not None:
                score += 2
                count += 1

            if debt_to_equity is not None and safe_compare(debt_to_equity, ">=", 0):
                if safe_compare(debt_to_equity, "<", 30):
                    score += 10
                elif safe_compare(debt_to_equity, "<", 50):
                    score += 8
                elif safe_compare(debt_to_equity, "<", 100):
                    score += 5
                elif safe_compare(debt_to_equity, "<", 200):
                    score += 2
                count += 1

            metrics["financial_health_score"] = (score / (count * 10) * 100) if count > 0 else 50
        except Exception as exc:
            if ENABLE_DEBUG:
                print(f"    ⚠ Health error: {str(exc)[:50]}")
            metrics["financial_health_score"] = 50

        return metrics

    def calculate_dividend_metrics(self, data: dict[str, Any]) -> dict[str, Any]:
        """Calculate dividend metrics and score."""
        info = data.get("info", {})
        metrics: dict[str, Any] = {}

        try:
            dividend_yield = info.get("dividendYield")
            payout_ratio = info.get("payoutRatio")

            metrics["dividend_yield"] = dividend_yield
            metrics["payout_ratio"] = payout_ratio

            score = 0
            count = 0
            if safe_compare(dividend_yield, ">", 0.04):
                score += 10
                count += 1
            elif safe_compare(dividend_yield, ">", 0.02):
                score += 7
                count += 1
            elif safe_compare(dividend_yield, ">", 0):
                score += 4
                count += 1

            if payout_ratio is not None:
                if safe_compare(payout_ratio, ">", 0.3) and safe_compare(payout_ratio, "<", 0.6):
                    score += 10
                elif safe_compare(payout_ratio, ">=", 0.6) and safe_compare(payout_ratio, "<", 0.8):
                    score += 6
                elif safe_compare(payout_ratio, "<", 0.3):
                    score += 7
                count += 1

            metrics["dividend_score"] = (score / (count * 10) * 100) if count > 0 else 0
        except Exception:
            metrics["dividend_score"] = 0

        return metrics

    def get_complete_fundamental_analysis(self) -> dict[str, Any]:
        """Return the full fundamental analysis payload for one ticker."""
        try:
            if ENABLE_DEBUG:
                print("    Analyzing fundamentals...")

            data = self.get_financial_data()
            if not data or not data.get("info"):
                return {"success": False, "reason": "no_financial_data"}

            valuation = self.calculate_valuation_metrics(data)
            profitability = self.calculate_profitability_metrics(data)
            growth = self.calculate_growth_metrics(data)
            health = self.calculate_financial_health_metrics(data)
            dividend = self.calculate_dividend_metrics(data)

            fundamental_score = (
                valuation.get("valuation_score", 50) * 0.25
                + profitability.get("profitability_score", 50) * 0.25
                + growth.get("growth_score", 50) * 0.30
                + health.get("financial_health_score", 50) * 0.20
            )

            info = data.get("info", {})
            pe = safe_get(valuation.get("pe_trailing"), 999)
            growth_rate = safe_get(growth.get("revenue_growth_annual"), 0)
            div_yield = safe_get(dividend.get("dividend_yield"), 0)

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
                "valuation_score": valuation.get("valuation_score", 50),
                "profitability_score": profitability.get("profitability_score", 50),
                "growth_score": growth.get("growth_score", 50),
                "financial_health_score": health.get("financial_health_score", 50),
                "dividend_score": dividend.get("dividend_score", 0),
                **valuation,
                **profitability,
                **growth,
                **health,
                **dividend,
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap"),
                "beta": info.get("beta"),
            }
        except Exception as exc:
            if ENABLE_DEBUG:
                print(f"    ✗ Fundamental analysis failed: {str(exc)[:100]}")
            return {"success": False, "reason": str(exc)[:100]}
