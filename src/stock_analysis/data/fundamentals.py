import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import yfinance as yf

from stock_analysis.logging import get_logger
from stock_analysis.reliability import RetryPolicy, get_rate_limiter

logger = get_logger(__name__)


@dataclass
class Fundamentals:
    symbol: str
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    source: str = "yfinance"

    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    dividend_yield: Optional[float] = None
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    earnings_date: Optional[datetime] = None
    beta: Optional[float] = None
    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    price_to_sales: Optional[float] = None
    price_to_book: Optional[float] = None
    enterprise_value: Optional[float] = None
    return_on_assets: Optional[float] = None
    return_on_equity: Optional[float] = None
    free_cash_flow: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    total_debt: Optional[float] = None
    total_cash: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None


class FundamentalsCollector:
    def __init__(self, timeout: int = 30, max_retries: int = 3, cache_ttl_hours: int = 24):
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_ttl_hours = cache_ttl_hours

    def _safe_get(self, info: dict, key: str, default: Optional[float] = None) -> Optional[float]:
        value = info.get(key, default)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_get_str(self, info: dict, key: str, default: Optional[str] = None) -> Optional[str]:
        value = info.get(key, default)
        if value is None:
            return None
        return str(value)

    def _safe_get_datetime(self, info: dict, key: str) -> Optional[datetime]:
        value = info.get(key)
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(value)
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None

    def collect(self, symbol: str) -> Fundamentals:
        base_symbol = symbol.replace(".NS", "").replace(".BO", "")
        ticker_symbol = base_symbol if base_symbol.endswith(".NS") else base_symbol + ".NS"

        backoff = RetryPolicy.for_external_data()
        for attempt in range(self.max_retries):
            if attempt:
                time.sleep(backoff.delay(attempt - 1))
            try:
                get_rate_limiter("yfinance").acquire()
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info

                if not info or not info.get("symbol"):
                    logger.warning("fundamentals_empty_response", symbol=symbol, attempt=attempt + 1)
                    continue

                fundamentals = Fundamentals(symbol=symbol)

                fundamentals.pe_ratio = self._safe_get(info, "trailingPE")
                fundamentals.pb_ratio = self._safe_get(info, "priceToBook")
                fundamentals.roe = self._safe_get(info, "returnOnEquity")
                fundamentals.debt_to_equity = self._safe_get(info, "debtToEquity")
                fundamentals.profit_margin = self._safe_get(info, "profitMargins")
                fundamentals.operating_margin = self._safe_get(info, "operatingMargins")
                fundamentals.revenue_growth = self._safe_get(info, "revenueGrowth")
                fundamentals.earnings_growth = self._safe_get(info, "earningsGrowth")
                fundamentals.dividend_yield = self._safe_get(info, "dividendYield")
                fundamentals.market_cap = self._safe_get(info, "marketCap")
                fundamentals.sector = self._safe_get_str(info, "sector")
                fundamentals.industry = self._safe_get_str(info, "industry")
                fundamentals.earnings_date = self._safe_get_datetime(info, "earningsDate")
                fundamentals.beta = self._safe_get(info, "beta")
                fundamentals.trailing_pe = self._safe_get(info, "trailingPE")
                fundamentals.forward_pe = self._safe_get(info, "forwardPE")
                fundamentals.price_to_sales = self._safe_get(info, "priceToSalesTrailing12Months")
                fundamentals.price_to_book = self._safe_get(info, "priceToBook")
                fundamentals.enterprise_value = self._safe_get(info, "enterpriseValue")
                fundamentals.return_on_assets = self._safe_get(info, "returnOnAssets")
                fundamentals.return_on_equity = self._safe_get(info, "returnOnEquity")
                fundamentals.free_cash_flow = self._safe_get(info, "freeCashflow")
                fundamentals.operating_cash_flow = self._safe_get(info, "operatingCashflow")
                fundamentals.total_debt = self._safe_get(info, "totalDebt")
                fundamentals.total_cash = self._safe_get(info, "totalCash")
                fundamentals.current_ratio = self._safe_get(info, "currentRatio")
                fundamentals.quick_ratio = self._safe_get(info, "quickRatio")

                logger.info("fundamentals_collected", symbol=symbol, fields=sum(1 for v in fundamentals.__dict__.values() if v is not None))
                return fundamentals

            except Exception as e:
                logger.warning("fundamentals_collection_failed", symbol=symbol, attempt=attempt + 1, error=str(e))
                if attempt == self.max_retries - 1:
                    logger.error("fundamentals_collection_exhausted", symbol=symbol)
                    return Fundamentals(symbol=symbol)

        return Fundamentals(symbol=symbol)

    def collect_batch(self, symbols: list[str]) -> dict[str, Fundamentals]:
        return {symbol: self.collect(symbol) for symbol in symbols}
