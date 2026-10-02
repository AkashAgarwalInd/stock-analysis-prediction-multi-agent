from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import yfinance as yf

from stock_analysis.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MarketContext:
    symbol: str
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    source: str = "yfinance"

    nifty_50: Optional[float] = None
    nifty_50_change: Optional[float] = None
    nifty_50_change_pct: Optional[float] = None
    bank_nifty: Optional[float] = None
    bank_nifty_change: Optional[float] = None
    bank_nifty_change_pct: Optional[float] = None
    india_vix: Optional[float] = None
    india_vix_change: Optional[float] = None
    usd_inr: Optional[float] = None
    usd_inr_change: Optional[float] = None
    sector_index: Optional[float] = None
    sector_index_change: Optional[float] = None
    sector_index_change_pct: Optional[float] = None
    sector_name: Optional[str] = None
    relative_strength_vs_nifty: Optional[float] = None
    beta: Optional[float] = None
    correlation_nifty: Optional[float] = None


SECTOR_TO_INDEX = {
    "Financial Services": "NIFTY_FIN_SERVICE.NS",
    "Banking": "NIFTY_BANK.NS",
    "Information Technology": "NIFTY_IT.NS",
    "Technology": "NIFTY_IT.NS",
    "Consumer Goods": "NIFTY_FMCG.NS",
    "FMCG": "NIFTY_FMCG.NS",
    "Automobile": "NIFTY_AUTO.NS",
    "Auto": "NIFTY_AUTO.NS",
    "Pharma": "NIFTY_PHARMA.NS",
    "Healthcare": "NIFTY_PHARMA.NS",
    "Energy": "NIFTY_ENERGY.NS",
    "Oil & Gas": "NIFTY_ENERGY.NS",
    "Metals": "NIFTY_METAL.NS",
    "Metal": "NIFTY_METAL.NS",
    "Realty": "NIFTY_REALTY.NS",
    "Real Estate": "NIFTY_REALTY.NS",
    "Infrastructure": "NIFTY_INFRA.NS",
    "Media": "NIFTY_MEDIA.NS",
    "Utilities": "NIFTY_UTILITIES.NS",
    "PSU Bank": "NIFTY_PSU_BANK.NS",
    "Private Bank": "NIFTY_PRIVATE_BANK.NS",
}


class MarketContextCollector:
    NIFTY_50_SYMBOL = "^NSEI"
    BANK_NIFTY_SYMBOL = "^NSEBANK"
    INDIA_VIX_SYMBOL = "^INDIAVIX"
    USD_INR_SYMBOL = "INR=X"

    def __init__(self, timeout: int = 30, max_retries: int = 3, cache_ttl_hours: int = 4):
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_ttl_hours = cache_ttl_hours

    def _get_quote(self, symbol: str) -> Optional[dict]:
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if info and info.get("symbol"):
                    return info
            except Exception as e:
                logger.debug("market_context_quote_failed", symbol=symbol, attempt=attempt + 1, error=str(e))
        return None

    def _get_history(self, symbol: str, period: str = "2d") -> Optional[list]:
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if not hist.empty:
                    return hist
            except Exception as e:
                logger.debug("market_context_history_failed", symbol=symbol, attempt=attempt + 1, error=str(e))
        return None

    def _calculate_change(self, current: Optional[float], previous: Optional[float]) -> tuple[Optional[float], Optional[float]]:
        if current is None or previous is None or previous == 0:
            return None, None
        change = current - previous
        change_pct = (change / previous) * 100
        return change, change_pct

    def _get_index_data(self, symbol: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
        info = self._get_quote(symbol)
        if not info:
            return None, None, None

        current = info.get("regularMarketPrice") or info.get("currentPrice")
        previous = info.get("regularMarketPreviousClose") or info.get("previousClose")

        change, change_pct = self._calculate_change(current, previous)
        return current, change, change_pct

    def _get_sector_index(self, sector: Optional[str]) -> tuple[Optional[str], Optional[float], Optional[float], Optional[float]]:
        if not sector:
            return None, None, None, None

        index_symbol = SECTOR_TO_INDEX.get(sector)
        if not index_symbol:
            return sector, None, None, None

        current, change, change_pct = self._get_index_data(index_symbol)
        return sector, current, change, change_pct

    def _calculate_relative_strength(self, symbol: str, nifty_data: tuple) -> Optional[float]:
        stock_hist = self._get_history(symbol + ".NS" if not symbol.endswith(".NS") else symbol, "3mo")
        nifty_hist = self._get_history(self.NIFTY_50_SYMBOL, "3mo")

        if stock_hist is None or nifty_hist is None:
            return None

        try:
            stock_close = stock_hist["Close"]
            nifty_close = nifty_hist["Close"]

            min_len = min(len(stock_close), len(nifty_close))
            if min_len < 20:
                return None

            stock_close = stock_close[-min_len:]
            nifty_close = nifty_close[-min_len:]

            stock_returns = stock_close.pct_change().dropna()
            nifty_returns = nifty_close.pct_change().dropna()

            min_len = min(len(stock_returns), len(nifty_returns))
            if min_len < 20:
                return None

            stock_returns = stock_returns[-min_len:]
            nifty_returns = nifty_returns[-min_len:]

            correlation = stock_returns.corr(nifty_returns)
            if correlation is None:
                return None

            stock_vol = stock_returns.std()
            nifty_vol = nifty_returns.std()

            if nifty_vol == 0:
                return None

            relative_strength = (stock_returns.mean() / stock_vol) / (nifty_returns.mean() / nifty_vol) if stock_vol > 0 else None
            return relative_strength

        except Exception as e:
            logger.debug("relative_strength_calc_failed", symbol=symbol, error=str(e))
            return None

    def collect(self, symbol: str, sector: Optional[str] = None) -> MarketContext:
        base_symbol = symbol.replace(".NS", "").replace(".BO", "")
        full_symbol = base_symbol if base_symbol.endswith(".NS") else base_symbol + ".NS"

        context = MarketContext(symbol=symbol)

        nifty_current, nifty_change, nifty_change_pct = self._get_index_data(self.NIFTY_50_SYMBOL)
        context.nifty_50 = nifty_current
        context.nifty_50_change = nifty_change
        context.nifty_50_change_pct = nifty_change_pct

        bank_nifty_current, bank_nifty_change, bank_nifty_change_pct = self._get_index_data(self.BANK_NIFTY_SYMBOL)
        context.bank_nifty = bank_nifty_current
        context.bank_nifty_change = bank_nifty_change
        context.bank_nifty_change_pct = bank_nifty_change_pct

        vix_current, vix_change, _ = self._get_index_data(self.INDIA_VIX_SYMBOL)
        context.india_vix = vix_current
        context.india_vix_change = vix_change

        usd_inr_current, usd_inr_change, _ = self._get_index_data(self.USD_INR_SYMBOL)
        context.usd_inr = usd_inr_current
        context.usd_inr_change = usd_inr_change

        sector_name, sector_current, sector_change, sector_change_pct = self._get_sector_index(sector)
        context.sector_name = sector_name
        context.sector_index = sector_current
        context.sector_index_change = sector_change
        context.sector_index_change_pct = sector_change_pct

        context.relative_strength_vs_nifty = self._calculate_relative_strength(full_symbol, (nifty_current, nifty_change, nifty_change_pct))

        stock_info = self._get_quote(full_symbol)
        if stock_info:
            context.beta = stock_info.get("beta")
            context.correlation_nifty = None

        logger.info("market_context_collected", symbol=symbol)
        return context

    def collect_batch(self, symbols: list[str], sectors: Optional[dict[str, str]] = None) -> dict[str, MarketContext]:
        results = {}
        for symbol in symbols:
            sector = sectors.get(symbol) if sectors else None
            results[symbol] = self.collect(symbol, sector)
        return results
