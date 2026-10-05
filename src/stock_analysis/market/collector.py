import time
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import pandas as pd
import yfinance as yf

from stock_analysis.logging import get_logger
from stock_analysis.reliability import get_rate_limiter

if TYPE_CHECKING:
    from stock_analysis.market.cache import PriceCache

logger = get_logger(__name__)


@dataclass
class PriceData:
    symbol: str
    date: date
    open: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    close: Optional[Decimal] = None
    adj_close: Optional[Decimal] = None
    volume: Optional[int] = None
    source: str = "yfinance"
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "date": self.date.isoformat(),
            "open": str(self.open) if self.open else None,
            "high": str(self.high) if self.high else None,
            "low": str(self.low) if self.low else None,
            "close": str(self.close) if self.close else None,
            "adj_close": str(self.adj_close) if self.adj_close else None,
            "volume": self.volume,
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
        }


@dataclass
class PriceHistory:
    symbol: str
    data: list[PriceData]
    source: str = "yfinance"
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    def to_dataframe(self) -> pd.DataFrame:
        if not self.data:
            return pd.DataFrame()
        rows = [d.to_dict() for d in self.data]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def get_close_series(self) -> pd.Series:
        df = self.to_dataframe()
        if "close" in df.columns:
            return df["close"]
        return pd.Series(dtype=float)


class MarketPriceCollector:
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    BASE_BACKOFF = 1.0
    MAX_BACKOFF = 30.0

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        base_backoff: float = BASE_BACKOFF,
        max_backoff: float = MAX_BACKOFF,
        cache: Optional["PriceCache"] = None,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self._cache = cache

    def fetch_history(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d",
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> PriceHistory:
        cache_key = f"{symbol}:{period}:{interval}:{start}:{end}"
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached:
                logger.debug("cache_hit", symbol=symbol, cache_key=cache_key)
                return cached

        for attempt in range(self.max_retries + 1):
            try:
                get_rate_limiter("yfinance").acquire()
                ticker = yf.Ticker(symbol)
                if start and end:
                    hist = ticker.history(
                        start=start.isoformat(),
                        end=end.isoformat(),
                        interval=interval,
                        timeout=self.timeout,
                    )
                else:
                    hist = ticker.history(
                        period=period,
                        interval=interval,
                        timeout=self.timeout,
                    )

                if hist.empty:
                    logger.warning("no_data_returned", symbol=symbol, attempt=attempt)
                    if attempt < self.max_retries:
                        self._backoff(attempt)
                        continue
                    return PriceHistory(symbol=symbol, data=[], source="yfinance")

                price_data = []
                for idx, row in hist.iterrows():
                    pd_obj = PriceData(
                        symbol=symbol,
                        date=idx.date() if hasattr(idx, "date") else idx,
                        open=Decimal(str(row.get("Open", 0))) if pd.notna(row.get("Open")) else None,
                        high=Decimal(str(row.get("High", 0))) if pd.notna(row.get("High")) else None,
                        low=Decimal(str(row.get("Low", 0))) if pd.notna(row.get("Low")) else None,
                        close=Decimal(str(row.get("Close", 0))) if pd.notna(row.get("Close")) else None,
                        adj_close=Decimal(str(row.get("Adj Close", 0))) if pd.notna(row.get("Adj Close")) else None,
                        volume=int(row.get("Volume", 0)) if pd.notna(row.get("Volume")) else None,
                        source="yfinance",
                        fetched_at=datetime.utcnow(),
                    )
                    price_data.append(pd_obj)

                history = PriceHistory(
                    symbol=symbol,
                    data=price_data,
                    source="yfinance",
                    fetched_at=datetime.utcnow(),
                    start_date=price_data[0].date if price_data else None,
                    end_date=price_data[-1].date if price_data else None,
                )

                if self._cache:
                    self._cache.set(cache_key, history)

                logger.info("price_history_fetched", symbol=symbol, count=len(price_data))
                return history

            except Exception as e:
                logger.warning("fetch_failed", symbol=symbol, attempt=attempt, error=str(e))
                if attempt < self.max_retries:
                    self._backoff(attempt)
                else:
                    logger.error("fetch_exhausted", symbol=symbol, error=str(e))
                    return PriceHistory(symbol=symbol, data=[], source="yfinance", fetched_at=datetime.utcnow())

        return PriceHistory(symbol=symbol, data=[], source="yfinance", fetched_at=datetime.utcnow())

    def _backoff(self, attempt: int) -> None:
        delay = min(self.base_backoff * (2 ** attempt), self.max_backoff)
        logger.debug("backing_off", attempt=attempt, delay=delay)
        time.sleep(delay)

    def fetch_latest(self, symbol: str) -> Optional[PriceData]:
        history = self.fetch_history(symbol, period="5d")
        if history.data:
            return history.data[-1]
        return None

    def fetch_multiple(
        self,
        symbols: list[str],
        period: str = "2y",
        interval: str = "1d",
    ) -> dict[str, PriceHistory]:
        results = {}
        for symbol in symbols:
            results[symbol] = self.fetch_history(symbol, period, interval)
        return results


@lru_cache(maxsize=1)
def get_price_collector(cache: Optional["PriceCache"] = None) -> MarketPriceCollector:
    return MarketPriceCollector(cache=cache)
