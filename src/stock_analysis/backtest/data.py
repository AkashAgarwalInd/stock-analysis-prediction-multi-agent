"""Historical market data for backtests, with a point-in-time view (Plan.md §33).

A backtest loads each symbol's full daily history once. The forecast side only
ever receives a ``PointInTimeView`` cut at the as-of date, which cannot return
a bar dated after it. Evaluation receives ``HistoricalPriceSource``, which only
serves prices up to the session the backtest clock has reached.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from stock_analysis.config.settings import get_settings
from stock_analysis.logging import get_logger
from stock_analysis.reliability import (
    RetryPolicy,
    get_rate_limiter,
    is_retryable_data_error,
    retry_call,
)
from stock_analysis.review.prices import PriceBar, PriceSeries, is_placeholder_bar

logger = get_logger(__name__)

INDIA_VIX_SYMBOL = "^INDIAVIX"
# The live quant node uses period="2y"; the view keeps the same calendar span
QUANT_HISTORY_DAYS = 730


class LookaheadError(RuntimeError):
    """Something asked for data dated after the point in time it is allowed to see."""


@dataclass(frozen=True)
class HistoricalBar:
    """One daily OHLCV bar (prices adjusted for splits and dividends by the source)."""

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    dividend: float = 0.0
    split: float = 0.0


class HistoricalMarketData:
    """Full daily histories per symbol, loaded once for a backtest."""

    def __init__(self, bars: Mapping[str, Iterable[HistoricalBar]]):
        self._bars = {
            symbol: sorted(series, key=lambda b: b.date) for symbol, series in bars.items()
        }

    @property
    def symbols(self) -> list[str]:
        return sorted(self._bars)

    def has_bar(self, symbol: str, day: date) -> bool:
        return any(b.date == day for b in self._bars.get(symbol, []))

    def last_date(self, symbol: str) -> Optional[date]:
        bars = self._bars.get(symbol)
        return bars[-1].date if bars else None

    def as_of(self, day: date) -> PointInTimeView:
        """What was known after the close on ``day``: nothing dated later."""
        return PointInTimeView(
            {s: [b for b in bars if b.date <= day] for s, bars in self._bars.items()}, day
        )

    def price_source(self) -> HistoricalPriceSource:
        return HistoricalPriceSource(self._bars)


class PointInTimeView:
    """Bars up to and including ``as_of_date``; the only data a forecast may use."""

    def __init__(self, bars: dict[str, list[HistoricalBar]], as_of_date: date):
        if any(b.date > as_of_date for series in bars.values() for b in series):
            raise LookaheadError(f"View for {as_of_date} was given later bars")
        self._bars = bars
        self.as_of_date = as_of_date

    def bars(self, symbol: str, *, days: Optional[int] = None) -> list[HistoricalBar]:
        """Bars for ``symbol``; with ``days``, only those in the trailing calendar span."""
        series = self._bars.get(symbol, [])
        if days is None:
            return list(series)
        start = self.as_of_date - timedelta(days=days)
        return [b for b in series if b.date > start]

    def quant_prices(self, symbol: str) -> list[HistoricalBar]:
        """The ``PriceFetcher`` for the graph: the same 2-year span the live node uses."""
        return self.bars(symbol, days=QUANT_HISTORY_DAYS)

    def frame(self, symbol: str, *, days: Optional[int] = None) -> pd.DataFrame:
        """OHLCV frame (lower-case columns, date index) for indicator calculations."""
        rows = self.bars(symbol, days=days)
        return pd.DataFrame(
            {
                "open": [b.open for b in rows],
                "high": [b.high for b in rows],
                "low": [b.low for b in rows],
                "close": [b.close for b in rows],
                "volume": [b.volume for b in rows],
            },
            index=pd.DatetimeIndex([pd.Timestamp(b.date) for b in rows]),
        )


class HistoricalPriceSource:
    """``PriceSource`` for outcome scoring that never serves bars beyond ``available_until``.

    The backtest advances ``available_until`` to the session it has evaluated, so a
    review can only see prices that existed at that point in the simulation.
    """

    def __init__(self, bars: dict[str, list[HistoricalBar]]):
        self._bars = bars
        self.available_until: Optional[date] = None

    def fetch(self, symbol: str, start: date, end: date) -> PriceSeries:
        if self.available_until is None or end > self.available_until:
            raise LookaheadError(
                f"Requested {symbol} prices up to {end}; the backtest has only reached "
                f"{self.available_until}"
            )
        series = self._bars.get(symbol)
        if not series:
            return PriceSeries(symbol=symbol, error="no historical data loaded")
        return PriceSeries(
            symbol=symbol,
            bars=[
                PriceBar(date=b.date, close=b.close, dividend=b.dividend, split=b.split)
                for b in series
                if start <= b.date <= end
            ],
        )


def load_yfinance_history(symbols: Iterable[str], start: date, end: date) -> HistoricalMarketData:
    """Download adjusted daily OHLCV (with dividend/split columns) for each symbol."""
    import yfinance as yf

    data: dict[str, list[HistoricalBar]] = {}
    for symbol in symbols:
        try:
            hist = retry_call(
                lambda symbol=symbol: yf.Ticker(symbol).history(
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),  # yfinance's end is exclusive
                    interval="1d",
                    actions=True,
                    auto_adjust=True,
                    timeout=get_settings().external_request_timeout_seconds,
                ),
                operation=f"yfinance:backtest_history:{symbol}",
                policy=RetryPolicy.for_external_data(),
                retryable=is_retryable_data_error,
                limiter=get_rate_limiter("yfinance"),
            )
        except Exception as err:  # the SDK raises many unrelated error types
            logger.warning("backtest_history_failed", symbol=symbol, error=str(err))
            data[symbol] = []
            continue
        bars = []
        for idx, row in hist.iterrows():
            close = row.get("Close")
            if close is None or math.isnan(float(close)):
                continue
            day = idx.date() if hasattr(idx, "date") else idx
            if is_placeholder_bar(day, row.get("Volume")):
                continue
            bars.append(
                HistoricalBar(
                    date=day,
                    open=float(row.get("Open", close)),
                    high=float(row.get("High", close)),
                    low=float(row.get("Low", close)),
                    close=float(close),
                    volume=float(row.get("Volume", 0.0) or 0.0),
                    dividend=float(row.get("Dividends", 0.0) or 0.0),
                    split=float(row.get("Stock Splits", 0.0) or 0.0),
                )
            )
        data[symbol] = bars
        logger.info("backtest_history_loaded", symbol=symbol, bars=len(bars))
    return HistoricalMarketData(data)
