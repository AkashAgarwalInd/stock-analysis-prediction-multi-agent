"""Actual price retrieval for outcome scoring.

Kept separate from ``market.collector`` because evaluation needs the
corporate-action columns (dividends, splits) that the collector discards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional, Protocol

from stock_analysis.config.settings import get_settings
from stock_analysis.logging import get_logger
from stock_analysis.market.calendar import get_trading_calendar
from stock_analysis.reliability import (
    RetryPolicy,
    get_rate_limiter,
    is_retryable_data_error,
    retry_call,
)

logger = get_logger(__name__)

NIFTY_50_SYMBOL = "^NSEI"


@dataclass(frozen=True)
class PriceBar:
    """One daily bar; ``split`` is the split ratio (0 when none), ``dividend`` per share."""

    date: date
    close: float
    dividend: float = 0.0
    split: float = 0.0


@dataclass(frozen=True)
class PriceSeries:
    symbol: str
    bars: list[PriceBar] = field(default_factory=list)
    error: Optional[str] = None

    def window(self, start: date, end: date) -> list[PriceBar]:
        """Bars with ``start <= date <= end``; anything after ``end`` is never used."""
        return sorted((b for b in self.bars if start <= b.date <= end), key=lambda b: b.date)


def is_placeholder_bar(day: date, volume: Any) -> bool:
    """A zero-volume bar on an NSE holiday: yfinance repeats the previous close for some
    stocks on holidays, which is not a session (Muhurat sessions have volume and are kept)."""
    try:
        no_volume = volume is not None and float(volume) == 0.0
    except (TypeError, ValueError):
        return False
    return no_volume and get_trading_calendar().is_holiday(day)


class PriceSource(Protocol):
    def fetch(self, symbol: str, start: date, end: date) -> PriceSeries:
        """Daily bars for ``symbol`` from ``start`` to ``end`` inclusive."""
        ...


class YFinancePriceSource:
    """Daily closes plus dividend/split actions from yfinance (auto-adjusted closes)."""

    def fetch(self, symbol: str, start: date, end: date) -> PriceSeries:
        import yfinance as yf

        try:
            hist = retry_call(
                lambda: yf.Ticker(symbol).history(
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),  # yfinance's end is exclusive
                    interval="1d",
                    actions=True,
                    auto_adjust=True,
                    timeout=get_settings().external_request_timeout_seconds,
                ),
                operation=f"yfinance:actual_prices:{symbol}",
                policy=RetryPolicy.for_external_data(),
                retryable=is_retryable_data_error,
                limiter=get_rate_limiter("yfinance"),
            )
        except Exception as err:  # the SDK raises many unrelated error types
            logger.warning("actual_prices_fetch_failed", symbol=symbol, error=str(err))
            return PriceSeries(symbol=symbol, error=str(err))

        bars = []
        for idx, row in hist.iterrows():
            close = row.get("Close")
            if close is None or math.isnan(float(close)):
                continue
            day = idx.date() if hasattr(idx, "date") else idx
            if is_placeholder_bar(day, row.get("Volume")):
                continue
            bars.append(
                PriceBar(
                    date=day,
                    close=float(close),
                    dividend=float(row.get("Dividends", 0.0) or 0.0),
                    split=float(row.get("Stock Splits", 0.0) or 0.0),
                )
            )
        return PriceSeries(symbol=symbol, bars=bars)
