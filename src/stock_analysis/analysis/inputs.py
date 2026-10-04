"""Live analyst inputs for ``stock-analysis analyze`` (Plan.md Phase 17).

Collects the stock's price history, fundamentals, recent news and the market
context, and reduces each to a compact summary for ``GraphState`` (no
DataFrames or article bodies in state). A source that fails is recorded as
``{"unavailable": <reason>}`` so its analyst reports the gap: collecting
inputs never raises. Price bars after the last completed NSE session (today's
bar while the market is open) are dropped, so the forecast is made from
closing prices only.
"""

from __future__ import annotations

import asyncio
import html
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Optional, Protocol

import pandas as pd

from stock_analysis.data.collector import DataCollector
from stock_analysis.data.fundamentals import Fundamentals
from stock_analysis.data.market_context import MarketContext
from stock_analysis.data.news import NewsCollection
from stock_analysis.database import Database
from stock_analysis.indicators.technical import latest_indicator_summary
from stock_analysis.logging import get_logger
from stock_analysis.market.calendar import TradingCalendar
from stock_analysis.review.outcome_scorer import last_completed_trading_date
from stock_analysis.schemas.graph_state import GraphState
from stock_analysis.versions import QUANT_HISTORY_PERIOD

logger = get_logger(__name__)

MAX_NEWS_ARTICLES = 10
_MAX_NEWS_SUMMARY_CHARS = 200
_DECIMALS = 4


class InputSource(Protocol):
    """Where the live inputs come from (replaced by a fake in tests)."""

    def prices(self, symbol: str) -> list[Any]:
        """Daily bars (``.date``, ``.open``, ``.high``, ``.low``, ``.close``, ``.volume``)."""
        ...

    def fundamentals(self, symbol: str) -> Fundamentals: ...

    def news(self, symbol: str) -> NewsCollection: ...

    def market_context(self, symbol: str, sector: Optional[str]) -> MarketContext: ...


class LiveInputSource:
    """yfinance prices plus the cached fundamentals/news/market-context collectors."""

    def __init__(self, db: Database):
        self._data = DataCollector(db)

    def prices(self, symbol: str) -> list[Any]:
        from stock_analysis.market.collector import get_price_collector

        history = get_price_collector().fetch_history(
            symbol, period=QUANT_HISTORY_PERIOD, interval="1d"
        )
        return list(history.data or [])

    def fundamentals(self, symbol: str) -> Fundamentals:
        return self._data.get_fundamentals(symbol)

    def news(self, symbol: str) -> NewsCollection:
        async def fetch() -> NewsCollection:
            try:
                return await self._data.get_news(symbol)
            finally:
                await self._data.close()

        return asyncio.run(fetch())

    def market_context(self, symbol: str, sector: Optional[str]) -> MarketContext:
        return self._data.get_market_context(symbol, sector)


@dataclass
class AnalysisInputs:
    """The initial graph state plus the bars behind its indicators and quant baseline."""

    state: GraphState
    bars: list[Any] = field(default_factory=list)
    unavailable: dict[str, str] = field(default_factory=dict)


def _plain(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, _DECIMALS)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _compact(record: Any) -> dict[str, Any]:
    """A collector dataclass as a JSON-ready dict without empty fields."""
    return {k: _plain(v) for k, v in asdict(record).items() if v is not None}


def _number(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def price_frame(bars: list[Any]) -> pd.DataFrame:
    """OHLCV frame (lower-case columns, date index) for the indicator calculations."""
    rows = [b for b in bars if b.close is not None]
    return pd.DataFrame(
        {
            "open": [_number(b.open) for b in rows],
            "high": [_number(b.high) for b in rows],
            "low": [_number(b.low) for b in rows],
            "close": [_number(b.close) for b in rows],
            "volume": [_number(b.volume) for b in rows],
        },
        index=pd.DatetimeIndex([pd.Timestamp(b.date) for b in rows]),
    )


def news_summary(news: NewsCollection) -> dict[str, Any]:
    """Article count and the newest headlines (titles and short summaries only)."""
    items = sorted(news.items, key=lambda i: i.published_at, reverse=True)
    return {
        "source": news.source,
        "fetched_at": _plain(news.fetched_at),
        "article_count": len(news.items),
        "articles": [
            {
                "title": html.unescape(item.title),
                "source": item.source,
                "published_at": _plain(item.published_at),
                "summary": html.unescape(item.summary or "")[:_MAX_NEWS_SUMMARY_CHARS] or None,
            }
            for item in items[:MAX_NEWS_ARTICLES]
        ],
    }


def collect_inputs(
    ticker: str,
    symbol: str,
    company_name: str,
    source: InputSource,
    *,
    run_at: Optional[datetime] = None,
    disabled_analysts: Optional[dict[str, str]] = None,
    calendar: Optional[TradingCalendar] = None,
) -> AnalysisInputs:
    """Collect and summarize every input for ``symbol``; a failing source is skipped.

    ``ticker`` is the NSE ticker without suffix (RELIANCE), ``symbol`` the
    yfinance symbol (RELIANCE.NS). Inputs of disabled analysts are not fetched.
    Bars after the last session completed at ``run_at`` (default: now) are dropped.
    """
    disabled = dict(disabled_analysts or {})
    unavailable: dict[str, str] = {}

    def attempt(part: str, load: Any) -> Any:
        try:
            return load()
        except Exception as err:  # a failing source must never block the forecast
            logger.warning("analysis_input_failed", part=part, symbol=symbol, error=str(err))
            unavailable[part] = f"{type(err).__name__}: {err}"[:300]
            return None

    bars = attempt("prices", lambda: source.prices(symbol)) or []
    session = last_completed_trading_date(run_at or datetime.now(UTC), calendar)
    partial = [b for b in bars if b.date > session]
    if partial:
        logger.info(
            "analysis_partial_bars_dropped",
            symbol=symbol,
            last_completed_session=session.isoformat(),
            dropped=[b.date.isoformat() for b in partial],
        )
        bars = [b for b in bars if b.date <= session]
    technical: Optional[dict[str, Any]] = None
    if "technical" not in disabled:
        technical = attempt("technical", lambda: latest_indicator_summary(price_frame(bars)))
        if technical is None and not {"technical", "prices"} & unavailable.keys():
            unavailable["technical"] = f"not enough price history ({len(bars)} daily bars)"

    fundamentals = sector = None
    if "fundamental" not in disabled:
        record = attempt("fundamentals", lambda: source.fundamentals(ticker))
        if record is not None:
            fundamentals, sector = _compact(record), record.sector
    news = None
    if "sentiment" not in disabled:
        collection = attempt("news", lambda: source.news(ticker))
        news = news_summary(collection) if collection is not None else None
    context = None
    if "context" not in disabled:
        record = attempt("market_context", lambda: source.market_context(ticker, sector))
        context = _compact(record) if record is not None else None

    def summary(value: Optional[dict[str, Any]], *parts: str) -> Optional[dict[str, Any]]:
        if value is not None:
            return value
        reasons = [unavailable[p] for p in parts if p in unavailable]
        return {"unavailable": "; ".join(reasons)} if reasons else None

    state = GraphState(
        symbol=ticker,
        resolved_symbol=symbol,
        company_name=company_name,
        run_at=run_at,
        disabled_analysts=disabled,
        technical_indicators_summary=summary(technical, "prices", "technical"),
        fundamentals_summary=summary(fundamentals, "fundamentals"),
        news_summary=summary(news, "news"),
        market_context_summary=summary(context, "market_context"),
    )
    return AnalysisInputs(state=state, bars=bars, unavailable=unavailable)
