from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from stock_analysis.data.cache import DataCache
from stock_analysis.data.fundamentals import Fundamentals, FundamentalsCollector
from stock_analysis.data.market_context import MarketContext, MarketContextCollector
from stock_analysis.data.news import NewsCollection, NewsCollector, NewsItem
from stock_analysis.database import Database
from stock_analysis.logging import get_logger

logger = get_logger(__name__)


_DATETIME_FIELDS = ("fetched_at", "published_at", "earnings_date")


def _restore_datetimes(data: dict) -> dict:
    """Turn ISO strings written by the cache serializer back into datetimes."""
    restored = dict(data)
    for key in _DATETIME_FIELDS:
        if isinstance(restored.get(key), str):
            restored[key] = datetime.fromisoformat(restored[key])
    return restored


def _fundamentals_from_dict(data: dict) -> Fundamentals:
    return Fundamentals(**_restore_datetimes(data))


def _news_from_dict(data: dict) -> NewsCollection:
    data = _restore_datetimes(data)
    data["items"] = [NewsItem(**_restore_datetimes(item)) for item in data.get("items", [])]
    return NewsCollection(**data)


def _market_context_from_dict(data: dict) -> MarketContext:
    return MarketContext(**_restore_datetimes(data))


@dataclass
class CompleteData:
    symbol: str
    fundamentals: Optional[Fundamentals] = None
    news: Optional[NewsCollection] = None
    market_context: Optional[MarketContext] = None
    fetched_at: datetime = None

    def __post_init__(self):
        if self.fetched_at is None:
            self.fetched_at = datetime.utcnow()


class DataCollector:
    def __init__(
        self,
        db: Database,
        fundamentals_ttl_hours: int = 24,
        news_ttl_hours: int = 6,
        market_context_ttl_hours: int = 4,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.db = db
        self.fundamentals_collector = FundamentalsCollector(timeout=timeout, max_retries=max_retries)
        self.news_collector = NewsCollector(timeout=timeout, max_retries=max_retries)
        self.market_context_collector = MarketContextCollector(timeout=timeout, max_retries=max_retries)

        self.fundamentals_cache = DataCache[Fundamentals](
            db, "fundamentals_cache", fundamentals_ttl_hours, deserializer=_fundamentals_from_dict
        )
        self.news_cache = DataCache[NewsCollection](
            db, "news_cache", news_ttl_hours, deserializer=_news_from_dict
        )
        self.market_context_cache = DataCache[MarketContext](
            db, "market_context_cache", market_context_ttl_hours, deserializer=_market_context_from_dict
        )

    def _fundamentals_key(self, symbol: str) -> str:
        return f"fundamentals:{symbol}"

    def _news_key(self, symbol: str) -> str:
        return f"news:{symbol}"

    def _market_context_key(self, symbol: str) -> str:
        return f"market_context:{symbol}"

    def get_fundamentals(self, symbol: str, use_cache: bool = True) -> Fundamentals:
        cache_key = self._fundamentals_key(symbol)

        if use_cache:
            cached = self.fundamentals_cache.get(cache_key)
            if cached:
                logger.debug("fundamentals_cache_hit", symbol=symbol)
                return cached

        logger.info("fundamentals_fetching", symbol=symbol)
        fundamentals = self.fundamentals_collector.collect(symbol)

        if use_cache:
            self.fundamentals_cache.set(
                cache_key, symbol, fundamentals,
                source=fundamentals.source, fetched_at=fundamentals.fetched_at
            )

        return fundamentals

    async def get_news(self, symbol: str, use_cache: bool = True) -> NewsCollection:
        cache_key = self._news_key(symbol)

        if use_cache:
            cached = self.news_cache.get(cache_key)
            if cached:
                logger.debug("news_cache_hit", symbol=symbol)
                return cached

        logger.info("news_fetching", symbol=symbol)
        news = await self.news_collector.collect(symbol)

        if use_cache:
            self.news_cache.set(
                cache_key, symbol, news,
                source=news.source, fetched_at=news.fetched_at
            )

        return news

    def get_market_context(self, symbol: str, sector: Optional[str] = None, use_cache: bool = True) -> MarketContext:
        cache_key = self._market_context_key(symbol)

        if use_cache:
            cached = self.market_context_cache.get(cache_key)
            if cached:
                logger.debug("market_context_cache_hit", symbol=symbol)
                return cached

        logger.info("market_context_fetching", symbol=symbol)
        market_context = self.market_context_collector.collect(symbol, sector)

        if use_cache:
            self.market_context_cache.set(
                cache_key, symbol, market_context,
                source=market_context.source, fetched_at=market_context.fetched_at
            )

        return market_context

    def get_complete(self, symbol: str, sector: Optional[str] = None, use_cache: bool = True) -> CompleteData:
        fundamentals = self.get_fundamentals(symbol, use_cache)
        sector = sector or fundamentals.sector
        market_context = self.get_market_context(symbol, sector, use_cache)

        return CompleteData(
            symbol=symbol,
            fundamentals=fundamentals,
            market_context=market_context,
        )

    async def get_complete_async(self, symbol: str, sector: Optional[str] = None, use_cache: bool = True) -> CompleteData:
        fundamentals = self.get_fundamentals(symbol, use_cache)
        sector = sector or fundamentals.sector
        market_context = self.get_market_context(symbol, sector, use_cache)
        news = await self.get_news(symbol, use_cache)

        return CompleteData(
            symbol=symbol,
            fundamentals=fundamentals,
            news=news,
            market_context=market_context,
        )

    def clear_symbol_cache(self, symbol: str) -> dict[str, int]:
        return {
            "fundamentals": self.fundamentals_cache.clear_symbol(symbol),
            "news": self.news_cache.clear_symbol(symbol),
            "market_context": self.market_context_cache.clear_symbol(symbol),
        }

    def clear_all_expired(self) -> dict[str, int]:
        return {
            "fundamentals": self.fundamentals_cache.clear_expired(),
            "news": self.news_cache.clear_expired(),
            "market_context": self.market_context_cache.clear_expired(),
        }

    def get_cache_stats(self) -> dict[str, dict]:
        return {
            "fundamentals": self.fundamentals_cache.get_stats(),
            "news": self.news_cache.get_stats(),
            "market_context": self.market_context_cache.get_stats(),
        }

    async def close(self) -> None:
        await self.news_collector.close()
