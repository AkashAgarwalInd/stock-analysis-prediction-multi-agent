import asyncio
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Optional

import feedparser
import httpx

from stock_analysis.logging import get_logger
from stock_analysis.reliability import RetryPolicy, get_rate_limiter

logger = get_logger(__name__)


@dataclass
class NewsItem:
    title: str
    url: str
    source: str
    published_at: datetime
    summary: Optional[str] = None
    symbol: Optional[str] = None
    fetched_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class NewsCollection:
    symbol: str
    items: list[NewsItem]
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    source: str = "google_news_rss"


class NewsCollector:
    GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        lookback_days: int = 14,
        cache_ttl_hours: int = 6,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.lookback_days = lookback_days
        self.cache_ttl_hours = cache_ttl_hours
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, follow_redirects=True)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _build_query(self, symbol: str) -> str:
        base = symbol.replace(".NS", "").replace(".BO", "")
        return f"{base} stock OR {base} share OR {base} equity NSE India"

    def _parse_rss_date(self, date_str: str) -> Optional[datetime]:
        for fmt in (
            "%a, %d %b %Y %H:%M:%S %Z",
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

    def _is_within_lookback(self, published_at: Optional[datetime]) -> bool:
        """True if ``published_at`` falls inside the lookback window.

        An undated article is excluded rather than assumed recent: callers must
        not treat an unknown date as fresh.
        """
        if published_at is None:
            return False
        cutoff = datetime.utcnow() - timedelta(days=self.lookback_days)
        if published_at.tzinfo is not None:
            # Convert to UTC before dropping tzinfo. A bare replace() would read
            # 15:00+05:30 as 15:00 UTC -- a 5.5h error at the window boundary.
            published_at = published_at.astimezone(UTC).replace(tzinfo=None)
        return published_at >= cutoff

    def _clean_summary(self, summary: str) -> str:
        clean = re.sub(r"<[^>]+>", "", summary)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean[:500] if len(clean) > 500 else clean

    def _deduplicate(self, items: list[NewsItem]) -> list[NewsItem]:
        seen = set()
        unique = []
        for item in items:
            key = (item.title.lower().strip(), item.url)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    async def collect(self, symbol: str) -> NewsCollection:
        query = self._build_query(symbol)
        params = {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}

        items: list[NewsItem] = []

        backoff = RetryPolicy.for_external_data()
        for attempt in range(self.max_retries):
            if attempt:
                await asyncio.sleep(backoff.delay(attempt - 1))
            try:
                await get_rate_limiter("news").acquire_async()
                client = self._get_client()
                response = await client.get(self.GOOGLE_NEWS_RSS_URL, params=params)
                response.raise_for_status()

                feed = feedparser.parse(response.text)
                undated = 0

                for entry in feed.entries:
                    published = self._parse_rss_date(entry.get("published", ""))
                    if published is None:
                        # Logged below: a feed-format change would otherwise
                        # silently drop every article.
                        undated += 1
                        continue
                    if not self._is_within_lookback(published):
                        continue

                    title = entry.get("title", "").strip()
                    url = entry.get("link", "").strip()
                    source = entry.get("source", {}).get("title", "Google News")
                    summary = self._clean_summary(entry.get("summary", ""))

                    if title and url:
                        items.append(
                            NewsItem(
                                title=title,
                                url=url,
                                source=source,
                                published_at=published,
                                summary=summary,
                                symbol=symbol,
                            )
                        )

                items = self._deduplicate(items)
                if undated:
                    logger.warning(
                        "news_entries_unparseable_date", symbol=symbol, skipped=undated
                    )
                logger.info("news_collected", symbol=symbol, count=len(items))
                return NewsCollection(symbol=symbol, items=items)

            except httpx.TimeoutException:
                logger.warning("news_collection_timeout", symbol=symbol, attempt=attempt + 1)
            except httpx.HTTPStatusError as e:
                logger.warning("news_collection_http_error", symbol=symbol, status=e.response.status_code, attempt=attempt + 1)
            except Exception as e:
                logger.warning("news_collection_failed", symbol=symbol, attempt=attempt + 1, error=str(e))

            if attempt == self.max_retries - 1:
                logger.error("news_collection_exhausted", symbol=symbol)

        return NewsCollection(symbol=symbol, items=[])

    async def collect_batch(self, symbols: list[str]) -> dict[str, NewsCollection]:
        results = {}
        for symbol in symbols:
            results[symbol] = await self.collect(symbol)
        return results
