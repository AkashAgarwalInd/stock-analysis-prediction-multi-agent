"""Hindsight news for postmortems from the Google News RSS collector.

The feed only covers roughly the last ``lookback_days`` (14 by default), so
reviewing a forecast soon after its target date sees the window's news, while
an old forecast sees little or none (Plan.md §34 historical news limitation).
Only items published inside the requested window are returned.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Optional

from stock_analysis.data.news import NewsCollection, NewsCollector
from stock_analysis.schemas.learning import HindsightNewsItem

MAX_HEADLINE_CHARS = 300
MAX_ITEMS = 15


class RssHindsightNewsSource:
    """``HindsightNewsSource`` backed by ``NewsCollector`` (network access)."""

    def __init__(self, lookback_days: int = 14, max_items: int = MAX_ITEMS):
        self.lookback_days = lookback_days
        self.max_items = max_items

    def fetch(self, symbol: str, start: datetime, end: datetime) -> list[HindsightNewsItem]:
        """Headlines published after ``start`` and at or before ``end``, oldest first."""
        collection = asyncio.run(self._collect(symbol))
        items = []
        for item in collection.items:
            published = _aware(item.published_at)
            if published is None or not start < published <= end or not item.title.strip():
                continue
            items.append(
                HindsightNewsItem(
                    published_at=published,
                    headline=item.title.strip()[:MAX_HEADLINE_CHARS],
                    source=item.source or None,
                )
            )
        items.sort(key=lambda i: i.published_at)
        return items[: self.max_items]

    async def _collect(self, symbol: str) -> NewsCollection:
        # A fresh collector per call: its HTTP client is bound to this event loop
        collector = NewsCollector(lookback_days=self.lookback_days)
        try:
            return await collector.collect(symbol)
        finally:
            await collector.close()


def _aware(value: Optional[datetime]) -> Optional[datetime]:
    """RSS dates parsed with ``%Z`` are naive UTC."""
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value
