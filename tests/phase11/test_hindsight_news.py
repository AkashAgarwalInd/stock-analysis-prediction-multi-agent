"""RSS-backed hindsight news: only items inside the forecast window, safely bounded."""

from datetime import UTC, datetime, timedelta

from stock_analysis.data.news import NewsCollection, NewsCollector, NewsItem
from stock_analysis.learning import RssHindsightNewsSource

START = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)
END = START + timedelta(days=7)


def _item(title, published):
    return NewsItem(title=title, url=f"https://x/{title}", source="Wire", published_at=published)


def test_filters_to_the_window_and_closes_the_client(monkeypatch):
    closed = []

    async def collect(self, symbol):
        return NewsCollection(
            symbol=symbol,
            items=[
                _item("before", START - timedelta(hours=1)),
                _item("naive-utc", (START + timedelta(days=1)).replace(tzinfo=None)),
                _item("aware", START + timedelta(days=2)),
                _item("after", END + timedelta(minutes=1)),
                _item("x" * 400, START + timedelta(days=3)),
            ],
        )

    async def close(self):
        closed.append(True)

    monkeypatch.setattr(NewsCollector, "collect", collect)
    monkeypatch.setattr(NewsCollector, "close", close)

    items = RssHindsightNewsSource().fetch("RELIANCE.NS", START, END)
    assert [i.headline[:9] for i in items] == ["naive-utc", "aware", "xxxxxxxxx"]
    assert all(i.published_at.tzinfo is not None for i in items)
    assert len(items[-1].headline) == 300
    assert closed == [True]


def test_caps_the_number_of_items(monkeypatch):
    async def collect(self, symbol):
        return NewsCollection(
            symbol=symbol, items=[_item(f"n{i}", START + timedelta(hours=i + 1)) for i in range(40)]
        )

    async def close(self):
        return None

    monkeypatch.setattr(NewsCollector, "collect", collect)
    monkeypatch.setattr(NewsCollector, "close", close)
    assert len(RssHindsightNewsSource(max_items=5).fetch("RELIANCE.NS", START, END)) == 5
