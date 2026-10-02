from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from stock_analysis.data.news import NewsCollection, NewsCollector, NewsItem


class MockFeedParserDict(dict):
    def __getattr__(self, name):
        return self.get(name)


class TestNewsItem:
    def test_news_item_creation(self):
        item = NewsItem(
            title="Test News",
            url="https://example.com/news",
            source="Test Source",
            published_at=datetime.utcnow(),
            summary="Test summary",
            symbol="RELIANCE",
        )
        assert item.title == "Test News"
        assert item.url == "https://example.com/news"
        assert item.source == "Test Source"
        assert item.symbol == "RELIANCE"

    def test_news_item_minimal(self):
        item = NewsItem(
            title="Test",
            url="https://example.com",
            source="Source",
            published_at=datetime.utcnow(),
        )
        assert item.summary is None
        assert item.symbol is None


class TestNewsCollection:
    def test_news_collection_creation(self):
        items = [
            NewsItem(title="News 1", url="https://ex.com/1", source="S1", published_at=datetime.utcnow()),
            NewsItem(title="News 2", url="https://ex.com/2", source="S2", published_at=datetime.utcnow()),
        ]
        collection = NewsCollection(symbol="RELIANCE", items=items)
        assert collection.symbol == "RELIANCE"
        assert len(collection.items) == 2


class TestNewsCollector:
    @pytest.fixture
    def collector(self):
        return NewsCollector(timeout=10, max_retries=1, lookback_days=14)

    @pytest.fixture
    def mock_rss_feed(self):
        recent_date = (datetime.utcnow() - timedelta(days=2)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        return MockFeedParserDict({
            "entries": [
                MockFeedParserDict({
                    "title": "Reliance stock rises on strong earnings",
                    "link": "https://example.com/news1",
                    "published": recent_date,
                    "source": {"title": "Economic Times"},
                    "summary": "Reliance Industries reported strong quarterly earnings...",
                }),
                MockFeedParserDict({
                    "title": "Market update: Nifty hits new high",
                    "link": "https://example.com/news2",
                    "published": recent_date,
                    "source": {"title": "Moneycontrol"},
                    "summary": "Nifty 50 reached a new all-time high today...",
                }),
            ]
        })

    @pytest.mark.asyncio
    async def test_collect_success(self, collector, mock_rss_feed):
        with patch.object(collector, "_get_client") as mock_client:
            mock_response = Mock()
            mock_response.text = "rss xml content"
            mock_response.raise_for_status = Mock()
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            with patch("stock_analysis.data.news.feedparser.parse") as mock_parse:
                mock_parse.return_value = mock_rss_feed

                result = await collector.collect("RELIANCE")

                assert isinstance(result, NewsCollection)
                assert result.symbol == "RELIANCE"
                assert len(result.items) == 2
                assert result.items[0].title == "Reliance stock rises on strong earnings"
                assert result.items[0].source == "Economic Times"

    @pytest.mark.asyncio
    async def test_collect_empty_feed(self, collector):
        with patch.object(collector, "_get_client") as mock_client:
            mock_response = Mock()
            mock_response.text = "rss xml content"
            mock_response.raise_for_status = Mock()
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            with patch("stock_analysis.data.news.feedparser.parse") as mock_parse:
                mock_parse.return_value = MockFeedParserDict({"entries": []})

                result = await collector.collect("RELIANCE")

                assert isinstance(result, NewsCollection)
                assert len(result.items) == 0

    @pytest.mark.asyncio
    async def test_collect_timeout(self, collector):
        import httpx
        with patch.object(collector, "_get_client") as mock_client:
            mock_client.return_value.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

            result = await collector.collect("RELIANCE")

            assert isinstance(result, NewsCollection)
            assert len(result.items) == 0

    @pytest.mark.asyncio
    async def test_collect_http_error(self, collector):
        import httpx
        with patch.object(collector, "_get_client") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_client.return_value.get = AsyncMock(
                side_effect=httpx.HTTPStatusError("Not Found", request=Mock(), response=mock_response)
            )

            result = await collector.collect("RELIANCE")

            assert isinstance(result, NewsCollection)
            assert len(result.items) == 0

    @pytest.mark.asyncio
    async def test_collect_old_news_filtered(self, collector):
        old_date = (datetime.utcnow() - timedelta(days=30)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        recent_date = (datetime.utcnow() - timedelta(days=2)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        mock_rss_feed = MockFeedParserDict({
            "entries": [
                MockFeedParserDict({
                    "title": "Old News",
                    "link": "https://example.com/old",
                    "published": old_date,
                    "source": {"title": "Old Source"},
                    "summary": "Old news summary",
                }),
                MockFeedParserDict({
                    "title": "Recent News",
                    "link": "https://example.com/recent",
                    "published": recent_date,
                    "source": {"title": "Recent Source"},
                    "summary": "Recent news summary",
                })
            ]
        })

        with patch.object(collector, "_get_client") as mock_client:
            mock_response = Mock()
            mock_response.text = "rss xml content"
            mock_response.raise_for_status = Mock()
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            with patch("stock_analysis.data.news.feedparser.parse") as mock_parse:
                mock_parse.return_value = mock_rss_feed

                result = await collector.collect("RELIANCE")

                assert len(result.items) == 1
                assert result.items[0].title == "Recent News"

    @pytest.mark.asyncio
    async def test_collect_deduplication(self, collector):
        recent_date = (datetime.utcnow() - timedelta(days=2)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        mock_rss_feed = MockFeedParserDict({
            "entries": [
                MockFeedParserDict({
                    "title": "Same News",
                    "link": "https://example.com/same",
                    "published": recent_date,
                    "source": {"title": "Source 1"},
                    "summary": "Summary 1",
                }),
                MockFeedParserDict({
                    "title": "Same News",
                    "link": "https://example.com/same",
                    "published": recent_date,
                    "source": {"title": "Source 2"},
                    "summary": "Summary 2",
                }),
            ]
        })

        with patch.object(collector, "_get_client") as mock_client:
            mock_response = Mock()
            mock_response.text = "rss xml content"
            mock_response.raise_for_status = Mock()
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            with patch("stock_analysis.data.news.feedparser.parse") as mock_parse:
                mock_parse.return_value = mock_rss_feed

                result = await collector.collect("RELIANCE")

                assert len(result.items) == 1

    @pytest.mark.asyncio
    async def test_collect_html_stripped_from_summary(self, collector):
        recent_date = (datetime.utcnow() - timedelta(days=2)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        mock_rss_feed = MockFeedParserDict({
            "entries": [
                MockFeedParserDict({
                    "title": "News with HTML",
                    "link": "https://example.com/html",
                    "published": recent_date,
                    "source": {"title": "Source"},
                    "summary": "<p>This has <b>HTML</b> tags</p> and &nbsp; entities",
                })
            ]
        })

        with patch.object(collector, "_get_client") as mock_client:
            mock_response = Mock()
            mock_response.text = "rss xml content"
            mock_response.raise_for_status = Mock()
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            with patch("stock_analysis.data.news.feedparser.parse") as mock_parse:
                mock_parse.return_value = mock_rss_feed

                result = await collector.collect("RELIANCE")

                assert "<p>" not in result.items[0].summary
                assert "<b>" not in result.items[0].summary
                assert "This has HTML tags" in result.items[0].summary

    @pytest.mark.asyncio
    async def test_collect_summary_truncated(self, collector):
        long_summary = "A" * 600
        recent_date = (datetime.utcnow() - timedelta(days=2)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        mock_rss_feed = MockFeedParserDict({
            "entries": [
                MockFeedParserDict({
                    "title": "Long Summary",
                    "link": "https://example.com/long",
                    "published": recent_date,
                    "source": {"title": "Source"},
                    "summary": long_summary,
                })
            ]
        })

        with patch.object(collector, "_get_client") as mock_client:
            mock_response = Mock()
            mock_response.text = "rss xml content"
            mock_response.raise_for_status = Mock()
            mock_client.return_value.get = AsyncMock(return_value=mock_response)

            with patch("stock_analysis.data.news.feedparser.parse") as mock_parse:
                mock_parse.return_value = mock_rss_feed

                result = await collector.collect("RELIANCE")

                assert len(result.items[0].summary) <= 500

    @pytest.mark.asyncio
    async def test_collect_batch(self, collector):
        with patch.object(collector, "collect") as mock_collect:
            mock_collect.side_effect = [
                NewsCollection(symbol="RELIANCE", items=[NewsItem(title="N1", url="u1", source="S1", published_at=datetime.utcnow())]),
                NewsCollection(symbol="TCS", items=[NewsItem(title="N2", url="u2", source="S2", published_at=datetime.utcnow())]),
            ]
            results = await collector.collect_batch(["RELIANCE", "TCS"])

            assert len(results) == 2
            assert results["RELIANCE"].items[0].title == "N1"
            assert results["TCS"].items[0].title == "N2"

    @pytest.mark.asyncio
    async def test_close(self, collector):
        await collector.close()
        assert collector._client is None
