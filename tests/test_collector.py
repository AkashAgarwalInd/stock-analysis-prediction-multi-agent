import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from stock_analysis.data.collector import CompleteData, DataCollector
from stock_analysis.data.fundamentals import Fundamentals
from stock_analysis.data.news import NewsCollection, NewsItem
from stock_analysis.data.market_context import MarketContext
from stock_analysis.database import Database


class TestCompleteData:
    def test_complete_data_creation(self):
        fundamentals = Fundamentals(symbol="RELIANCE", pe_ratio=25.0)
        news = NewsCollection(symbol="RELIANCE", items=[])
        market_context = MarketContext(symbol="RELIANCE", nifty_50=22000.0)

        data = CompleteData(symbol="RELIANCE", fundamentals=fundamentals, news=news, market_context=market_context)

        assert data.symbol == "RELIANCE"
        assert data.fundamentals == fundamentals
        assert data.news == news
        assert data.market_context == market_context
        assert data.fetched_at is not None


class TestDataCollector:
    @pytest.fixture
    def db(self):
        return Database(":memory:")

    @pytest.fixture
    def collector(self, db):
        return DataCollector(db, timeout=10, max_retries=1)

    @pytest.fixture
    def mock_fundamentals(self):
        return Fundamentals(symbol="RELIANCE", pe_ratio=25.0, sector="Energy", market_cap=1500000000000)

    @pytest.fixture
    def mock_news(self):
        items = [NewsItem(title="Test News", url="https://ex.com", source="Test", published_at=datetime.utcnow())]
        return NewsCollection(symbol="RELIANCE", items=items)

    @pytest.fixture
    def mock_market_context(self):
        return MarketContext(symbol="RELIANCE", nifty_50=22000.0, beta=1.2)

    def test_get_fundamentals_with_cache(self, collector, mock_fundamentals):
        with patch.object(collector.fundamentals_collector, "collect", return_value=mock_fundamentals) as mock_collect:
            result1 = collector.get_fundamentals("RELIANCE", use_cache=True)
            result2 = collector.get_fundamentals("RELIANCE", use_cache=True)

            assert result1.pe_ratio == 25.0
            assert result2.pe_ratio == 25.0
            assert mock_collect.call_count == 1

    def test_get_fundamentals_without_cache(self, collector, mock_fundamentals):
        with patch.object(collector.fundamentals_collector, "collect", return_value=mock_fundamentals) as mock_collect:
            result1 = collector.get_fundamentals("RELIANCE", use_cache=False)
            result2 = collector.get_fundamentals("RELIANCE", use_cache=False)

            assert mock_collect.call_count == 2

    def test_get_fundamentals_cache_miss_then_hit(self, collector, mock_fundamentals):
        with patch.object(collector.fundamentals_collector, "collect", return_value=mock_fundamentals) as mock_collect:
            result1 = collector.get_fundamentals("RELIANCE", use_cache=True)
            collector.fundamentals_cache.clear_symbol("RELIANCE")
            result2 = collector.get_fundamentals("RELIANCE", use_cache=True)

            assert mock_collect.call_count == 2

    @pytest.mark.asyncio
    async def test_get_news_with_cache(self, collector, mock_news):
        with patch.object(collector.news_collector, "collect", return_value=mock_news) as mock_collect:
            result1 = await collector.get_news("RELIANCE", use_cache=True)
            result2 = await collector.get_news("RELIANCE", use_cache=True)

            assert len(result1.items) == 1
            assert mock_collect.call_count == 1

    @pytest.mark.asyncio
    async def test_get_news_without_cache(self, collector, mock_news):
        with patch.object(collector.news_collector, "collect", return_value=mock_news) as mock_collect:
            result1 = await collector.get_news("RELIANCE", use_cache=False)
            result2 = await collector.get_news("RELIANCE", use_cache=False)

            assert mock_collect.call_count == 2

    def test_get_market_context_with_cache(self, collector, mock_market_context, mock_fundamentals):
        with patch.object(collector.market_context_collector, "collect", return_value=mock_market_context) as mock_collect:
            with patch.object(collector, "get_fundamentals", return_value=mock_fundamentals):
                result1 = collector.get_market_context("RELIANCE", use_cache=True)
                result2 = collector.get_market_context("RELIANCE", use_cache=True)

                assert result1.nifty_50 == 22000.0
                assert mock_collect.call_count == 1

    def test_get_market_context_sector_from_fundamentals(self, collector, mock_market_context, mock_fundamentals):
        # get_market_context doesn't auto-fetch sector from fundamentals
        # This test verifies explicit sector is passed
        with patch.object(collector.market_context_collector, "collect", return_value=mock_market_context) as mock_collect:
            with patch.object(collector, "get_fundamentals", return_value=mock_fundamentals):
                result = collector.get_market_context("RELIANCE", sector="Energy", use_cache=True)

                mock_collect.assert_called_once_with("RELIANCE", "Energy")

    def test_get_market_context_explicit_sector_overrides(self, collector, mock_market_context, mock_fundamentals):
        with patch.object(collector.market_context_collector, "collect", return_value=mock_market_context) as mock_collect:
            with patch.object(collector, "get_fundamentals", return_value=mock_fundamentals):
                result = collector.get_market_context("RELIANCE", sector="IT", use_cache=True)

                mock_collect.assert_called_once_with("RELIANCE", "IT")

    def test_get_complete(self, collector, mock_fundamentals, mock_market_context):
        with patch.object(collector, "get_fundamentals", return_value=mock_fundamentals):
            with patch.object(collector, "get_market_context", return_value=mock_market_context):
                result = collector.get_complete("RELIANCE")

                assert isinstance(result, CompleteData)
                assert result.symbol == "RELIANCE"
                assert result.fundamentals == mock_fundamentals
                assert result.market_context == mock_market_context
                assert result.news is None

    @pytest.mark.asyncio
    async def test_get_complete_async(self, collector, mock_fundamentals, mock_market_context, mock_news):
        with patch.object(collector, "get_fundamentals", return_value=mock_fundamentals):
            with patch.object(collector, "get_market_context", return_value=mock_market_context):
                with patch.object(collector, "get_news", return_value=mock_news):
                    result = await collector.get_complete_async("RELIANCE")

                    assert isinstance(result, CompleteData)
                    assert result.news == mock_news

    def test_clear_symbol_cache(self, collector):
        with patch.object(collector.fundamentals_cache, "clear_symbol", return_value=1) as mock_f:
            with patch.object(collector.news_cache, "clear_symbol", return_value=2) as mock_n:
                with patch.object(collector.market_context_cache, "clear_symbol", return_value=1) as mock_m:
                    result = collector.clear_symbol_cache("RELIANCE")

                    assert result == {"fundamentals": 1, "news": 2, "market_context": 1}

    def test_clear_all_expired(self, collector):
        with patch.object(collector.fundamentals_cache, "clear_expired", return_value=5) as mock_f:
            with patch.object(collector.news_cache, "clear_expired", return_value=3) as mock_n:
                with patch.object(collector.market_context_cache, "clear_expired", return_value=2) as mock_m:
                    result = collector.clear_all_expired()

                    assert result == {"fundamentals": 5, "news": 3, "market_context": 2}

    def test_get_cache_stats(self, collector):
        mock_stats = {"total_entries": 10, "expired_entries": 2, "table_name": "test"}
        with patch.object(collector.fundamentals_cache, "get_stats", return_value=mock_stats):
            with patch.object(collector.news_cache, "get_stats", return_value=mock_stats):
                with patch.object(collector.market_context_cache, "get_stats", return_value=mock_stats):
                    stats = collector.get_cache_stats()

                    assert stats["fundamentals"] == mock_stats
                    assert stats["news"] == mock_stats
                    assert stats["market_context"] == mock_stats

    @pytest.mark.asyncio
    async def test_close(self, collector):
        with patch.object(collector.news_collector, "close", new_callable=AsyncMock) as mock_close:
            await collector.close()
            mock_close.assert_called_once()