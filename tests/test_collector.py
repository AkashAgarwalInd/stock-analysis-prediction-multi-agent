from datetime import date, datetime
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest

from stock_analysis.market.collector import MarketPriceCollector, PriceData, PriceHistory
from stock_analysis.market.cache import PriceCache


class TestPriceData:
    def test_price_data_creation(self):
        pd_obj = PriceData(
            symbol="RELIANCE",
            date=date(2024, 1, 15),
            open=Decimal("2500.00"),
            high=Decimal("2550.00"),
            low=Decimal("2480.00"),
            close=Decimal("2530.00"),
            adj_close=Decimal("2530.00"),
            volume=1000000,
            source="yfinance",
            fetched_at=datetime(2024, 1, 15, 10, 0, 0),
        )
        assert pd_obj.symbol == "RELIANCE"
        assert pd_obj.close == Decimal("2530.00")

    def test_price_data_to_dict(self):
        pd_obj = PriceData(
            symbol="RELIANCE",
            date=date(2024, 1, 15),
            close=Decimal("2530.00"),
            volume=1000000,
        )
        d = pd_obj.to_dict()
        assert d["symbol"] == "RELIANCE"
        assert d["close"] == "2530.00"
        assert d["volume"] == 1000000


class TestPriceHistory:
    def test_price_history_creation(self):
        pd_obj = PriceData(
            symbol="RELIANCE",
            date=date(2024, 1, 15),
            close=Decimal("2530.00"),
        )
        history = PriceHistory(
            symbol="RELIANCE",
            data=[pd_obj],
            source="yfinance",
        )
        assert history.symbol == "RELIANCE"
        assert len(history.data) == 1

    def test_price_history_to_dataframe(self):
        pd_obj1 = PriceData(
            symbol="RELIANCE",
            date=date(2024, 1, 15),
            close=Decimal("2530.00"),
            volume=1000000,
        )
        pd_obj2 = PriceData(
            symbol="RELIANCE",
            date=date(2024, 1, 16),
            close=Decimal("2540.00"),
            volume=1100000,
        )
        history = PriceHistory(
            symbol="RELIANCE",
            data=[pd_obj1, pd_obj2],
        )
        df = history.to_dataframe()
        assert len(df) == 2
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_price_history_get_close_series(self):
        pd_obj1 = PriceData(symbol="RELIANCE", date=date(2024, 1, 15), close=Decimal("2530.00"))
        pd_obj2 = PriceData(symbol="RELIANCE", date=date(2024, 1, 16), close=Decimal("2540.00"))
        history = PriceHistory(symbol="RELIANCE", data=[pd_obj1, pd_obj2])
        series = history.get_close_series()
        assert len(series) == 2
        assert series.iloc[0] == 2530.0


class TestMarketPriceCollector:
    def setup_method(self):
        self.collector = MarketPriceCollector(timeout=5, max_retries=0)

    @patch("stock_analysis.market.collector.yf.Ticker")
    def test_fetch_history_success(self, mock_ticker_class):
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        mock_hist = pd.DataFrame({
            "Open": [2500.0, 2510.0],
            "High": [2550.0, 2560.0],
            "Low": [2480.0, 2490.0],
            "Close": [2530.0, 2540.0],
            "Adj Close": [2530.0, 2540.0],
            "Volume": [1000000, 1100000],
        }, index=pd.to_datetime(["2024-01-15", "2024-01-16"]))

        mock_ticker.history.return_value = mock_hist

        history = self.collector.fetch_history("RELIANCE.NS", period="5d")

        assert len(history.data) == 2
        assert history.data[0].close == Decimal("2530.0")
        assert history.data[1].close == Decimal("2540.0")
        assert history.source == "yfinance"
        mock_ticker.history.assert_called_once()

    @patch("stock_analysis.market.collector.yf.Ticker")
    def test_fetch_history_empty(self, mock_ticker_class):
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.return_value = pd.DataFrame()

        history = self.collector.fetch_history("INVALID.NS", period="5d")

        assert len(history.data) == 0
        assert history.symbol == "INVALID.NS"

    @patch("stock_analysis.market.collector.yf.Ticker")
    def test_fetch_history_with_start_end(self, mock_ticker_class):
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        mock_hist = pd.DataFrame({
            "Close": [2530.0],
        }, index=pd.to_datetime(["2024-01-15"]))

        mock_ticker.history.return_value = mock_hist

        history = self.collector.fetch_history(
            "RELIANCE.NS",
            start=date(2024, 1, 15),
            end=date(2024, 1, 16)
        )

        mock_ticker.history.assert_called_once()
        call_args = mock_ticker.history.call_args
        assert call_args[1]["start"] == "2024-01-15"
        assert call_args[1]["end"] == "2024-01-16"

    @patch("stock_analysis.market.collector.yf.Ticker")
    def test_fetch_history_retry_on_exception(self, mock_ticker_class):
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.side_effect = [
            Exception("Network error"),
            pd.DataFrame({"Close": [2530.0]}, index=pd.to_datetime(["2024-01-15"])),
        ]

        collector = MarketPriceCollector(max_retries=1, base_backoff=0.01)
        history = collector.fetch_history("RELIANCE.NS", period="5d")

        assert len(history.data) == 1
        assert mock_ticker.history.call_count == 2

    @patch("stock_analysis.market.collector.yf.Ticker")
    def test_fetch_latest(self, mock_ticker_class):
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        mock_hist = pd.DataFrame({
            "Close": [2530.0, 2540.0],
        }, index=pd.to_datetime(["2024-01-15", "2024-01-16"]))

        mock_ticker.history.return_value = mock_hist

        latest = self.collector.fetch_latest("RELIANCE.NS")

        assert latest is not None
        assert latest.close == Decimal("2540.0")

    @patch("stock_analysis.market.collector.yf.Ticker")
    def test_fetch_latest_empty(self, mock_ticker_class):
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.return_value = pd.DataFrame()

        latest = self.collector.fetch_latest("RELIANCE.NS")

        assert latest is None

    def test_fetch_multiple(self):
        with patch.object(self.collector, "fetch_history") as mock_fetch:
            mock_fetch.return_value = PriceHistory(symbol="RELIANCE.NS", data=[])

            results = self.collector.fetch_multiple(["RELIANCE.NS", "TCS.NS"])

            assert len(results) == 2
            assert mock_fetch.call_count == 2


class TestMarketPriceCollectorWithCache:
    def test_cache_hit(self):
        mock_cache = MagicMock(spec=PriceCache)
        mock_cache.get.return_value = PriceHistory(
            symbol="RELIANCE.NS",
            data=[PriceData(symbol="RELIANCE.NS", date=date(2024, 1, 15), close=Decimal("2530.0"))],
            source="cache"
        )

        collector = MarketPriceCollector(cache=mock_cache, max_retries=0)

        with patch("stock_analysis.market.collector.yf.Ticker") as mock_ticker:
            history = collector.fetch_history("RELIANCE.NS", period="5d")

            assert history.source == "cache"
            mock_ticker.assert_not_called()
            mock_cache.get.assert_called_once()

    def test_cache_miss_then_store(self):
        mock_cache = MagicMock(spec=PriceCache)
        mock_cache.get.return_value = None

        collector = MarketPriceCollector(cache=mock_cache, max_retries=0)

        with patch("stock_analysis.market.collector.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker_class.return_value = mock_ticker
            mock_ticker.history.return_value = pd.DataFrame({
                "Close": [2530.0],
            }, index=pd.to_datetime(["2024-01-15"]))

            history = collector.fetch_history("RELIANCE.NS", period="5d")

            assert history.source == "yfinance"
            mock_cache.set.assert_called_once()