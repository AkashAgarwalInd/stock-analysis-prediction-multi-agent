import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from stock_analysis.data.fundamentals import Fundamentals, FundamentalsCollector
from stock_analysis.data.news import NewsItem, NewsCollection, NewsCollector
from stock_analysis.data.market_context import MarketContext, MarketContextCollector
from stock_analysis.data.cache import DataCache
from stock_analysis.data.collector import CompleteData, DataCollector
from stock_analysis.database import Database


@pytest.fixture
def temp_db():
    db = Database(":memory:")
    yield db
    db.close()


class TestFundamentals:
    def test_fundamentals_creation(self):
        f = Fundamentals(symbol="RELIANCE", pe_ratio=25.5, sector="Energy")
        assert f.symbol == "RELIANCE"
        assert f.pe_ratio == 25.5
        assert f.sector == "Energy"
        assert f.fetched_at is not None

    def test_fundamentals_optional_fields_none_by_default(self):
        f = Fundamentals(symbol="TEST")
        assert f.pe_ratio is None
        assert f.pb_ratio is None
        assert f.roe is None
        assert f.market_cap is None


class TestFundamentalsCollector:
    @pytest.fixture
    def collector(self):
        return FundamentalsCollector(timeout=10, max_retries=1)

    @patch("stock_analysis.data.fundamentals.yf.Ticker")
    def test_collect_success(self, mock_ticker, collector):
        mock_info = {
            "symbol": "RELIANCE.NS",
            "shortName": "Reliance Industries Ltd",
            "trailingPE": 25.5,
            "priceToBook": 2.1,
            "returnOnEquity": 0.15,
            "debtToEquity": 0.5,
            "profitMargins": 0.12,
            "operatingMargins": 0.18,
            "revenueGrowth": 0.10,
            "earningsGrowth": 0.15,
            "dividendYield": 0.003,
            "marketCap": 1500000000000,
            "sector": "Energy",
            "industry": "Oil & Gas",
            "earningsDate": 1704067200,
            "beta": 1.2,
            "forwardPE": 22.0,
            "priceToSalesTrailing12Months": 1.5,
            "enterpriseValue": 1600000000000,
            "returnOnAssets": 0.08,
            "freeCashflow": 50000000000,
            "operatingCashflow": 80000000000,
            "totalDebt": 200000000000,
            "totalCash": 50000000000,
            "currentRatio": 1.2,
            "quickRatio": 0.9,
        }
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = mock_info
        mock_ticker.return_value = mock_ticker_instance

        result = collector.collect("RELIANCE")

        assert isinstance(result, Fundamentals)
        assert result.symbol == "RELIANCE"
        assert result.pe_ratio == 25.5
        assert result.pb_ratio == 2.1
        assert result.roe == 0.15
        assert result.sector == "Energy"
        assert result.market_cap == 1500000000000

    @patch("stock_analysis.data.fundamentals.yf.Ticker")
    def test_collect_empty_response(self, mock_ticker, collector):
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {}
        mock_ticker.return_value = mock_ticker_instance

        result = collector.collect("INVALID")

        assert isinstance(result, Fundamentals)
        assert result.symbol == "INVALID"
        assert result.pe_ratio is None

    @patch("stock_analysis.data.fundamentals.yf.Ticker")
    def test_collect_exception_retries(self, mock_ticker, collector):
        mock_ticker.side_effect = Exception("Network error")

        result = collector.collect("RELIANCE")

        assert isinstance(result, Fundamentals)
        assert result.symbol == "RELIANCE"
        assert result.pe_ratio is None
        assert mock_ticker.call_count == collector.max_retries

    @patch("stock_analysis.data.fundamentals.yf.Ticker")
    def test_collect_partial_data(self, mock_ticker, collector):
        mock_info = {
            "symbol": "RELIANCE.NS",
            "trailingPE": 25.5,
            "marketCap": 1500000000000,
        }
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = mock_info
        mock_ticker.return_value = mock_ticker_instance

        result = collector.collect("RELIANCE")

        assert result.pe_ratio == 25.5
        assert result.market_cap == 1500000000000
        assert result.pb_ratio is None
        assert result.sector is None

    @patch("stock_analysis.data.fundamentals.yf.Ticker")
    def test_collect_invalid_types_handled(self, mock_ticker, collector):
        mock_info = {
            "symbol": "RELIANCE.NS",
            "trailingPE": "invalid",
            "marketCap": None,
            "sector": 123,
        }
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = mock_info
        mock_ticker.return_value = mock_ticker_instance

        result = collector.collect("RELIANCE")

        assert result.pe_ratio is None
        assert result.market_cap is None
        assert result.sector == "123"

    def test_collect_batch(self, collector):
        with patch.object(collector, "collect") as mock_collect:
            mock_collect.side_effect = [
                Fundamentals(symbol="RELIANCE", pe_ratio=25.0),
                Fundamentals(symbol="TCS", pe_ratio=30.0),
            ]
            results = collector.collect_batch(["RELIANCE", "TCS"])

            assert len(results) == 2
            assert results["RELIANCE"].pe_ratio == 25.0
            assert results["TCS"].pe_ratio == 30.0