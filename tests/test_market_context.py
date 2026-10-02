from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from stock_analysis.data.market_context import MarketContext, MarketContextCollector


class TestMarketContext:
    def test_market_context_creation(self):
        ctx = MarketContext(
            symbol="RELIANCE",
            nifty_50=22000.0,
            nifty_50_change=100.0,
            nifty_50_change_pct=0.45,
            india_vix=15.5,
            usd_inr=83.0,
        )
        assert ctx.symbol == "RELIANCE"
        assert ctx.nifty_50 == 22000.0
        assert ctx.india_vix == 15.5
        assert ctx.usd_inr == 83.0

    def test_market_context_optional_fields_none(self):
        ctx = MarketContext(symbol="TEST")
        assert ctx.nifty_50 is None
        assert ctx.bank_nifty is None
        assert ctx.india_vix is None
        assert ctx.sector_index is None
        assert ctx.beta is None


class TestMarketContextCollector:
    @pytest.fixture
    def collector(self):
        return MarketContextCollector(timeout=10, max_retries=1)

    @patch("stock_analysis.data.market_context.yf.Ticker")
    def test_collect_success(self, mock_ticker, collector):
        def mock_info(symbol):
            if symbol == "^NSEI":
                return {"symbol": "^NSEI", "regularMarketPrice": 22000.0, "regularMarketPreviousClose": 21900.0}
            elif symbol == "^NSEBANK":
                return {"symbol": "^NSEBANK", "regularMarketPrice": 48000.0, "regularMarketPreviousClose": 47800.0}
            elif symbol == "^INDIAVIX":
                return {"symbol": "^INDIAVIX", "regularMarketPrice": 15.5, "regularMarketPreviousClose": 15.0}
            elif symbol == "INR=X":
                return {"symbol": "INR=X", "regularMarketPrice": 83.0, "regularMarketPreviousClose": 82.9}
            elif symbol == "NIFTY_FIN_SERVICE.NS":
                return {"symbol": "NIFTY_FIN_SERVICE.NS", "regularMarketPrice": 20000.0, "regularMarketPreviousClose": 19900.0}
            elif symbol == "RELIANCE.NS":
                return {"symbol": "RELIANCE.NS", "beta": 1.2}
            return None

        def mock_history(symbol, period):
            import numpy as np
            import pandas as pd
            dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
            base = 100 if "NSEI" not in symbol else 22000
            prices = base * (1 + np.cumsum(np.random.randn(100) * 0.01))
            df = pd.DataFrame({"Close": prices}, index=dates)
            return df

        mock_ticker_instance = Mock()
        mock_ticker_instance.info.side_effect = lambda: mock_info(mock_ticker_instance._symbol) if hasattr(mock_ticker_instance, '_symbol') else {}
        mock_ticker_instance.history.side_effect = lambda period: mock_history(mock_ticker_instance._symbol, period) if hasattr(mock_ticker_instance, '_symbol') else None

        def create_ticker(symbol):
            instance = Mock()
            instance._symbol = symbol
            instance.info = mock_info(symbol)
            instance.history = Mock(side_effect=lambda period: mock_history(symbol, period))
            return instance

        mock_ticker.side_effect = create_ticker

        result = collector.collect("RELIANCE", sector="Financial Services")

        assert isinstance(result, MarketContext)
        assert result.symbol == "RELIANCE"
        assert result.nifty_50 == 22000.0
        assert result.nifty_50_change == 100.0
        assert result.nifty_50_change_pct == pytest.approx(0.4566, rel=0.01)
        assert result.bank_nifty == 48000.0
        assert result.india_vix == 15.5
        assert result.usd_inr == 83.0
        assert result.sector_name == "Financial Services"
        assert result.sector_index == 20000.0
        assert result.beta == 1.2

    @patch("stock_analysis.data.market_context.yf.Ticker")
    def test_collect_missing_indices(self, mock_ticker, collector):
        def create_ticker(symbol):
            instance = Mock()
            instance.info = {}
            instance.history = Mock(return_value=None)
            return instance

        mock_ticker.side_effect = create_ticker

        result = collector.collect("RELIANCE")

        assert isinstance(result, MarketContext)
        assert result.symbol == "RELIANCE"
        assert result.nifty_50 is None
        assert result.bank_nifty is None
        assert result.india_vix is None
        assert result.usd_inr is None

    @patch("stock_analysis.data.market_context.yf.Ticker")
    def test_collect_sector_not_mapped(self, mock_ticker, collector):
        mock_nifty_info = {"symbol": "^NSEI", "regularMarketPrice": 22000.0, "regularMarketPreviousClose": 21900.0}
        mock_stock_info = {"symbol": "RELIANCE.NS", "beta": 1.2}

        def create_ticker(symbol):
            instance = Mock()
            if symbol == "^NSEI":
                instance.info = mock_nifty_info
            elif symbol == "RELIANCE.NS":
                instance.info = mock_stock_info
            else:
                instance.info = {}
            instance.history = Mock(return_value=None)
            return instance

        mock_ticker.side_effect = create_ticker

        result = collector.collect("RELIANCE", sector="Unknown Sector")

        assert result.sector_name == "Unknown Sector"
        assert result.sector_index is None

    @patch("stock_analysis.data.market_context.yf.Ticker")
    def test_collect_exception_handling(self, mock_ticker, collector):
        mock_ticker.side_effect = Exception("Network error")

        result = collector.collect("RELIANCE")

        assert isinstance(result, MarketContext)
        assert result.symbol == "RELIANCE"

    def test_collect_batch(self, collector):
        with patch.object(collector, "collect") as mock_collect:
            mock_collect.side_effect = [
                MarketContext(symbol="RELIANCE", nifty_50=22000.0),
                MarketContext(symbol="TCS", nifty_50=22000.0),
            ]
            results = collector.collect_batch(["RELIANCE", "TCS"], {"RELIANCE": "Energy", "TCS": "IT"})

            assert len(results) == 2
            assert results["RELIANCE"].nifty_50 == 22000.0
            assert results["TCS"].nifty_50 == 22000.0

    def test_calculate_change(self, collector):
        change, change_pct = collector._calculate_change(110.0, 100.0)
        assert change == 10.0
        assert change_pct == 10.0

        change, change_pct = collector._calculate_change(None, 100.0)
        assert change is None
        assert change_pct is None

        change, change_pct = collector._calculate_change(100.0, 0)
        assert change is None
        assert change_pct is None
