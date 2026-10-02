from datetime import date

import numpy as np
import pandas as pd
import pytest

from stock_analysis.indicators.technical import (
    TechnicalIndicators,
    TrendDirection,
    calculate_52week_position,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_drawdown,
    calculate_ema,
    calculate_macd,
    calculate_momentum,
    calculate_realized_volatility,
    calculate_relative_strength,
    calculate_rolling_returns,
    calculate_rsi,
    calculate_sma,
    calculate_volume_trend,
    compute_all_indicators,
)


class TestSMA:
    def test_sma_basic(self):
        series = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        result = calculate_sma(series, 5)
        assert len(result) == len(series)
        assert pd.isna(result.iloc[3])
        assert result.iloc[4] == 12.0
        assert result.iloc[9] == 17.0

    def test_sma_window_larger_than_series(self):
        series = pd.Series([10, 11, 12])
        result = calculate_sma(series, 5)
        assert all(pd.isna(result))


class TestEMA:
    def test_ema_basic(self):
        series = pd.Series([10, 11, 12, 13, 14, 15])
        result = calculate_ema(series, 3)
        assert len(result) == len(series)
        assert not pd.isna(result.iloc[-1])
        assert result.iloc[-1] > 14


class TestRSI:
    def test_rsi_basic(self):
        series = pd.Series([10, 12, 11, 13, 14, 13, 15, 16, 14, 17, 18, 16, 19, 20, 19])
        result = calculate_rsi(series, 5)
        assert len(result) == len(series)
        assert all((result.dropna() >= 0) & (result.dropna() <= 100))

    def test_rsi_oversold(self):
        series = pd.Series([100, 90, 80, 70, 60, 50, 40, 30, 20, 15])
        result = calculate_rsi(series, 5)
        assert result.iloc[-1] < 30

    def test_rsi_overbought(self):
        series = pd.Series([10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        result = calculate_rsi(series, 5)
        assert result.iloc[-1] > 70


class TestMACD:
    def test_macd_basic(self):
        series = pd.Series([10 + i * 0.1 for i in range(50)])
        macd, signal, hist = calculate_macd(series)
        assert len(macd) == len(series)
        assert len(signal) == len(series)
        assert len(hist) == len(series)
        assert not pd.isna(macd.iloc[-1])
        assert not pd.isna(signal.iloc[-1])


class TestATR:
    def test_atr_basic(self):
        high = pd.Series([15, 16, 17, 18, 19, 20])
        low = pd.Series([10, 11, 12, 13, 14, 15])
        close = pd.Series([12, 13, 14, 15, 16, 17])
        result = calculate_atr(high, low, close, 3)
        assert len(result) == len(close)
        assert not pd.isna(result.iloc[-1])
        assert result.iloc[-1] > 0


class TestBollingerBands:
    def test_bollinger_bands_basic(self):
        series = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
        upper, middle, lower, width, percent = calculate_bollinger_bands(series, 10)
        assert len(upper) == len(series)
        assert all(upper.dropna() >= middle.dropna())
        assert all(middle.dropna() >= lower.dropna())
        assert all(width.dropna() >= 0)
        assert all((percent.dropna() >= 0) & (percent.dropna() <= 1))


class TestVolumeTrend:
    def test_volume_trend_up(self):
        # Last 5 values are 2000, current is 2000, but SMA of last 5 is 1000 (from first 5)
        volume = pd.Series([1000] * 5 + [2000] * 5)
        trend, ratio = calculate_volume_trend(volume, 5)
        # At index 9, SMA is mean of indices 5-9 = 2000, current is 2000, ratio = 1.0
        # But at index 5, SMA is mean of indices 0-4 = 1000, current is 2000, ratio = 2.0
        # We check index 5 which is the first of the high volume period
        assert trend.iloc[5] == TrendDirection.UP.value
        assert ratio.iloc[5] > 1.2

    def test_volume_trend_down(self):
        # Last 5 values are 1000, current is 1000, but SMA of last 5 is 2000 (from first 5)
        volume = pd.Series([2000] * 5 + [1000] * 5)
        trend, ratio = calculate_volume_trend(volume, 5)
        # At index 5, SMA is mean of indices 0-4 = 2000, current is 1000, ratio = 0.5
        assert trend.iloc[5] == TrendDirection.DOWN.value
        assert ratio.iloc[5] < 0.8

    def test_volume_trend_sideways(self):
        volume = pd.Series([1000] * 10)
        trend, ratio = calculate_volume_trend(volume, 5)
        assert trend.iloc[-1] == TrendDirection.SIDEWAYS.value
        assert 0.8 <= ratio.iloc[-1] <= 1.2


class Test52WeekPosition:
    def test_52week_position(self):
        close = pd.Series(range(100, 200))
        high = pd.Series(range(105, 205))
        low = pd.Series(range(95, 195))
        week52_high, week52_low, position = calculate_52week_position(close, high, low, 50)

        assert len(position) == len(close)
        assert all((position.dropna() >= 0) & (position.dropna() <= 1))


class TestRealizedVolatility:
    def test_realized_volatility(self):
        close = pd.Series([100 + i + np.sin(i) * 2 for i in range(50)])
        vol = calculate_realized_volatility(close, 20)
        assert len(vol) == len(close)
        assert all(vol.dropna() >= 0)


class TestRollingReturns:
    def test_rolling_returns(self):
        close = pd.Series([100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120])
        returns = calculate_rolling_returns(close, 5)
        assert len(returns) == len(close)
        assert returns.iloc[5] == pytest.approx(0.1, rel=0.1)


class TestDrawdown:
    def test_drawdown_calculation(self):
        close = pd.Series([100, 110, 105, 115, 100, 90, 95, 100, 110, 120])
        drawdown, running_max, max_dd = calculate_drawdown(close)

        assert len(drawdown) == len(close)
        assert drawdown.iloc[0] == 0
        assert drawdown.iloc[4] < 0
        assert max_dd.iloc[-1] <= 0


class TestMomentum:
    def test_momentum(self):
        close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
        momentum = calculate_momentum(close, 5)
        assert len(momentum) == len(close)
        assert momentum.iloc[5] == pytest.approx(0.05, rel=0.1)


class TestRelativeStrength:
    def test_relative_strength(self):
        close = pd.Series([100, 102, 104, 106, 108, 110])
        bench = pd.Series([100, 101, 102, 103, 104, 105])
        rs = calculate_relative_strength(close, bench, 3)
        assert len(rs) == len(close)
        assert rs.iloc[-1] > 0


class TestComputeAllIndicators:
    def test_compute_all_indicators_minimum_data(self):
        df = pd.DataFrame({
            "open": [100 + i for i in range(250)],
            "high": [105 + i for i in range(250)],
            "low": [95 + i for i in range(250)],
            "close": [102 + i for i in range(250)],
            "volume": [1000000 for _ in range(250)],
        })

        results = compute_all_indicators(df)
        assert len(results) > 0
        assert all(isinstance(r, TechnicalIndicators) for r in results)

    def test_compute_all_indicators_insufficient_data(self):
        df = pd.DataFrame({
            "open": [100, 101],
            "high": [105, 106],
            "low": [95, 96],
            "close": [102, 103],
            "volume": [1000000, 1000000],
        })

        results = compute_all_indicators(df)
        assert len(results) == 0

    def test_compute_all_indicators_with_benchmark(self):
        df = pd.DataFrame({
            "open": [100 + i for i in range(250)],
            "high": [105 + i for i in range(250)],
            "low": [95 + i for i in range(250)],
            "close": [102 + i for i in range(250)],
            "volume": [1000000 for _ in range(250)],
        })
        bench = pd.DataFrame({
            "close": [1000 + i for i in range(250)],
        })

        results = compute_all_indicators(df, bench)
        assert len(results) > 0
        assert results[-1].relative_strength is not None

    def test_compute_all_indicators_missing_column(self):
        df = pd.DataFrame({
            "open": [100 + i for i in range(250)],
            "high": [105 + i for i in range(250)],
            "low": [95 + i for i in range(250)],
            "volume": [1000000 for _ in range(250)],
        })

        results = compute_all_indicators(df)
        assert len(results) == 0


class TestTechnicalIndicatorsDataclass:
    def test_technical_indicators_creation(self):
        ti = TechnicalIndicators(
            symbol="RELIANCE",
            date=date(2024, 1, 15),
            close=2500.0,
            sma_20=2480.0,
            rsi_14=65.0,
        )
        assert ti.symbol == "RELIANCE"
        assert ti.close == 2500.0
        assert ti.sma_20 == 2480.0
        assert ti.rsi_14 == 65.0
