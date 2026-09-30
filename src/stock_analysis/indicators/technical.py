import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Optional
from enum import Enum

from stock_analysis.logging import get_logger

logger = get_logger(__name__)


class TrendDirection(Enum):
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


@dataclass
class IndicatorResult:
    name: str
    value: Optional[float]
    timestamp: date
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TechnicalIndicators:
    symbol: str
    date: date
    close: float
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    atr_14: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None
    bb_percent: Optional[float] = None
    volume_trend: Optional[str] = None
    volume_sma_ratio: Optional[float] = None
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None
    week52_position: Optional[float] = None
    realized_vol_20: Optional[float] = None
    realized_vol_60: Optional[float] = None
    rolling_return_20: Optional[float] = None
    rolling_return_60: Optional[float] = None
    max_drawdown: Optional[float] = None
    current_drawdown: Optional[float] = None
    momentum_10: Optional[float] = None
    momentum_20: Optional[float] = None
    relative_strength: Optional[float] = None


def _ensure_series(series: pd.Series) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        return series.iloc[:, 0]
    return series


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    series = _ensure_series(series)
    return series.rolling(window=window, min_periods=window).mean()


def calculate_ema(series: pd.Series, window: int) -> pd.Series:
    series = _ensure_series(series)
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    series = _ensure_series(series)
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    series = _ensure_series(series)
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14
) -> pd.Series:
    high = _ensure_series(high)
    low = _ensure_series(low)
    close = _ensure_series(close)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    return atr


def calculate_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    series = _ensure_series(series)
    middle = calculate_sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    width = (upper - lower) / middle
    percent = (series - lower) / (upper - lower)
    return upper, middle, lower, width, percent


def calculate_volume_trend(volume: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series]:
    volume = _ensure_series(volume)
    volume_sma = calculate_sma(volume, window)
    ratio = volume / volume_sma
    trend = pd.Series(index=volume.index, dtype=object)
    trend[ratio > 1.2] = TrendDirection.UP.value
    trend[ratio < 0.8] = TrendDirection.DOWN.value
    trend[(ratio >= 0.8) & (ratio <= 1.2)] = TrendDirection.SIDEWAYS.value
    return trend, ratio


def calculate_52week_position(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    window: int = 252
) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = _ensure_series(close)
    high = _ensure_series(high)
    low = _ensure_series(low)

    week52_high = high.rolling(window=window, min_periods=window).max()
    week52_low = low.rolling(window=window, min_periods=window).min()
    position = (close - week52_low) / (week52_high - week52_low)
    return week52_high, week52_low, position


def calculate_realized_volatility(
    close: pd.Series,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    close = _ensure_series(close)
    returns = close.pct_change()
    vol = returns.rolling(window=window, min_periods=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def calculate_rolling_returns(
    close: pd.Series,
    window: int = 20
) -> pd.Series:
    close = _ensure_series(close)
    returns = (close / close.shift(window)) - 1
    return returns


def calculate_drawdown(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = _ensure_series(close)
    running_max = close.expanding().max()
    drawdown = (close - running_max) / running_max
    max_drawdown = drawdown.expanding().min()
    return drawdown, running_max, max_drawdown


def calculate_momentum(close: pd.Series, window: int = 10) -> pd.Series:
    close = _ensure_series(close)
    momentum = close.pct_change(window)
    return momentum


def calculate_relative_strength(
    close: pd.Series,
    benchmark_close: pd.Series,
    window: int = 60
) -> pd.Series:
    close = _ensure_series(close)
    benchmark_close = _ensure_series(benchmark_close)

    asset_returns = close.pct_change(window)
    bench_returns = benchmark_close.pct_change(window)
    rs = (1 + asset_returns) / (1 + bench_returns) - 1
    return rs


def compute_all_indicators(
    df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None
) -> list[TechnicalIndicators]:
    if df.empty or len(df) < 200:
        logger.warning("insufficient_data_for_indicators", rows=len(df))
        return []

    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            logger.error("missing_required_column", column=col)
            return []

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    sma_20 = calculate_sma(close, 20)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    ema_12 = calculate_ema(close, 12)
    ema_26 = calculate_ema(close, 26)
    rsi_14 = calculate_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close)
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_middle, bb_lower, bb_width, bb_percent = calculate_bollinger_bands(close)
    volume_trend, volume_ratio = calculate_volume_trend(volume)
    week52_high, week52_low, week52_pos = calculate_52week_position(close, high, low)
    realized_vol_20 = calculate_realized_volatility(close, 20)
    realized_vol_60 = calculate_realized_volatility(close, 60)
    rolling_ret_20 = calculate_rolling_returns(close, 20)
    rolling_ret_60 = calculate_rolling_returns(close, 60)
    drawdown, running_max, max_dd = calculate_drawdown(close)
    momentum_10 = calculate_momentum(close, 10)
    momentum_20 = calculate_momentum(close, 20)

    relative_strength = None
    if benchmark_df is not None and "close" in benchmark_df.columns:
        relative_strength = calculate_relative_strength(close, benchmark_df["close"])

    results = []
    for idx in df.index:
        if pd.isna(sma_20.loc[idx]) and pd.isna(rsi_14.loc[idx]):
            continue

        indicators = TechnicalIndicators(
            symbol="",
            date=idx.date() if hasattr(idx, "date") else idx,
            close=float(close.loc[idx]),
            sma_20=float(sma_20.loc[idx]) if pd.notna(sma_20.loc[idx]) else None,
            sma_50=float(sma_50.loc[idx]) if pd.notna(sma_50.loc[idx]) else None,
            sma_200=float(sma_200.loc[idx]) if pd.notna(sma_200.loc[idx]) else None,
            ema_12=float(ema_12.loc[idx]) if pd.notna(ema_12.loc[idx]) else None,
            ema_26=float(ema_26.loc[idx]) if pd.notna(ema_26.loc[idx]) else None,
            rsi_14=float(rsi_14.loc[idx]) if pd.notna(rsi_14.loc[idx]) else None,
            macd=float(macd_line.loc[idx]) if pd.notna(macd_line.loc[idx]) else None,
            macd_signal=float(macd_signal.loc[idx]) if pd.notna(macd_signal.loc[idx]) else None,
            macd_histogram=float(macd_hist.loc[idx]) if pd.notna(macd_hist.loc[idx]) else None,
            atr_14=float(atr_14.loc[idx]) if pd.notna(atr_14.loc[idx]) else None,
            bb_upper=float(bb_upper.loc[idx]) if pd.notna(bb_upper.loc[idx]) else None,
            bb_middle=float(bb_middle.loc[idx]) if pd.notna(bb_middle.loc[idx]) else None,
            bb_lower=float(bb_lower.loc[idx]) if pd.notna(bb_lower.loc[idx]) else None,
            bb_width=float(bb_width.loc[idx]) if pd.notna(bb_width.loc[idx]) else None,
            bb_percent=float(bb_percent.loc[idx]) if pd.notna(bb_percent.loc[idx]) else None,
            volume_trend=volume_trend.loc[idx] if pd.notna(volume_trend.loc[idx]) else None,
            volume_sma_ratio=float(volume_ratio.loc[idx]) if pd.notna(volume_ratio.loc[idx]) else None,
            week52_high=float(week52_high.loc[idx]) if pd.notna(week52_high.loc[idx]) else None,
            week52_low=float(week52_low.loc[idx]) if pd.notna(week52_low.loc[idx]) else None,
            week52_position=float(week52_pos.loc[idx]) if pd.notna(week52_pos.loc[idx]) else None,
            realized_vol_20=float(realized_vol_20.loc[idx]) if pd.notna(realized_vol_20.loc[idx]) else None,
            realized_vol_60=float(realized_vol_60.loc[idx]) if pd.notna(realized_vol_60.loc[idx]) else None,
            rolling_return_20=float(rolling_ret_20.loc[idx]) if pd.notna(rolling_ret_20.loc[idx]) else None,
            rolling_return_60=float(rolling_ret_60.loc[idx]) if pd.notna(rolling_ret_60.loc[idx]) else None,
            max_drawdown=float(max_dd.loc[idx]) if pd.notna(max_dd.loc[idx]) else None,
            current_drawdown=float(drawdown.loc[idx]) if pd.notna(drawdown.loc[idx]) else None,
            momentum_10=float(momentum_10.loc[idx]) if pd.notna(momentum_10.loc[idx]) else None,
            momentum_20=float(momentum_20.loc[idx]) if pd.notna(momentum_20.loc[idx]) else None,
            relative_strength=float(relative_strength.loc[idx]) if relative_strength is not None and pd.notna(relative_strength.loc[idx]) else None,
        )
        results.append(indicators)

    return results