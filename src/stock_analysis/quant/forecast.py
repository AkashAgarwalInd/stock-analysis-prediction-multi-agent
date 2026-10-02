from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class QuantBaseline:
    last_close: float
    horizon_trading_days: int
    prob_up: float
    prob_flat: float
    prob_down: float
    expected_return_pct: float
    p10_price: float
    p50_price: float
    p90_price: float
    weekly_vol_pct: float
    method: str


@dataclass(frozen=True)
class DailyQuantiles:
    """Distribution of the simulated price at one trading day of the horizon."""

    day_index: int
    predicted_return_pct: float
    p10_price: float
    p50_price: float
    p90_price: float
    prob_up: float


class VolatilityModel(ABC):
    @abstractmethod
    def estimate(self, returns: NDArray[np.float64]) -> float:
        pass

    @abstractmethod
    def simulate_paths(
        self,
        returns: NDArray[np.float64],
        n_paths: int,
        horizon: int,
        drift: float,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        pass


class EWMAVolatility(VolatilityModel):
    def __init__(self, lambda_: float = 0.94):
        self.lambda_ = lambda_

    def estimate(self, returns: NDArray[np.float64]) -> float:
        if len(returns) == 0:
            return 0.0
        weights = np.array([(1 - self.lambda_) * (self.lambda_ ** i) for i in range(len(returns))])
        weights = weights[::-1]
        weights = weights / weights.sum()
        variance = np.sum(weights * (returns ** 2))
        return float(np.sqrt(variance))

    def simulate_paths(
        self,
        returns: NDArray[np.float64],
        n_paths: int,
        horizon: int,
        drift: float,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        vol = self.estimate(returns)
        if vol == 0:
            return np.full((n_paths, horizon), drift)
        shocks = rng.normal(0, vol, size=(n_paths, horizon))
        return drift + shocks


class HistoricalReturns:
    def __init__(self, prices: NDArray[np.float64]):
        if len(prices) < 2:
            raise ValueError("Need at least 2 prices to compute returns")
        self.prices = prices
        self.log_returns = np.diff(np.log(prices))
        self.simple_returns = np.diff(prices) / prices[:-1]

    def get_log_returns(self) -> NDArray[np.float64]:
        return self.log_returns

    def get_simple_returns(self) -> NDArray[np.float64]:
        return self.simple_returns

    def mean_log_return(self) -> float:
        return float(np.mean(self.log_returns))

    def mean_simple_return(self) -> float:
        return float(np.mean(self.simple_returns))


def compute_momentum_drift(returns: NDArray[np.float64], window: int = 20) -> float:
    if len(returns) < window:
        return 0.0
    recent = returns[-window:]
    return float(np.mean(recent))


class QuantForecaster:
    DEFAULT_N_PATHS = 10_000
    DEFAULT_HORIZON = 5
    FLAT_THRESHOLD_PCT = 0.5

    def __init__(
        self,
        n_paths: int = DEFAULT_N_PATHS,
        horizon: int = DEFAULT_HORIZON,
        flat_threshold_pct: float = FLAT_THRESHOLD_PCT,
        volatility_model: Optional[VolatilityModel] = None,
        seed: Optional[int] = None,
    ):
        self.n_paths = n_paths
        self.horizon = horizon
        self.flat_threshold_pct = flat_threshold_pct
        self.volatility_model = volatility_model or EWMAVolatility()
        self.seed = seed

    def forecast(
        self,
        prices: NDArray[np.float64],
        calibration_params: Optional[dict] = None,
        regime: Optional[str] = None,
    ) -> QuantBaseline:
        return self.forecast_with_daily_path(prices, calibration_params, regime)[0]

    def forecast_with_daily_path(
        self,
        prices: NDArray[np.float64],
        calibration_params: Optional[dict] = None,
        regime: Optional[str] = None,
    ) -> tuple[QuantBaseline, list[DailyQuantiles]]:
        """Return the endpoint baseline plus per-day quantiles from the same paths.

        Both come from a single simulation, so the last daily entry matches the
        endpoint baseline exactly.
        """
        if len(prices) < 2:
            raise ValueError("Need at least 2 price points")

        last_close = float(prices[-1])
        returns_calculator = HistoricalReturns(prices)
        log_returns = returns_calculator.get_log_returns()

        drift = returns_calculator.mean_log_return()
        if calibration_params and "momentum_drift" in calibration_params:
            drift = calibration_params["momentum_drift"]
        elif regime == "trending":
            drift = compute_momentum_drift(log_returns)

        rng = np.random.default_rng(self.seed)
        vol = self.volatility_model.estimate(log_returns)
        paths = self.volatility_model.simulate_paths(
            log_returns, self.n_paths, self.horizon, drift, rng
        )

        cumulative_returns = np.cumsum(paths, axis=1)
        flat_threshold = self.flat_threshold_pct / 100.0
        daily_path = [
            self._daily_quantiles(day + 1, last_close, cumulative_returns[:, day], flat_threshold)
            for day in range(cumulative_returns.shape[1])
        ]
        final_log_returns = cumulative_returns[:, -1]
        final_prices = last_close * np.exp(final_log_returns)

        p10 = float(np.percentile(final_prices, 10))
        p50 = float(np.percentile(final_prices, 50))
        p90 = float(np.percentile(final_prices, 90))

        expected_return = float(np.mean(final_log_returns))
        expected_return_pct = (np.exp(expected_return) - 1) * 100

        p_up = float(np.mean(final_log_returns > flat_threshold))
        p_down = float(np.mean(final_log_returns < -flat_threshold))
        p_flat = 1.0 - p_up - p_down

        weekly_vol = vol * np.sqrt(5) * 100

        return QuantBaseline(
            last_close=last_close,
            horizon_trading_days=self.horizon,
            prob_up=p_up,
            prob_flat=p_flat,
            prob_down=p_down,
            expected_return_pct=expected_return_pct,
            p10_price=p10,
            p50_price=p50,
            p90_price=p90,
            weekly_vol_pct=weekly_vol,
            method="ewma_monte_carlo",
        ), daily_path

    @staticmethod
    def _daily_quantiles(
        day_index: int,
        last_close: float,
        log_returns: NDArray[np.float64],
        flat_threshold: float,
    ) -> DailyQuantiles:
        """Summarise the simulated cumulative log returns at one horizon day."""
        prices = last_close * np.exp(log_returns)
        return DailyQuantiles(
            day_index=day_index,
            predicted_return_pct=float((np.exp(np.mean(log_returns)) - 1) * 100),
            p10_price=float(np.percentile(prices, 10)),
            p50_price=float(np.percentile(prices, 50)),
            p90_price=float(np.percentile(prices, 90)),
            prob_up=float(np.mean(log_returns > flat_threshold)),
        )
