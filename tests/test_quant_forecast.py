import numpy as np
import pytest

from stock_analysis.quant import (
    EWMAVolatility,
    HistoricalReturns,
    QuantBaseline,
    QuantForecaster,
    VolatilityModel,
    compute_momentum_drift,
)


class TestQuantBaseline:
    def test_quant_baseline_creation(self):
        baseline = QuantBaseline(
            last_close=100.0,
            horizon_trading_days=5,
            prob_up=0.5,
            prob_flat=0.3,
            prob_down=0.2,
            expected_return_pct=1.5,
            p10_price=98.0,
            p50_price=101.0,
            p90_price=104.0,
            weekly_vol_pct=2.5,
            method="ewma_monte_carlo",
        )
        assert baseline.last_close == 100.0
        assert baseline.horizon_trading_days == 5
        assert baseline.prob_up == 0.5
        assert baseline.prob_flat == 0.3
        assert baseline.prob_down == 0.2
        assert baseline.expected_return_pct == 1.5
        assert baseline.p10_price == 98.0
        assert baseline.p50_price == 101.0
        assert baseline.p90_price == 104.0
        assert baseline.weekly_vol_pct == 2.5
        assert baseline.method == "ewma_monte_carlo"


class TestHistoricalReturns:
    def test_historical_returns_creation(self):
        prices = np.array([100.0, 101.0, 102.0, 101.5, 103.0])
        hr = HistoricalReturns(prices)
        assert len(hr.get_log_returns()) == 4
        assert len(hr.get_simple_returns()) == 4

    def test_historical_returns_insufficient_data(self):
        prices = np.array([100.0])
        with pytest.raises(ValueError, match="Need at least 2 prices"):
            HistoricalReturns(prices)

    def test_mean_log_return(self):
        prices = np.array([100.0, 101.0, 102.01, 103.0301])
        hr = HistoricalReturns(prices)
        mean_log = hr.mean_log_return()
        assert abs(mean_log - 0.01) < 0.001

    def test_mean_simple_return(self):
        prices = np.array([100.0, 101.0, 102.0, 103.0])
        hr = HistoricalReturns(prices)
        mean_simple = hr.mean_simple_return()
        assert abs(mean_simple - 0.01) < 0.001


class TestEWMAVolatility:
    def test_estimate_basic(self):
        returns = np.array([0.01, -0.01, 0.02, -0.02, 0.015])
        ewma = EWMAVolatility(lambda_=0.94)
        vol = ewma.estimate(returns)
        assert vol > 0
        assert isinstance(vol, float)

    def test_estimate_empty_returns(self):
        returns = np.array([])
        ewma = EWMAVolatility()
        vol = ewma.estimate(returns)
        assert vol == 0.0

    def test_estimate_zero_returns(self):
        returns = np.zeros(10)
        ewma = EWMAVolatility()
        vol = ewma.estimate(returns)
        assert vol == 0.0

    def test_simulate_paths_shape(self):
        returns = np.random.normal(0, 0.01, 100)
        ewma = EWMAVolatility()
        rng = np.random.default_rng(42)
        paths = ewma.simulate_paths(returns, n_paths=100, horizon=5, drift=0.001, rng=rng)
        assert paths.shape == (100, 5)

    def test_simulate_paths_zero_vol(self):
        returns = np.zeros(10)
        ewma = EWMAVolatility()
        rng = np.random.default_rng(42)
        paths = ewma.simulate_paths(returns, n_paths=100, horizon=5, drift=0.001, rng=rng)
        assert np.allclose(paths, 0.001)

    def test_deterministic_seed(self):
        returns = np.random.normal(0, 0.01, 100)
        ewma = EWMAVolatility()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        paths1 = ewma.simulate_paths(returns, n_paths=100, horizon=5, drift=0.001, rng=rng1)
        paths2 = ewma.simulate_paths(returns, n_paths=100, horizon=5, drift=0.001, rng=rng2)
        assert np.array_equal(paths1, paths2)


class TestComputeMomentumDrift:
    def test_compute_momentum_drift_basic(self):
        returns = np.array([0.01] * 30)
        drift = compute_momentum_drift(returns, window=20)
        assert abs(drift - 0.01) < 1e-10

    def test_compute_momentum_drift_insufficient_data(self):
        returns = np.array([0.01, 0.02])
        drift = compute_momentum_drift(returns, window=20)
        assert drift == 0.0

    def test_compute_momentum_drift_variable(self):
        returns = np.array([0.02] * 10 + [-0.01] * 10)
        drift = compute_momentum_drift(returns, window=10)
        assert abs(drift - (-0.01)) < 1e-10


class TestQuantForecaster:
    @pytest.fixture
    def stable_prices(self):
        np.random.seed(42)
        base = 100.0
        returns = np.random.normal(0.0005, 0.005, 100)
        prices = [base]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        return np.array(prices)

    @pytest.fixture
    def high_vol_prices(self):
        np.random.seed(42)
        base = 100.0
        returns = np.random.normal(0.001, 0.03, 100)
        prices = [base]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        return np.array(prices)

    @pytest.fixture
    def low_vol_prices(self):
        np.random.seed(42)
        base = 100.0
        returns = np.random.normal(0.0002, 0.001, 100)
        prices = [base]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        return np.array(prices)

    @pytest.fixture
    def trending_prices(self):
        np.random.seed(42)
        base = 100.0
        returns = np.random.normal(0.005, 0.01, 100)
        prices = [base]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        return np.array(prices)

    def test_forecast_deterministic_seed(self, stable_prices):
        forecaster1 = QuantForecaster(seed=42, n_paths=1000)
        forecaster2 = QuantForecaster(seed=42, n_paths=1000)
        result1 = forecaster1.forecast(stable_prices)
        result2 = forecaster2.forecast(stable_prices)
        assert result1.p10_price == result2.p10_price
        assert result1.p50_price == result2.p50_price
        assert result1.p90_price == result2.p90_price
        assert result1.expected_return_pct == result2.expected_return_pct
        assert result1.prob_up == result2.prob_up
        assert result1.prob_flat == result2.prob_flat
        assert result1.prob_down == result2.prob_down
        assert result1.weekly_vol_pct == result2.weekly_vol_pct

    def test_forecast_different_seeds_different_results(self, stable_prices):
        forecaster1 = QuantForecaster(seed=42, n_paths=1000)
        forecaster2 = QuantForecaster(seed=123, n_paths=1000)
        result1 = forecaster1.forecast(stable_prices)
        result2 = forecaster2.forecast(stable_prices)
        assert result1.p50_price != result2.p50_price

    def test_forecast_probability_sum(self, stable_prices):
        forecaster = QuantForecaster(seed=42, n_paths=10000)
        result = forecaster.forecast(stable_prices)
        total = result.prob_up + result.prob_flat + result.prob_down
        assert abs(total - 1.0) < 0.01

    def test_forecast_p10_lt_p50_lt_p90(self, stable_prices):
        forecaster = QuantForecaster(seed=42, n_paths=10000)
        result = forecaster.forecast(stable_prices)
        assert result.p10_price < result.p50_price < result.p90_price

    def test_forecast_flat_threshold(self, stable_prices):
        forecaster = QuantForecaster(seed=42, n_paths=10000, flat_threshold_pct=0.5)
        result = forecaster.forecast(stable_prices)
        assert result.prob_flat >= 0
        assert result.prob_flat <= 1

    def test_forecast_stable_prices(self, stable_prices):
        forecaster = QuantForecaster(seed=42, n_paths=10000)
        result = forecaster.forecast(stable_prices)
        assert result.last_close == stable_prices[-1]
        assert result.horizon_trading_days == 5
        assert result.method == "ewma_monte_carlo"
        assert result.weekly_vol_pct >= 0

    def test_forecast_high_vol_prices(self, high_vol_prices):
        forecaster = QuantForecaster(seed=42, n_paths=10000)
        result = forecaster.forecast(high_vol_prices)
        assert result.weekly_vol_pct > 0
        assert result.p90_price - result.p10_price > 0

    def test_forecast_low_vol_prices(self, low_vol_prices):
        forecaster = QuantForecaster(seed=42, n_paths=10000)
        result = forecaster.forecast(low_vol_prices)
        assert result.weekly_vol_pct >= 0
        spread = result.p90_price - result.p10_price
        assert spread >= 0

    def test_forecast_with_calibration_params(self, stable_prices):
        forecaster = QuantForecaster(seed=42, n_paths=1000)
        result = forecaster.forecast(stable_prices, calibration_params={"momentum_drift": 0.01})
        assert result.expected_return_pct > 0

    def test_forecast_with_regime_trending(self, trending_prices):
        forecaster = QuantForecaster(seed=42, n_paths=1000)
        result = forecaster.forecast(trending_prices, regime="trending")
        assert result.expected_return_pct > 0

    def test_forecast_insufficient_data(self):
        forecaster = QuantForecaster(seed=42)
        prices = np.array([100.0])
        with pytest.raises(ValueError, match="Need at least 2 price points"):
            forecaster.forecast(prices)

    def test_forecast_custom_horizon(self, stable_prices):
        forecaster = QuantForecaster(seed=42, n_paths=1000, horizon=10)
        result = forecaster.forecast(stable_prices)
        assert result.horizon_trading_days == 10

    def test_forecast_custom_n_paths(self, stable_prices):
        forecaster = QuantForecaster(seed=42, n_paths=5000, horizon=5)
        result = forecaster.forecast(stable_prices)
        assert isinstance(result, QuantBaseline)

    def test_forecast_returns_valid_baseline(self, stable_prices):
        forecaster = QuantForecaster(seed=42, n_paths=1000)
        result = forecaster.forecast(stable_prices)
        assert isinstance(result, QuantBaseline)
        assert result.prob_up >= 0 and result.prob_up <= 1
        assert result.prob_flat >= 0 and result.prob_flat <= 1
        assert result.prob_down >= 0 and result.prob_down <= 1
        assert result.p10_price > 0
        assert result.p50_price > 0
        assert result.p90_price > 0
        assert result.weekly_vol_pct >= 0


class TestVolatilityModelInterface:
    def test_volatility_model_is_abstract(self):
        with pytest.raises(TypeError):
            VolatilityModel()

    def test_custom_volatility_model(self):
        class ConstantVol(VolatilityModel):
            def estimate(self, returns):
                return 0.02

            def simulate_paths(self, returns, n_paths, horizon, drift, rng):
                return np.full((n_paths, horizon), drift + 0.02)

        prices = np.array([100.0] + [100.0 * (1.001 ** i) for i in range(1, 50)])
        forecaster = QuantForecaster(seed=42, n_paths=100, volatility_model=ConstantVol())
        result = forecaster.forecast(prices)
        assert isinstance(result, QuantBaseline)
