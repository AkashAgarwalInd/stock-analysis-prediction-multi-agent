"""Plan.md Phase 9: metric definitions (§15), checked against hand-computed values."""

import math

import pytest

from stock_analysis.review import metrics


class TestDirection:
    @pytest.mark.parametrize(
        ("ret", "expected"), [(0.6, "up"), (0.5, "flat"), (-0.5, "flat"), (-0.51, "down")]
    )
    def test_realized_uses_flat_band(self, ret, expected):
        assert metrics.realized_direction(ret, 0.5) == expected

    def test_predicted_is_most_probable(self):
        assert metrics.predicted_direction(0.5, 0.2, 0.3) == "up"
        assert metrics.predicted_direction(0.2, 0.3, 0.5) == "down"

    def test_exact_tie_is_no_call(self):
        assert metrics.predicted_direction(0.4, 0.2, 0.4) == "flat"


class TestBrier:
    def test_hand_computed(self):
        # (0.5-1)^2 + (0.2-0)^2 + (0.3-0)^2
        assert metrics.brier_score(0.5, 0.2, 0.3, "up") == pytest.approx(0.38)

    def test_bounds(self):
        assert metrics.brier_score(1.0, 0.0, 0.0, "up") == 0.0
        assert metrics.brier_score(1.0, 0.0, 0.0, "down") == 2.0


class TestPinball:
    def test_asymmetric_penalty(self):
        assert metrics.pinball_loss(0.1, 0.0, 1.0) == pytest.approx(0.1)
        assert metrics.pinball_loss(0.1, 0.0, -1.0) == pytest.approx(0.9)
        assert metrics.pinball_loss(0.5, 2.0, 2.0) == 0.0

    def test_quantile_losses_in_return_points(self):
        # quantiles at -5%, 0%, +5% returns; actual +2%
        losses = metrics.quantile_losses(100.0, 95.0, 100.0, 105.0, 2.0)
        assert losses["p10"] == pytest.approx(0.1 * 7.0)
        assert losses["p50"] == pytest.approx(0.5 * 2.0)
        assert losses["p90"] == pytest.approx(0.1 * 3.0)
        assert losses["mean"] == pytest.approx((0.7 + 1.0 + 0.3) / 3)


class TestPit:
    def test_matches_forecast_quantiles(self):
        assert metrics.pit_percentile(95.0, 95.0, 100.0, 108.0) == pytest.approx(0.1)
        assert metrics.pit_percentile(100.0, 95.0, 100.0, 108.0) == pytest.approx(0.5)
        assert metrics.pit_percentile(108.0, 95.0, 100.0, 108.0) == pytest.approx(0.9)

    def test_monotonic_and_bounded(self):
        values = [
            metrics.pit_percentile(x, 95.0, 100.0, 108.0) for x in (60, 90, 99, 101, 120, 200)
        ]
        assert values == sorted(values)
        assert 0.0 <= values[0] < 0.01 and 0.99 < values[-1] <= 1.0

    def test_rejects_unordered_quantiles(self):
        with pytest.raises(ValueError):
            metrics.pit_percentile(100.0, 100.0, 100.0, 108.0)


class TestVolatility:
    def test_hand_computed_weekly_vol(self):
        closes = [100.0, 101.0, 99.0, 100.0]
        returns = [math.log(101 / 100), math.log(99 / 101), math.log(100 / 99)]
        mean = sum(returns) / 3
        sd = math.sqrt(sum((r - mean) ** 2 for r in returns) / 2)
        assert metrics.realized_weekly_vol_pct(closes) == pytest.approx(sd * math.sqrt(5) * 100)

    def test_constant_drift_has_zero_vol(self):
        assert metrics.realized_weekly_vol_pct([100.0, 101.0, 102.01]) == pytest.approx(
            0.0, abs=1e-9
        )

    def test_too_few_closes(self):
        assert metrics.realized_weekly_vol_pct([100.0, 101.0]) is None

    def test_ratio(self):
        assert metrics.vol_ratio(4.0, 2.0) == 2.0
        assert metrics.vol_ratio(4.0, 0.0) is None
        assert metrics.vol_ratio(None, 2.0) is None
