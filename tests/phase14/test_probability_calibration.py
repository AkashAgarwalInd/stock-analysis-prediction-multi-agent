"""Plan.md Phase 14: probability buckets, observed frequency, Brier score and calibration
reporting ("done when" synthetic probability data produces expected calibration statistics)."""

import random
import statistics
from datetime import date, timedelta

import pytest

from stock_analysis.learning import (
    probability_calibration,
    reliability_table,
    render_probability_calibration,
)
from stock_analysis.learning.evaluation import bucket_index
from stock_analysis.schemas.track_record import EvaluatedForecast, VariantScore


def _table(pairs, n_buckets=10, min_samples=20):
    return reliability_table(pairs, event="up", n_buckets=n_buckets, min_samples=min_samples)


def _calibrated_pairs(n, seed=7):
    """Outcomes drawn with exactly the predicted probability."""
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        p = rng.random()
        pairs.append((p, rng.random() < p))
    return pairs


def _overconfident_pairs(n, seed=11):
    """The true probability is only half as far from 50% as the predicted one."""
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        p = rng.random()
        pairs.append((p, rng.random() < 0.5 + 0.5 * (p - 0.5)))
    return pairs


class TestBuckets:
    @pytest.mark.parametrize(
        ("p", "expected"),
        [(0.0, 0), (0.05, 0), (0.1, 1), (0.3, 3), (0.7, 7), (0.6999, 6), (0.999, 9), (1.0, 9)],
    )
    def test_bucket_index(self, p, expected):
        assert bucket_index(p, 10) == expected

    def test_bucket_index_rejects_non_probabilities(self):
        with pytest.raises(ValueError):
            bucket_index(1.01, 10)
        with pytest.raises(ValueError):
            bucket_index(-0.01, 10)

    def test_bucket_count_is_configurable(self):
        table = _table([(0.1, False), (0.4, True), (0.6, True), (0.9, True)], n_buckets=2)
        assert [(b.low, b.high, b.n) for b in table.buckets] == [(0.0, 0.5, 2), (0.5, 1.0, 2)]


class TestHandComputedStatistics:
    """Ten forecasts at 20% (2 happened) and ten at 80% (6 happened)."""

    @pytest.fixture
    def table(self):
        pairs = [(0.2, i < 2) for i in range(10)] + [(0.8, i < 6) for i in range(10)]
        return _table(pairs)

    def test_buckets_and_observed_frequency(self, table):
        low, high = table.buckets
        assert (low.low, low.n, low.mean_predicted, low.observed.rate) == (0.2, 10, 0.2, 0.2)
        assert (high.low, high.n, high.observed.hits) == (0.8, 10, 6)
        assert high.mean_predicted == pytest.approx(0.8)
        assert high.gap == pytest.approx(-0.2)
        assert low.gap == pytest.approx(0.0)
        assert high.observed.ci_low < 0.6 < high.observed.ci_high

    def test_brier_and_murphy_decomposition(self, table):
        assert table.n == 20 and table.base_rate == pytest.approx(0.4)
        assert table.brier == pytest.approx(0.22)
        assert table.reliability == pytest.approx(0.02)
        assert table.resolution == pytest.approx(0.04)
        assert table.uncertainty == pytest.approx(0.24)
        assert table.ece == pytest.approx(0.1)
        # Exact when every forecast in a bucket has the same probability
        assert table.brier == pytest.approx(
            table.reliability - table.resolution + table.uncertainty
        )

    def test_finding_is_withheld_below_the_minimum(self, table):
        assert not table.finding.startswith("insufficient")
        small = _table([(0.2, False)] * 5, min_samples=20)
        assert small.finding == "insufficient evidence: 5 forecasts, at least 20 needed"
        assert small.buckets[0].n == 5  # the statistics are still shown

    def test_empty(self):
        empty = _table([])
        assert empty.n == 0 and empty.buckets == [] and empty.brier is None
        assert empty.finding.startswith("insufficient evidence: 0 forecasts")


class TestSyntheticData:
    def test_calibrated_probabilities_match_observed_frequencies(self):
        table = _table(_calibrated_pairs(20_000))
        assert len(table.buckets) == 10
        assert all(abs(b.gap) < 0.03 for b in table.buckets)
        assert all(
            b.observed.ci_low <= b.mean_predicted <= b.observed.ci_high for b in table.buckets
        )
        assert table.ece < 0.02
        assert table.reliability < 0.001
        # Uniform probabilities with matching outcomes: base rate 1/2, resolution 1/12
        assert table.base_rate == pytest.approx(0.5, abs=0.01)
        assert table.resolution == pytest.approx(1 / 12, abs=0.01)
        assert table.brier == pytest.approx(1 / 6, abs=0.01)  # E[p(1-p)] for p ~ U(0, 1)
        assert table.finding.startswith("observed frequencies are consistent")

    def test_overconfidence_is_detected(self):
        table = _table(_overconfident_pairs(20_000))
        high = [b for b in table.buckets if b.low >= 0.7]
        low = [b for b in table.buckets if b.high <= 0.3]
        assert all(b.gap < -0.05 for b in high)  # happened less often than predicted
        assert all(b.gap > 0.05 for b in low)  # and the "unlikely" outcomes more often
        assert table.ece == pytest.approx(0.125, abs=0.01)  # E|p - q| = 0.5 * E|p - 0.5|
        assert table.reliability > 0.01
        assert "happened less often than predicted in" in table.finding
        assert "80%–90%" in table.finding and "90%–100%" in table.finding
        assert "happened more often than predicted in" in table.finding

    def test_constant_climatology_has_no_resolution(self):
        rng = random.Random(3)
        table = _table([(0.4, rng.random() < 0.4) for _ in range(5_000)])
        assert len(table.buckets) == 1
        assert table.resolution == pytest.approx(0.0)
        assert table.uncertainty == pytest.approx(0.24, abs=0.01)
        assert table.brier == pytest.approx(table.reliability + table.uncertainty)


def _evaluated(i, up, flat, down, realized):
    from stock_analysis.review import metrics

    score = VariantScore(
        prob_up=up,
        prob_flat=flat,
        prob_down=down,
        direction_correct=metrics.predicted_direction(up, flat, down) == realized,
        signed_error_pct=0.0,
        in_80pct_band=True,
        brier=metrics.brier_score(up, flat, down, realized),
        pinball=0.5,
        sharpness_pct=5.0,
    )
    naive = VariantScore(direction_correct=realized == "flat", signed_error_pct=0.0)
    return EvaluatedForecast(
        forecast_id=f"f{i}",
        ticker="RELIANCE",
        as_of_date=date(2026, 1, 5) + timedelta(weeks=i),
        target_date=date(2026, 1, 12) + timedelta(weeks=i),
        realized_direction=realized,
        actual_return_pct=0.0,
        variants={
            "naive_flat": naive,
            "quant_uncalibrated": score,
            "quant_calibrated": score,
            "final": score,
        },
    )


class TestVariantCalibration:
    @pytest.fixture
    def forecasts(self):
        rng = random.Random(5)
        out = []
        for i in range(3000):
            up = rng.choice([0.2, 0.4, 0.6])
            down = round((1 - up) * 0.6, 4)
            flat = 1 - up - down
            draw = rng.random()
            realized = "up" if draw < up else ("down" if draw < up + down else "flat")
            out.append(_evaluated(i, up, flat, down, realized))
        return out

    def test_three_events_and_pooled(self, forecasts):
        cal = probability_calibration(forecasts, "final", n_buckets=10)
        assert set(cal.events) == {"up", "flat", "down", "all"}
        assert cal.n_forecasts == 3000
        assert cal.events["up"].n == cal.events["flat"].n == cal.events["down"].n == 3000
        assert cal.events["all"].n == 9000
        # The pooled binary Brier is a third of the up/flat/down Brier score
        assert cal.events["all"].brier == pytest.approx(cal.brier / 3)
        assert cal.brier == pytest.approx(
            statistics.fmean(f.variants["final"].brier for f in forecasts)
        )
        up = cal.events["up"]
        assert [round(b.mean_predicted, 2) for b in up.buckets] == [0.2, 0.4, 0.6]
        assert all(abs(b.gap) < 0.05 for b in up.buckets)

    def test_naive_benchmark_has_no_probabilities(self, forecasts):
        with pytest.raises(ValueError, match="states no probabilities"):
            probability_calibration(forecasts, "naive_flat")

    def test_bucket_setting(self, forecasts, monkeypatch):
        from stock_analysis.config.settings import get_settings

        monkeypatch.setenv("PROBABILITY_CALIBRATION_BUCKETS", "5")
        get_settings.cache_clear()
        cal = probability_calibration(forecasts, "final")
        assert all(b.high - b.low == pytest.approx(0.2) for b in cal.events["up"].buckets)

    @pytest.mark.parametrize("value", ["0", "101"])
    def test_bucket_setting_is_bounded(self, value, monkeypatch):
        from pydantic import ValidationError

        from stock_analysis.config.settings import get_settings

        monkeypatch.setenv("PROBABILITY_CALIBRATION_BUCKETS", value)
        get_settings.cache_clear()
        with pytest.raises(ValidationError):
            get_settings()

    def test_render(self, forecasts):
        cals = [probability_calibration(forecasts, v) for v in ("quant_calibrated", "final")]
        text = render_probability_calibration(cals, n_buckets=10, ticker="RELIANCE")
        assert "## Probability calibration for RELIANCE" in text
        assert "#### P(up)" in text and "#### P(down)" in text
        assert "| 20%–30% |" in text
        assert "### Quant calibrated: mean up/flat/down Brier" in text
        assert "- P(up): Brier" in text and "ECE" in text

    def test_render_empty(self):
        cal = probability_calibration([], "final")
        text = render_probability_calibration([cal], n_buckets=10)
        assert "No scored forecasts yet." in text
