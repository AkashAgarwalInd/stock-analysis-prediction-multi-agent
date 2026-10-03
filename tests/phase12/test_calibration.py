"""Plan.md Phase 12 / §63: bounded, shrunk, versioned calibration."""

import math
import sqlite3
from datetime import date, timedelta
from statistics import NormalDist

import pytest

from stock_analysis.config.settings import get_settings
from stock_analysis.database import LearningStore
from stock_analysis.learning import estimate_calibration, observation_from, update_calibration
from stock_analysis.learning.calibration import NORMAL_MEDIAN_ABS
from stock_analysis.schemas.learning import CalibrationObservation
from tests.learning_helpers import evaluated_at, save_scored, snapshot_as_of, week

SIGMA = 0.027  # predicted weekly volatility (log), like the fixture's 2.7%


def _observations(n: int, scale: float, *, bias: float = 0.0) -> list[CalibrationObservation]:
    """Residuals at evenly spaced normal quantiles: realised spread is exactly ``scale`` x sigma.

    Alternating the sign of the quantile pairs keeps the sample symmetric around ``bias``.
    """
    quantiles = [NormalDist().inv_cdf((i + 0.5) / n) for i in range(n)]
    return [
        CalibrationObservation(
            forecast_id=f"f{i}",
            as_of_date=date(2026, 1, 1) + timedelta(days=7 * i),
            target_date=date(2026, 1, 8) + timedelta(days=7 * i),
            residual_log=math.log1p(bias / 100) + scale * SIGMA * q,
            predicted_sigma_log=SIGMA,
        )
        for i, q in enumerate(quantiles)
    ]


class TestMinimumSamples:
    @pytest.mark.parametrize("n", [0, 1, 7])
    def test_below_minimum_nothing_changes(self, n):
        est = estimate_calibration(_observations(n, 3.0, bias=2.0) if n else [])
        assert (est.vol_multiplier, est.p50_bias_shift_pct) == (1.0, 0.0)
        assert est.n_samples == n
        assert "at least 8" in est.reason[0]

    def test_applies_at_minimum(self):
        assert estimate_calibration(_observations(8, 1.5)).vol_multiplier > 1.0

    def test_minimum_is_configurable(self, monkeypatch):
        monkeypatch.setenv("CALIBRATION_MIN_SAMPLES", "12")
        get_settings.cache_clear()
        assert estimate_calibration(_observations(10, 1.5)).vol_multiplier == 1.0


class TestShrinkageAndBounds:
    def test_raw_estimate_matches_the_true_scale(self):
        est = estimate_calibration(_observations(40, 1.3))
        assert est.evidence["raw_vol_multiplier"] == pytest.approx(1.3, abs=0.02)

    @pytest.mark.parametrize("n", [8, 24, 80])
    def test_shrinks_toward_one(self, n):
        est = estimate_calibration(_observations(n, 1.3))
        raw = est.evidence["raw_vol_multiplier"]
        weight = n / (n + 8)
        assert est.evidence["shrinkage_weight"] == pytest.approx(weight, abs=1e-4)
        assert est.vol_multiplier == pytest.approx(1 + weight * (raw - 1), abs=1e-4)
        assert 1.0 < est.vol_multiplier < raw

    @pytest.mark.parametrize(("scale", "bound"), [(4.0, 1.5), (0.2, 0.8)])
    def test_bounds(self, scale, bound):
        assert estimate_calibration(_observations(200, scale)).vol_multiplier == bound

    def test_synthetic_volatility_converges(self):
        """True volatility 1.5x predicted: the multiplier climbs toward 1.5 as evidence grows."""
        values = [
            estimate_calibration(_observations(n, 1.5)).vol_multiplier for n in (8, 20, 52, 200)
        ]
        assert values == sorted(values)
        assert values[0] == pytest.approx(1.25, abs=0.02)
        assert 1.45 < values[-1] <= 1.5

    def test_well_calibrated_model_stays_near_one(self):
        assert estimate_calibration(_observations(52, 1.0)).vol_multiplier == pytest.approx(
            1.0, abs=0.02
        )


class TestBias:
    def test_persistent_bias_is_shrunk_and_bounded(self):
        est = estimate_calibration(_observations(40, 0.5, bias=0.8))
        assert est.evidence["bias_applied"] is True
        assert est.p50_bias_shift_pct == pytest.approx(40 / 48 * 0.8, abs=0.02)
        big = estimate_calibration(_observations(40, 0.5, bias=5.0))
        assert big.p50_bias_shift_pct == get_settings().calibration_bias_max_shift_pct

    def test_noisy_bias_is_ignored(self):
        # same mean error, but far too noisy to be statistically meaningful
        est = estimate_calibration(_observations(20, 3.0, bias=0.8))
        assert est.evidence["bias_applied"] is False
        assert est.p50_bias_shift_pct == 0.0

    def test_no_bias_without_enough_samples(self):
        assert estimate_calibration(_observations(7, 0.5, bias=0.8)).p50_bias_shift_pct == 0.0

    def test_volatility_is_measured_around_the_shifted_centre(self):
        est = estimate_calibration(_observations(60, 1.0, bias=0.8))
        assert est.evidence["bias_applied"] is True
        assert est.vol_multiplier == pytest.approx(1.0, abs=0.03)


@pytest.fixture
def learning_store(migrated_db):
    return LearningStore(migrated_db)


@pytest.fixture
def scored_weeks(adjusted_state, store, outcome_store):
    """``n`` weekly forecasts that each missed by ~1.1 predicted sigmas, alternating up/down.

    Typical misses of 1.1 sigma (vs 0.67 for a calibrated model) imply a multiplier ~1.65,
    which shrinkage keeps below the 1.5 bound at small sample sizes.
    """

    def _make(n, start=0):
        snaps = []
        for i in range(start, start + n):
            snap = snapshot_as_of(adjusted_state, week(i))
            final = 3.0 if i % 2 else -3.0
            save_scored(store, outcome_store, snap, final, amplitude=0.6)
            snaps.append(snap)
        return snaps

    return _make


def _update(ticker, now, store, outcome_store, learning_store):
    return update_calibration(
        ticker,
        now=now,
        snapshot_store=store,
        outcome_store=outcome_store,
        learning_store=learning_store,
    )


class TestStoredCalibration:
    def test_observation_uses_the_uncalibrated_baseline(self, scored_weeks, outcome_store):
        (snap,) = scored_weeks(1)
        obs = observation_from(snap, outcome_store.get(snap.forecast_id))
        basis = outcome_store.get(snap.forecast_id).actual_close_on_forecast_basis
        assert obs.residual_log == pytest.approx(
            math.log(basis / snap.shadow_baseline["p50_price"])
        )
        assert obs.predicted_sigma_log == pytest.approx(
            snap.shadow_baseline["weekly_vol_pct"] / 100
        )

    def test_seven_bad_weeks_change_nothing(
        self, scored_weeks, store, outcome_store, learning_store
    ):
        snaps = scored_weeks(7)
        update = _update("RELIANCE", evaluated_at(snaps[-1]), store, outcome_store, learning_store)
        assert not update.changed and update.stored is None
        assert learning_store.calibration_history("RELIANCE") == []

    def test_eighth_week_stores_an_audited_version(
        self, scored_weeks, store, outcome_store, learning_store
    ):
        snaps = scored_weeks(8)
        now = evaluated_at(snaps[-1])
        update = _update("RELIANCE", now, store, outcome_store, learning_store)

        assert update.changed
        params = learning_store.latest_calibration("RELIANCE", as_of=now)
        assert params == update.stored
        assert params.version == 1
        assert (params.previous_vol_multiplier, params.previous_p50_bias_shift_pct) == (1.0, 0.0)
        assert 1.0 < params.vol_multiplier <= 1.5
        assert params.n_samples == 8
        assert params.evidence["forecast_ids"] == [s.forecast_id for s in snaps]
        assert any("80% band" in r for r in params.reason)
        assert params.calibrator_version == "1"
        assert params.created_at == now

    def test_versions_only_advance_on_material_change(
        self, scored_weeks, store, outcome_store, learning_store
    ):
        snaps = scored_weeks(8)
        now = evaluated_at(snaps[-1])
        _update("RELIANCE", now, store, outcome_store, learning_store)
        assert not _update("RELIANCE", now, store, outcome_store, learning_store).changed

        more = scored_weeks(8, start=8)
        second = _update("RELIANCE", evaluated_at(more[-1]), store, outcome_store, learning_store)
        assert second.changed and second.stored.version == 2
        first = learning_store.calibration_history("RELIANCE")[0]
        assert second.stored.previous_vol_multiplier == first.vol_multiplier
        assert second.stored.vol_multiplier > first.vol_multiplier  # more evidence, less shrinkage

    def test_point_in_time(self, scored_weeks, store, outcome_store, learning_store):
        snaps = scored_weeks(8)
        # evaluated just before the eighth outcome existed: only seven observations
        early = evaluated_at(snaps[-1]) - timedelta(minutes=1)
        assert not _update("RELIANCE", early, store, outcome_store, learning_store).changed
        now = evaluated_at(snaps[-1])
        _update("RELIANCE", now, store, outcome_store, learning_store)
        assert learning_store.latest_calibration("RELIANCE", as_of=early) is None
        assert learning_store.latest_calibration("RELIANCE", as_of=now).version == 1

    def test_versions_must_be_sequential(self, scored_weeks, store, outcome_store, learning_store):
        snaps = scored_weeks(8)
        update = _update("RELIANCE", evaluated_at(snaps[-1]), store, outcome_store, learning_store)
        with pytest.raises(Exception, match="must be version 2"):
            learning_store.save_calibration(update.stored)

    @pytest.mark.parametrize(
        "sql",
        ["UPDATE calibration_params SET vol_multiplier = 1.0", "DELETE FROM calibration_params"],
    )
    def test_calibration_history_is_immutable(
        self, scored_weeks, store, outcome_store, learning_store, migrated_db, sql
    ):
        snaps = scored_weeks(8)
        _update("RELIANCE", evaluated_at(snaps[-1]), store, outcome_store, learning_store)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            migrated_db.execute(sql)


def test_normal_median_abs_constant():
    assert NormalDist().inv_cdf(0.75) == pytest.approx(NORMAL_MEDIAN_ABS, abs=1e-4)


class TestIndependentEvidence:
    def test_overlapping_daily_forecasts_count_once(
        self, adjusted_state, store, outcome_store, learning_store
    ):
        """Eight daily forecasts in one bad fortnight are not eight independent weeks."""
        snaps = []
        day = week(0)
        while len(snaps) < 8:
            if day.weekday() < 5:
                snap = snapshot_as_of(adjusted_state, day)
                save_scored(store, outcome_store, snap, 3.0 if len(snaps) % 2 else -3.0)
                snaps.append(snap)
            day += timedelta(days=1)

        update = _update("RELIANCE", evaluated_at(snaps[-1]), store, outcome_store, learning_store)
        assert not update.changed
        assert update.estimate.n_samples == 2  # the newest, and the first one it does not overlap
        assert len(update.estimate.evidence["overlapping_forecasts_excluded"]) == 6
        assert snaps[-1].forecast_id in update.estimate.evidence["forecast_ids"]

    def test_tiny_configured_minimum_does_not_crash(self, monkeypatch):
        monkeypatch.setenv("CALIBRATION_MIN_SAMPLES", "1")
        get_settings.cache_clear()
        est = estimate_calibration(_observations(1, 2.0))
        assert est.vol_multiplier == 1.0
