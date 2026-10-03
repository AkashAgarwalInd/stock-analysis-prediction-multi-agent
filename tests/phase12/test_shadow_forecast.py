"""Plan.md §23: every forecast keeps its uncalibrated baseline (the shadow forecast)."""

import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np
import pytest

from stock_analysis.config.settings import get_settings
from stock_analysis.database import LearningStore
from stock_analysis.learning import compare_benchmarks
from stock_analysis.quant import QuantForecaster
from stock_analysis.review import render_last_forecast_vs_actual
from stock_analysis.schemas.learning import CalibrationParams
from stock_analysis.snapshots import build_forecast_snapshot, render_forecast_report
from stock_analysis.snapshots.reproduce import reproduce_quant_baseline
from tests.forecast_helpers import MADE_AT, FakeLLM, run_graph
from tests.learning_helpers import score, snapshot_as_of, week

CALIBRATION = {"vol_multiplier": 1.3, "p50_bias_shift_pct": 0.5}


@pytest.fixture
def learning_store(migrated_db):
    return LearningStore(migrated_db)


@pytest.fixture
def calibrated(learning_store):
    params = CalibrationParams(
        ticker="RELIANCE",
        version=1,
        **CALIBRATION,
        previous_vol_multiplier=1.0,
        previous_p50_bias_shift_pct=0.0,
        n_samples=12,
        reason=["test calibration"],
        evidence={"forecast_ids": []},
        calibrator_version="1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    learning_store.save_calibration(params)
    return params


@pytest.fixture
def calibrated_run(price_history, initial_state, store, memory_store, learning_store, calibrated):
    llm = FakeLLM("valid")
    return llm, run_graph(llm, initial_state, store, memory_store, learning_store)


class TestQuantCalibration:
    @pytest.fixture
    def closes(self):
        rng = np.random.default_rng(1)
        return 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 300)))

    def _run(self, closes, params=None):
        return QuantForecaster(seed=42, n_paths=10_000).forecast_with_daily_path(closes, params)[0]

    def test_identity_calibration_is_bit_identical(self, closes):
        assert self._run(closes) == self._run(
            closes, {"vol_multiplier": 1.0, "p50_bias_shift_pct": 0.0}
        )

    def test_vol_multiplier_widens_the_band(self, closes):
        base = self._run(closes)
        wide = self._run(closes, {"vol_multiplier": 1.3})
        assert wide.weekly_vol_pct == pytest.approx(base.weekly_vol_pct * 1.3)
        assert wide.p10_price < base.p10_price and wide.p90_price > base.p90_price
        assert wide.p50_price == pytest.approx(base.p50_price, rel=2e-3)

    def test_bias_shift_moves_the_median(self, closes):
        base = self._run(closes)
        shifted = self._run(closes, {"p50_bias_shift_pct": 1.0})
        assert shifted.p50_price == pytest.approx(base.p50_price * 1.01, rel=1e-9)

    def test_rejects_non_positive_multiplier(self, closes):
        with pytest.raises(ValueError):
            self._run(closes, {"vol_multiplier": 0.0})


class TestShadowInGraph:
    def test_calibration_applied_and_uncalibrated_kept(self, calibrated_run, adjusted_state):
        llm, state = calibrated_run
        assert state.calibration == {"version": 1, **CALIBRATION}
        # the shadow is exactly what an uncalibrated run produces
        assert state.quant_baseline_uncalibrated == adjusted_state.quant_baseline
        quant, shadow = state.quant_baseline, state.quant_baseline_uncalibrated
        assert quant["p10_price"] < shadow["p10_price"] and quant["p90_price"] > shadow["p90_price"]
        # the predictor started from the calibrated baseline
        assert f'"p90_price": {quant["p90_price"]}' in llm.predictor_prompts[0]

    def test_snapshot_persists_all_three(self, calibrated_run, store):
        _, state = calibrated_run
        snap = store.get(state.forecast_id)
        assert snap.calibration_version == 1
        assert snap.calibration.model_dump() == {"version": 1, **CALIBRATION}
        assert snap.quant_baseline == state.quant_baseline
        assert snap.uncalibrated_baseline == state.quant_baseline_uncalibrated
        assert snap.final_forecast.quant_baseline == state.quant_baseline

    def test_both_baselines_are_reproducible(self, calibrated_run, store):
        _, state = calibrated_run
        snap = store.get(state.forecast_id)
        calibrated, _ = reproduce_quant_baseline(snap, store)
        uncalibrated, _ = reproduce_quant_baseline(snap, store, calibrated=False)
        assert asdict(calibrated) == snap.quant_baseline
        assert asdict(uncalibrated) == snap.uncalibrated_baseline

    def test_report_shows_the_shadow_forecast(self, calibrated_run, store):
        _, state = calibrated_run
        report = render_forecast_report(store.get(state.forecast_id))
        assert "Calibration v1: volatility multiplier 1.30, P50 shift +0.50%" in report
        assert "Before calibration (shadow forecast)" in report

    def test_learning_disabled_keeps_the_model_uncalibrated(
        self,
        price_history,
        initial_state,
        store,
        memory_store,
        learning_store,
        calibrated,
        monkeypatch,
        adjusted_state,
    ):
        monkeypatch.setenv("LEARNING_ENABLED", "false")
        get_settings.cache_clear()
        state = run_graph(FakeLLM("valid"), initial_state, store, memory_store, learning_store)
        assert state.calibration is None
        assert state.quant_baseline == state.quant_baseline_uncalibrated
        assert state.quant_baseline == adjusted_state.quant_baseline
        snap = store.get(state.forecast_id)
        assert snap.calibration_version == 0
        assert snap.uncalibrated_baseline == snap.quant_baseline

    def test_uncalibrated_forecast_still_stores_its_shadow(self, store, snapshot, migrated_db):
        store.save(snapshot)
        assert store.get(snapshot.forecast_id).uncalibrated_baseline == snapshot.quant_baseline
        row = migrated_db.fetchone(
            "SELECT calibration_version FROM shadow_forecasts WHERE forecast_id = ?",
            (snapshot.forecast_id,),
        )
        assert row["calibration_version"] == 0

    def test_snapshots_without_a_shadow_row_still_load(self, store, snapshot):
        legacy = snapshot.model_copy(update={"uncalibrated_baseline": None})
        store.save(legacy)
        loaded = store.get(legacy.forecast_id)
        assert loaded.uncalibrated_baseline is None
        assert loaded.shadow_baseline == loaded.quant_baseline

    @pytest.mark.parametrize(
        "sql",
        [
            "UPDATE shadow_forecasts SET uncalibrated_baseline_json = '{}'",
            "DELETE FROM shadow_forecasts",
        ],
    )
    def test_shadow_is_immutable(self, store, snapshot, migrated_db, sql):
        store.save(snapshot)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            migrated_db.execute(sql)

    def test_calibrated_snapshot_requires_its_shadow(self, adjusted_state):
        state = adjusted_state.model_copy(
            update={
                "calibration": {"version": 1, **CALIBRATION},
                "quant_baseline_uncalibrated": None,
            }
        )
        with pytest.raises(ValueError, match="uncalibrated baseline"):
            build_forecast_snapshot(state, made_at=MADE_AT)


class TestShadowScoring:
    @pytest.fixture
    def calibrated_snapshot(self, calibrated_run, adjusted_state):
        _, state = calibrated_run
        return snapshot_as_of(state, week(0))

    def test_three_way_comparison_is_scored_and_stored(
        self, calibrated_snapshot, store, outcome_store
    ):
        snap = calibrated_snapshot
        store.save(snap)
        outcome = score(snap, 3.8, amplitude=0.6)  # outside the uncalibrated band only
        outcome_store.save(outcome)
        stored = outcome_store.get(snap.forecast_id)

        assert stored == outcome
        assert stored.calibration_version == 1
        assert stored.uncalibrated_in_80pct_band is False
        assert stored.baseline_in_80pct_band is True
        assert stored.calibration_value_added == pytest.approx(
            stored.uncalibrated_loss - stored.baseline_loss
        )
        assert stored.llm_value_added == pytest.approx(stored.baseline_loss - stored.final_loss)
        text = render_last_forecast_vs_actual(snap, stored)
        assert "Uncalibrated quant loss (brier)" in text

    def test_uncalibrated_forecast_scores_identically(self, adjusted_state):
        snap = snapshot_as_of(adjusted_state, week(0))
        outcome = score(snap, 2.0)
        assert outcome.calibration_version == 0
        assert outcome.uncalibrated_loss == outcome.baseline_loss
        assert outcome.calibration_value_added == 0.0
        assert "Calibration: none applied" in render_last_forecast_vs_actual(snap, outcome)

    def test_benchmarks_report_whether_calibration_helped(
        self, calibrated_snapshot, adjusted_state
    ):
        plain = snapshot_as_of(adjusted_state, week(1))
        outcomes = [score(calibrated_snapshot, 3.8), score(plain, 2.0)]
        result = compare_benchmarks(outcomes)
        assert (result.n, result.n_calibrated) == (2, 1)
        assert result.mean_final_loss is not None
        assert "forecasts with calibration applied" in result.finding
        assert compare_benchmarks([outcomes[1]]).finding.startswith("No scored forecast used")
