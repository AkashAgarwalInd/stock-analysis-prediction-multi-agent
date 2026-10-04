"""Plan.md Phase 14 / §28, §31, §53: the rolling track record against the naive, uncalibrated
and calibrated quant benchmarks, and the `evaluate` / `calibration` commands."""

from datetime import UTC, date, datetime, time, timedelta

import pytest
from typer.testing import CliRunner

from stock_analysis.database import LearningStore
from stock_analysis.learning import (
    build_evaluation,
    build_track_record,
    evaluate_forecast,
    render_track_record,
)
from stock_analysis.schemas.learning import CalibrationParams
from stock_analysis.schemas.track_record import EvaluatedForecast, VariantScore
from tests.forecast_helpers import FakeLLM, run_graph
from tests.learning_helpers import evaluated_at, save_scored, score, snapshot_as_of, week


@pytest.fixture
def learning_store(migrated_db):
    return LearningStore(migrated_db)


@pytest.fixture
def calibration_v1(learning_store):
    params = CalibrationParams(
        ticker="RELIANCE",
        version=1,
        vol_multiplier=1.3,
        p50_bias_shift_pct=0.5,
        previous_vol_multiplier=1.0,
        previous_p50_bias_shift_pct=0.0,
        n_samples=12,
        reason=["Realized volatility exceeded the forecast in 9 of 12 weeks"],
        evidence={"forecast_ids": []},
        calibrator_version="1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    learning_store.save_calibration(params)
    return params


@pytest.fixture
def calibrated_state(
    price_history, initial_state, store, memory_store, learning_store, calibration_v1
):
    return run_graph(FakeLLM("valid"), initial_state, store, memory_store, learning_store)


class TestEvaluateForecast:
    def test_variants_match_the_outcome_scorer(self, calibrated_state):
        snap = snapshot_as_of(calibrated_state, week(0))
        outcome = score(snap, 3.8, amplitude=0.6)
        f = evaluate_forecast(snap, outcome)

        final, cal, uncal = (
            f.variants[v] for v in ("final", "quant_calibrated", "quant_uncalibrated")
        )
        assert final.brier == pytest.approx(outcome.brier)
        assert final.signed_error_pct == pytest.approx(outcome.signed_error_pct)
        assert final.in_80pct_band == outcome.in_80pct_band
        assert final.direction_correct == outcome.direction_correct
        assert final.pinball == pytest.approx(outcome.pinball["mean"])
        assert final.vol_ratio == pytest.approx(outcome.vol_ratio)
        assert cal.brier == pytest.approx(outcome.baseline_brier)
        assert cal.signed_error_pct == pytest.approx(outcome.baseline_signed_error_pct)
        assert cal.in_80pct_band == outcome.baseline_in_80pct_band
        assert cal.pinball == pytest.approx(outcome.baseline_pinball["mean"])
        assert uncal.brier == pytest.approx(outcome.uncalibrated_brier)
        assert uncal.in_80pct_band == outcome.uncalibrated_in_80pct_band
        assert uncal.pinball == pytest.approx(outcome.uncalibrated_pinball["mean"])
        assert f.calibration_value_added == pytest.approx(outcome.calibration_value_added)
        assert final.pit_percentile == pytest.approx(outcome.pit_percentile)
        assert f.calibrated
        assert f.adjusted == snap.final_forecast.adjustment_applied
        assert f.llm_value_added == pytest.approx(outcome.llm_value_added)
        # The calibrated band (multiplier 1.3) is wider than the uncalibrated one
        assert cal.sharpness_pct > uncal.sharpness_pct
        q = snap.quant_baseline
        assert cal.sharpness_pct == pytest.approx(
            (q["p90_price"] - q["p10_price"]) / snap.last_close * 100
        )

    def test_naive_flat_benchmark(self, adjusted_state):
        snap = snapshot_as_of(adjusted_state, week(0))
        for final_pct in (3.0, 0.0):
            outcome = score(snap, final_pct, amplitude=0.6)
            naive = evaluate_forecast(snap, outcome).variants["naive_flat"]
            assert naive.signed_error_pct == pytest.approx(outcome.actual_return_pct)
            assert naive.direction_correct == (outcome.realized_direction == "flat")
            assert naive.brier is None and naive.in_80pct_band is None and naive.pinball is None

    def test_rejects_an_unscored_or_foreign_outcome(self, adjusted_state):
        a, b = snapshot_as_of(adjusted_state, week(0)), snapshot_as_of(adjusted_state, week(1))
        with pytest.raises(ValueError):
            evaluate_forecast(a, score(b, 1.0))


AS_OF = datetime(2026, 10, 3, 12, tzinfo=UTC)


def _forecast(
    target: date,
    *,
    correct=True,
    llm=0.01,
    band=True,
    error=1.0,
    brier=0.4,
    pit=0.5,
    adjusted=True,
    calibrated=False,
):
    probabilistic = VariantScore(
        prob_up=0.5,
        prob_flat=0.3,
        prob_down=0.2,
        direction_correct=correct,
        signed_error_pct=error,
        in_80pct_band=band,
        brier=brier,
        pinball=0.5,
        vol_ratio=1.2,
        sharpness_pct=6.0,
        pit_percentile=pit,
    )
    return EvaluatedForecast(
        forecast_id=f"f{target}",
        ticker="RELIANCE",
        as_of_date=target - timedelta(days=7),
        target_date=target,
        realized_direction="up",
        actual_return_pct=error,
        llm_value_added=llm,
        calibration_value_added=0.0,
        adjusted=adjusted,
        calibrated=calibrated,
        variants={
            "naive_flat": VariantScore(direction_correct=False, signed_error_pct=error),
            "quant_uncalibrated": probabilistic,
            "quant_calibrated": probabilistic,
            "final": probabilistic,
        },
    )


def _weekly(n, **kw):
    """``n`` weekly forecasts, the newest targeting the Friday before AS_OF."""
    last = date(2026, 10, 2)
    return [_forecast(last - timedelta(weeks=i), **kw) for i in reversed(range(n))]


class TestRollingWindows:
    def test_window_membership(self):
        record = build_track_record(_weekly(40), as_of=AS_OF)
        assert record.n_forecasts == 40
        assert [(w.label, w.n) for w in record.windows] == [
            ("8 weeks", 8),
            ("26 weeks", 26),
            ("All time", 40),
        ]
        eight = record.windows[0]
        assert eight.start == date(2026, 8, 9)
        assert all(v.n == 8 for v in eight.variants)
        assert record.windows[2].start is None

    def test_variant_metrics(self):
        forecasts = _weekly(6, error=2.0) + _weekly(0)
        forecasts[0] = _forecast(forecasts[0].target_date, correct=False, band=False, error=-4.0)
        record = build_track_record(forecasts, as_of=AS_OF)
        all_time = {v.variant: v for v in record.windows[2].variants}
        final = all_time["final"]
        assert (final.direction.hits, final.direction.n) == (5, 6)
        assert final.coverage_80pct == pytest.approx(5 / 6)
        assert final.mean_signed_error_pct == pytest.approx((5 * 2.0 - 4.0) / 6)
        assert final.mean_abs_error_pct == pytest.approx((5 * 2.0 + 4.0) / 6)
        assert final.brier == pytest.approx(0.4)
        assert final.mean_sharpness_pct == pytest.approx(6.0)
        assert final.mean_vol_ratio == pytest.approx(1.2)
        naive = all_time["naive_flat"]
        assert naive.direction.hits == 0
        assert naive.brier is None and naive.coverage_80pct is None
        assert naive.mean_abs_error_pct == pytest.approx((5 * 2.0 + 4.0) / 6)

    def test_pit_distribution(self):
        pits = [0.05, 0.3, 0.6, 0.95, 0.97, 0.5]  # 0.5 is not below the median
        forecasts = [_forecast(f.target_date, pit=p) for f, p in zip(_weekly(6), pits, strict=True)]
        final = build_track_record(forecasts, as_of=AS_OF).windows[2].variants[-1]
        assert final.pit_below_p10 == pytest.approx(1 / 6)
        assert final.pit_below_p50 == pytest.approx(2 / 6)
        assert final.pit_above_p90 == pytest.approx(2 / 6)
        naive = build_track_record(forecasts, as_of=AS_OF).windows[2].variants[0]
        assert naive.pit_below_p10 is None

    def test_unadjusted_forecasts_get_no_llm_verdict(self):
        same = [_forecast(f.target_date, llm=0.0, adjusted=False) for f in _weekly(30)]
        window = build_track_record(same, as_of=AS_OF).windows[2]
        assert (window.n_adjusted, window.n_calibrated) == (0, 0)
        assert window.finding.startswith("no forecast was adjusted by the LLM")
        text = render_track_record(build_track_record(same, as_of=AS_OF))
        assert "none of the 30 forecasts was adjusted by the LLM" in text
        assert "no calibration was applied to any of the 30 forecasts" in text

    def test_adjusted_and_calibrated_counts(self):
        mixed = [
            _forecast(f.target_date, adjusted=i % 2 == 0, calibrated=i >= 4)
            for i, f in enumerate(_weekly(6))
        ]
        window = build_track_record(mixed, as_of=AS_OF).windows[2]
        assert (window.n_adjusted, window.n_calibrated) == (3, 2)
        text = render_track_record(build_track_record(mixed, as_of=AS_OF))
        assert "3 of 6 adjusted by the LLM" in text and "2 of 6 calibrated" in text

    def test_no_verdict_below_the_minimum(self):
        record = build_track_record(_weekly(10, llm=0.05), as_of=AS_OF)
        assert record.windows[2].finding.startswith("insufficient evidence: 10 forecasts")
        assert record.windows[2].llm_value_added.mean == pytest.approx(0.05)

    def test_verdicts_need_the_interval_to_exclude_zero(self):
        better = [
            _forecast(f.target_date, llm=0.02 + 0.01 * (i % 3)) for i, f in enumerate(_weekly(30))
        ]
        assert (
            build_track_record(better, as_of=AS_OF)
            .windows[2]
            .finding.startswith("the final forecast beat")
        )
        worse = [
            _forecast(f.target_date, llm=-0.02 - 0.01 * (i % 3)) for i, f in enumerate(_weekly(30))
        ]
        assert "was worse" in build_track_record(worse, as_of=AS_OF).windows[2].finding
        mixed = [_forecast(f.target_date, llm=0.05 * (-1) ** i) for i, f in enumerate(_weekly(30))]
        assert (
            build_track_record(mixed, as_of=AS_OF)
            .windows[2]
            .finding.startswith("no measurable difference")
        )

    def test_render(self):
        text = render_track_record(build_track_record(_weekly(30), as_of=AS_OF))
        assert "## Forecast track record" in text
        assert "### 8 weeks, targets from 2026-08-09: 8 forecasts" in text
        assert "### All time: 30 forecasts" in text
        for label in ("Naive flat", "Quant uncalibrated", "Quant calibrated", "Final forecast"):
            assert f"| {label} |" in text
        assert "LLM value added (calibrated-quant Brier − final Brier): +0.0100" in text
        assert "| 0% / 0% / 0% |" in text  # every PIT at the median

    def test_render_empty(self):
        text = render_track_record(build_track_record([], as_of=AS_OF))
        assert "No scored forecasts yet." in text


@pytest.fixture
def history(calibrated_state, adjusted_state, store, outcome_store):
    """Two calibrated and one uncalibrated scored week."""
    snaps = [
        snapshot_as_of(calibrated_state, week(0)),
        snapshot_as_of(calibrated_state, week(1)),
        snapshot_as_of(adjusted_state, week(2)),
    ]
    for snap, pct in zip(snaps, (3.8, -1.0, 0.1), strict=True):
        save_scored(store, outcome_store, snap, pct, amplitude=0.6)
    return snaps


def _later(snaps):
    return evaluated_at(snaps[-1]) + timedelta(days=1)


class TestBuildEvaluation:
    def test_from_stored_history(self, history, store, outcome_store):
        report = build_evaluation(store, outcome_store, as_of=_later(history))
        record = report.track_record
        assert record.n_forecasts == 3
        all_time = record.windows[2]
        assert all_time.n == 3
        assert all_time.calibration_value_added.n == 3
        final = next(c for c in report.probability_calibration if c.variant == "final")
        assert final.n_forecasts == 3 and final.events["up"].n == 3
        assert [c.variant for c in report.probability_calibration] == [
            "quant_uncalibrated",
            "quant_calibrated",
            "final",
        ]
        stored = [outcome_store.get(s.forecast_id) for s in history]
        assert final.brier == pytest.approx(sum(o.brier for o in stored) / 3)

    def test_counts_calibrated_forecasts(self, history, store, outcome_store):
        report = build_evaluation(store, outcome_store, as_of=_later(history))
        window = report.track_record.windows[2]
        assert window.n_calibrated == 2

    def test_point_in_time(self, history, store, outcome_store):
        before_last = datetime.combine(
            history[-1].target_date, time(9, 0), tzinfo=UTC
        )  # the last week has not been scored yet
        report = build_evaluation(store, outcome_store, as_of=before_last)
        assert report.track_record.n_forecasts == 2

    def test_overlapping_forecasts_count_once(self, history, adjusted_state, store, outcome_store):
        overlap = snapshot_as_of(adjusted_state, week(0) + timedelta(days=2))
        save_scored(store, outcome_store, overlap, 3.0, amplitude=0.6)
        record = build_evaluation(store, outcome_store, as_of=_later(history)).track_record
        assert (record.n_forecasts, record.overlapping_excluded) == (3, 1)

    def test_ticker_filter(self, history, store, outcome_store):
        report = build_evaluation(store, outcome_store, as_of=_later(history), ticker="TCS")
        assert report.track_record.n_forecasts == 0


class TestCli:
    def test_evaluate(self, history, migrated_db):
        from stock_analysis.cli import app

        args = ["evaluate", "--database", str(migrated_db.path), "--ticker", "reliance.ns"]
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "## Forecast track record for RELIANCE" in result.output
        assert "| Naive flat |" in result.output
        assert "## Probability calibration for RELIANCE" in result.output
        assert "#### P(up)" in result.output

    def test_calibration(self, history, migrated_db):
        from stock_analysis.cli import app

        result = CliRunner().invoke(app, ["calibration", "--database", str(migrated_db.path)])
        assert result.exit_code == 0, result.output
        assert "## Calibration versions for RELIANCE" in result.output
        assert "volatility multiplier 1.00 → 1.30" in result.output
        assert "Realized volatility exceeded the forecast" in result.output
        assert "### Quant uncalibrated" in result.output
        assert "### Final forecast" in result.output

    def test_calibration_without_versions(self, migrated_db):
        from stock_analysis.cli import app

        args = ["calibration", "--database", str(migrated_db.path), "--ticker", "TCS"]
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert "None stored: the quant baseline is used uncalibrated" in result.output
        assert "No scored forecasts yet." in result.output

    @pytest.mark.parametrize("command", ["evaluate", "calibration"])
    def test_tampered_forecast_is_reported(self, command, history, migrated_db):
        import sqlite3

        from stock_analysis.cli import app

        raw = sqlite3.connect(migrated_db.path)  # bypass the immutability trigger to tamper
        raw.execute("DROP TRIGGER forecast_snapshots_no_update")
        raw.execute("UPDATE forecast_snapshots SET p50_price = p50_price + 1")
        raw.commit()
        raw.close()
        result = CliRunner().invoke(app, [command, "--database", str(migrated_db.path)])
        assert result.exit_code == 1
        assert "Cannot read stored forecasts" in result.output

    @pytest.mark.parametrize("command", ["evaluate", "calibration", "scorecard"])
    def test_missing_and_unmigrated_databases(self, command, tmp_path):
        import sqlite3

        from stock_analysis.cli import app

        missing = CliRunner().invoke(app, [command, "--database", str(tmp_path / "nope.db")])
        assert missing.exit_code == 1 and "No database at" in missing.output
        path = tmp_path / "old.db"
        sqlite3.connect(path).close()
        old = CliRunner().invoke(app, [command, "--database", str(path)])
        assert old.exit_code == 1 and "stock-analysis migrate" in old.output
