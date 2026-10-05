"""Simulated vs live: every snapshot records whether it is a live forecast or a backtest
week, older snapshots get an inferred source, and the track record keeps the two apart."""

import sqlite3
from datetime import timedelta

import pytest
from alembic.config import Config
from typer.testing import CliRunner

from alembic import command
from stock_analysis.backtest.inputs import DISABLED_IN_BACKTEST
from stock_analysis.cli import app
from stock_analysis.database import Database, ForecastSnapshotStore
from stock_analysis.database.forecast_store import (
    ForecastSnapshotError,
    SnapshotIntegrityError,
    _daily_values,
    _decision_values,
    _shadow_values,
    _snapshot_values,
    _storage_hash,
)
from stock_analysis.langgraph.workflow import format_prior_context
from stock_analysis.learning import build_evaluation, render_track_records
from stock_analysis.review import render_last_forecast_vs_actual
from stock_analysis.schemas.memory import MemoryContext, OutcomeMetrics, TrackRecord, WindowMetrics
from stock_analysis.schemas.scorecard import HitRate
from stock_analysis.schemas.snapshot import (
    LEGACY_BACKTEST_REASONS,
    canonical_json,
    resolve_forecast_source,
)
from stock_analysis.snapshots.report import (
    _NOT_LOADED,
    SIMULATED_NOTE,
    _benchmark_lines,
    _track_record_lines,
)
from tests.forecast_helpers import ALEMBIC_DIR
from tests.learning_helpers import evaluated_at, save_scored, score, snapshot_as_of, week
from tests.phase19.test_acceptance import _invoke, app_db, market  # noqa: F401

QUANT_ONLY_BACKTEST = dict.fromkeys(
    ("technical", "fundamental", "sentiment", "context"), "quant-only backtest (LLM disabled)"
)


def _legacy(snapshot):
    """``snapshot`` as stored before revision 007: no source."""
    return snapshot.model_copy(update={"source": None})


class TestSnapshotSource:
    def test_live_by_default(self, adjusted_state):
        snap = snapshot_as_of(adjusted_state, week(0))
        assert (snap.source, snap.effective_source, snap.source_inferred) == ("live", "live", False)

    def test_backtest_from_graph_state(self, adjusted_state):
        snap = snapshot_as_of(adjusted_state, week(0), forecast_source="backtest")
        assert snap.source == snap.effective_source == "backtest"

    def test_source_is_not_part_of_the_input_hash(self, adjusted_state):
        live = snapshot_as_of(adjusted_state, week(0))
        backtest = snapshot_as_of(adjusted_state, week(0), forecast_source="backtest")
        assert live.data_snapshot_id == backtest.data_snapshot_id

    def test_corrections_keep_the_source(self, adjusted_state):
        snap = snapshot_as_of(adjusted_state, week(0), forecast_source="backtest")
        fixed = snap.corrected(reason="typo", corrected_at=snap.made_at)
        assert fixed.source == "backtest"


class TestLegacyInference:
    @pytest.mark.parametrize(
        ("disabled", "expected"),
        [
            ({}, "live"),
            (dict.fromkeys(QUANT_ONLY_BACKTEST, "quant-only run (LLM disabled)"), "live"),
            (QUANT_ONLY_BACKTEST, "backtest"),
            (DISABLED_IN_BACKTEST, "backtest"),
        ],
    )
    def test_from_disabled_analyst_reasons(self, disabled, expected):
        inputs = {"disabled_analysts": disabled} if disabled else {}
        assert resolve_forecast_source(None, inputs) == (expected, True)

    def test_stored_source_wins(self):
        inputs = {"disabled_analysts": QUANT_ONLY_BACKTEST}
        assert resolve_forecast_source("live", inputs) == ("live", False)

    def test_current_backtest_reasons_are_recognised(self):
        assert set(DISABLED_IN_BACKTEST.values()) <= LEGACY_BACKTEST_REASONS

    def test_unknown_source_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown forecast source"):
            resolve_forecast_source("paper", {})


class TestStore:
    def test_round_trip(self, store, adjusted_state):
        snap = snapshot_as_of(adjusted_state, week(0), forecast_source="backtest")
        store.save(snap)
        loaded = store.get(snap.forecast_id)
        assert loaded == snap and loaded.source == "backtest"

    def test_hash_covers_the_source(self, store, migrated_db, adjusted_state):
        snap = snapshot_as_of(adjusted_state, week(0))
        store.save(snap)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            migrated_db.execute("UPDATE forecast_snapshots SET source = 'backtest'")
        migrated_db.execute("DROP TRIGGER forecast_snapshots_no_update")
        migrated_db.execute("UPDATE forecast_snapshots SET source = 'backtest'")
        with pytest.raises(SnapshotIntegrityError):
            store.get(snap.forecast_id)

    def test_legacy_snapshot_is_stored_without_source(self, store, adjusted_state):
        snap = _legacy(
            snapshot_as_of(adjusted_state, week(0), disabled_analysts=DISABLED_IN_BACKTEST)
        )
        store.save(snap)
        loaded = store.get(snap.forecast_id)
        assert loaded.source is None
        assert (loaded.effective_source, loaded.source_inferred) == ("backtest", True)

    def test_hash_without_source_is_the_old_hash(self, adjusted_state):
        snap = _legacy(snapshot_as_of(adjusted_state, week(0)))
        parts = (_snapshot_values(snap), _decision_values(snap), _daily_values(snap))
        shadow = _shadow_values(snap)
        assert _storage_hash(*parts, shadow, None) == _storage_hash(*parts, shadow)


def _upgrade(path, revision):
    cfg = Config()
    cfg.set_main_option("script_location", str(ALEMBIC_DIR))
    cfg.attributes["database_url"] = f"sqlite:///{path}"
    command.upgrade(cfg, revision)


class TestMigration:
    def test_rows_stored_before_007_load_with_an_inferred_source(self, tmp_path, adjusted_state):
        path = tmp_path / "old.db"
        _upgrade(path, "006")
        db = Database(path)
        live = snapshot_as_of(adjusted_state, week(0))
        backtest = snapshot_as_of(adjusted_state, week(1), disabled_analysts=QUANT_ONLY_BACKTEST)
        with pytest.raises(ForecastSnapshotError, match="no source column"):
            ForecastSnapshotStore(db).ensure_schema()
        for snap in (_legacy(live), _legacy(backtest)):
            # What the store wrote before revision 007
            row = _snapshot_values(snap)
            parts = (row, _decision_values(snap), _daily_values(snap), _shadow_values(snap))
            columns = (
                "forecast_id, root_forecast_id, version, supersedes_forecast_id, "
                "correction_reason, corrected_at, ticker, resolved_symbol, company_name, made_at, "
                "as_of_date, target_date, horizon_trading_days, last_close, prob_up, prob_flat, "
                "prob_down, expected_return_pct, p10_price, p50_price, p90_price, "
                "adjustment_applied, fallback_to_quant, market_regime, risk_category, "
                "data_snapshot_id, calibration_version, model_versions_json, "
                "prompt_versions_json, quant_baseline_json, final_forecast_json, "
                "analyst_reports_json, critic_json, data_inputs_json, data_quality_json"
            )
            with db.transaction() as conn:
                conn.execute(
                    f"INSERT INTO forecast_snapshots ({columns}, snapshot_hash) "
                    f"VALUES ({', '.join('?' for _ in range(len(row) + 1))})",
                    (*row, _storage_hash(*parts)),
                )
                conn.executemany(
                    "INSERT INTO forecast_decisions (forecast_id, sequence, decision_engine, "
                    "decision_model, decision_model_version, decision_type, decision, confidence, "
                    "rationale, evidence_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(snap.forecast_id, *d) for d in parts[1]],
                )
                conn.executemany(
                    "INSERT INTO forecast_daily_predictions (forecast_id, day_index, target_date, "
                    "predicted_return_pct, p10_price, p50_price, p90_price, prob_up, source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(snap.forecast_id, *p) for p in parts[2]],
                )
                conn.execute(
                    "INSERT INTO shadow_forecasts (forecast_id, calibration_version, "
                    "calibration_json, uncalibrated_baseline_json) VALUES (?, ?, ?, ?)",
                    (snap.forecast_id, *parts[3]),
                )
        db.close()

        _upgrade(path, "head")
        db = Database(path)
        store = ForecastSnapshotStore(db)
        store.ensure_schema()
        old_live, old_backtest = store.get(live.forecast_id), store.get(backtest.forecast_id)
        assert old_live.source is None and old_backtest.source is None
        assert (old_live.effective_source, old_backtest.effective_source) == ("live", "backtest")
        assert canonical_json(old_live.data_inputs) == canonical_json(live.data_inputs)
        store.save(snapshot_as_of(adjusted_state, week(2), forecast_source="backtest"))
        assert [s.effective_source for s in store.list_for_ticker(live.ticker)] == [
            "backtest",
            "backtest",
            "live",
        ]
        db.close()


@pytest.fixture
def mixed_history(adjusted_state, store, outcome_store):
    """Two backtest weeks (one stored before sources were recorded) and two live weeks."""
    snaps = [
        snapshot_as_of(adjusted_state, week(0), forecast_source="backtest"),
        _legacy(snapshot_as_of(adjusted_state, week(1), disabled_analysts=DISABLED_IN_BACKTEST)),
        snapshot_as_of(adjusted_state, week(2)),
        snapshot_as_of(adjusted_state, week(3)),
    ]
    for snap, pct in zip(snaps, (3.8, -1.0, 0.1, 2.0), strict=True):
        save_scored(store, outcome_store, snap, pct, amplitude=0.6)
    return snaps


def _as_of(snaps):
    return evaluated_at(snaps[-1])


class TestEvaluationBySource:
    def test_one_track_record_per_source(self, mixed_history, store, outcome_store):
        report = build_evaluation(store, outcome_store, as_of=_as_of(mixed_history))
        assert [(r.source, r.n_forecasts) for r in report.track_records] == [
            ("live", 2),
            ("backtest", 2),
        ]
        assert report.record_for("backtest").n_inferred_source == 1
        assert report.record_for("live").n_inferred_source == 0
        assert report.n_forecasts == 4
        assert report.sources == {"live": 2, "backtest": 2}
        # Probability calibration pools both sources
        assert {c.n_forecasts for c in report.probability_calibration} == {4}

    def test_windows_hold_only_their_source(self, mixed_history, store, outcome_store):
        report = build_evaluation(store, outcome_store, as_of=_as_of(mixed_history))
        live, backtest = (report.record_for(s).windows[-1] for s in ("live", "backtest"))
        assert (live.n, backtest.n) == (2, 2)

    def test_only_backtest(self, mixed_history, store, outcome_store):
        report = build_evaluation(store, outcome_store, as_of=evaluated_at(mixed_history[1]))
        assert [r.source for r in report.track_records] == ["backtest"]
        assert report.record_for("live") is None

    def test_nothing_scored(self, store, outcome_store, mixed_history):
        report = build_evaluation(store, outcome_store, as_of=mixed_history[0].made_at)
        assert [(r.source, r.n_forecasts) for r in report.track_records] == [(None, 0)]
        assert "No scored forecasts yet" in render_track_records(report)

    def test_rendered_separately(self, mixed_history, store, outcome_store):
        text = render_track_records(
            build_evaluation(store, outcome_store, as_of=_as_of(mixed_history))
        )
        live, backtest = text.split("## Forecast track record: backtest (simulated) forecasts")
        assert "## Forecast track record: live forecasts" in live
        assert SIMULATED_NOTE not in live
        assert SIMULATED_NOTE in backtest
        assert "1 of these were stored before forecasts recorded their source" in backtest

    def test_overlaps_are_dropped_within_a_source(self, adjusted_state, store, outcome_store):
        live = snapshot_as_of(adjusted_state, week(0))
        # A backtest run later whose first week overlaps the live forecast's window
        backtest = snapshot_as_of(
            adjusted_state, week(0) + timedelta(days=2), forecast_source="backtest"
        )
        for snap, pct in ((live, 1.0), (backtest, -1.0)):
            save_scored(store, outcome_store, snap, pct, amplitude=0.6)
        report = build_evaluation(store, outcome_store, as_of=evaluated_at(backtest))
        assert [
            (r.source, r.n_forecasts, r.overlapping_excluded) for r in report.track_records
        ] == [("live", 1, 0), ("backtest", 1, 0)]
        # The pooled probability calibration still counts overlapping weeks once
        assert {c.n_forecasts for c in report.probability_calibration} == {1}
        assert report.sources == {"backtest": 1}

    def test_review_output_labels_backtest_weeks(self, adjusted_state):
        snap = snapshot_as_of(adjusted_state, week(0), forecast_source="backtest")
        outcome = score(snap, 1.0)
        assert "a simulated backtest week" in render_last_forecast_vs_actual(snap, outcome)
        live = snapshot_as_of(adjusted_state, week(0))
        live_outcome = score(live, 1.0)
        assert "simulated" not in render_last_forecast_vs_actual(live, live_outcome)

    def test_scorecards_count_sources(self, mixed_history, outcome_store, migrated_db):
        from stock_analysis.database import LearningStore
        from stock_analysis.learning import build_scorecards, render_scorecards

        cards = build_scorecards(
            outcome_store, LearningStore(migrated_db), as_of=_as_of(mixed_history)
        )
        assert cards.sources == {"live": 2, "backtest": 2}
        assert "Sources pooled here: 2 live, 2 backtest (simulated)" in render_scorecards(cards)


def _metrics(source, n):
    window = WindowMetrics(
        label="All time", n=n, direction=HitRate(n=n, hits=1), llm_value_added=0.01,
        n_adjusted=n, finding="too few",
    )  # fmt: skip
    return OutcomeMetrics(source=source, n_forecasts=n, min_samples=20, windows=[window])


def _context(track: TrackRecord) -> MemoryContext:
    return MemoryContext(loaded=True, track_record=track, learning_loaded=["track_record"])


def _track(**update) -> TrackRecord:
    return TrackRecord(
        ticker="RELIANCE",
        forecasts_made=5,
        forecasts_scored=5,
        forecasts_awaiting_outcome=0,
        **update,
    )


class TestForecastContext:
    def test_report_shows_each_source(self):
        track = _track(outcome_metrics_by_source=[_metrics("live", 2), _metrics("backtest", 3)])
        text = "\n".join(_track_record_lines(_context(track), "mixed"))
        live, backtest = text.split("**Backtest (simulated) forecasts**")
        assert "**Live forecasts**" in live and "2 forecasts" in live
        assert SIMULATED_NOTE in backtest and "3 forecasts" in backtest

    def test_older_pooled_context_still_renders(self):
        track = _track(outcome_metrics=_metrics(None, 2))
        text = "\n".join(_track_record_lines(_context(track), "mixed"))
        assert "**All time**: 2 forecasts" in text
        assert "forecasts**" not in text  # no source heading
        assert _benchmark_lines(_context(track)) == [_NOT_LOADED]  # no variants stored

    def test_prompt_marks_backtest_weeks(self):
        track = _track(outcome_metrics_by_source=[_metrics("backtest", 3)])
        prompt = "\n".join(format_prior_context(_context(track).model_dump(mode="json")))
        assert "Backtest (simulated) forecasts: simulated past weeks" in prompt
        assert "      All time: 3 forecasts" in prompt


class TestCommands:
    @pytest.fixture(autouse=True)
    def wide_console(self, monkeypatch):
        """Wide enough that the history table does not truncate cells."""
        monkeypatch.setattr("stock_analysis.cli.main.console.width", 200)

    def test_backtest_then_analyze_labels_both(self, app_db, market, tmp_path):  # noqa: F811
        _invoke("backtest", "--ticker", "RELIANCE", "--weeks", "4", "--no-llm")
        db = Database(app_db)
        sources = {s.source for s in ForecastSnapshotStore(db).list_for_ticker("RELIANCE")}
        assert sources == {"backtest"}

        evaluate = " ".join(_invoke("evaluate", "--ticker", "RELIANCE").split())
        assert "Forecast track record for RELIANCE: backtest (simulated) forecasts" in evaluate
        assert "for RELIANCE: live forecasts" not in evaluate

        report = tmp_path / "analysis.md"
        _invoke("analyze", "RELIANCE", "--no-llm", "-o", str(report))
        text = report.read_text()
        assert "- Source: live" in text
        assert "**Backtest (simulated) forecasts**" in text
        assert "a simulated backtest week" in text  # the last review was a backtest week

        history = _invoke("history", "--ticker", "RELIANCE")
        assert "backtest" in history and "live" in history
        db.close()

    def test_history_marks_inferred_sources(self, migrated_db, store, adjusted_state):
        store.save(
            _legacy(snapshot_as_of(adjusted_state, week(0), disabled_analysts=DISABLED_IN_BACKTEST))
        )
        result = CliRunner().invoke(app, ["history", "--ticker", adjusted_state.symbol])
        assert result.exit_code == 0, result.output
        assert "backtest*" in result.output
        assert "source inferred" in " ".join(result.output.split())
