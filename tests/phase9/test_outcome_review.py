"""Plan.md Phase 9 "done when": matured forecasts become scored and stored."""

import sqlite3
from datetime import UTC, date, datetime, timedelta

import pytest
from typer.testing import CliRunner

from stock_analysis.database import OutcomeStoreError
from stock_analysis.review import (
    NIFTY_50_SYMBOL,
    OutcomeReviewer,
    YFinancePriceSource,
    render_last_forecast_vs_actual,
)
from stock_analysis.review.report import SINGLE_OBSERVATION_NOTE
from stock_analysis.schemas.outcome import OutcomeStatus
from stock_analysis.snapshots import build_forecast_snapshot
from tests.forecast_helpers import MADE_AT
from tests.outcome_helpers import NOW_MATURED, FakePriceSource, nifty_series, path_series

UP_PATH = [0.5, 1.0, 1.8, 2.4, 3.0]


@pytest.fixture
def stored(store, snapshot):
    store.save(snapshot)
    return snapshot


def _source(snapshot, path=UP_PATH):
    return FakePriceSource(
        {
            snapshot.resolved_symbol: path_series(snapshot, path),
            NIFTY_50_SYMBOL: nifty_series(snapshot, 1.0),
        }
    )


class TestReviewer:
    def test_matured_forecast_is_scored_and_stored(self, stored, store, outcome_store):
        source = _source(stored)
        results = OutcomeReviewer(store, outcome_store, source).review_matured(NOW_MATURED)

        assert [r.status for r in results] == [OutcomeStatus.SCORED]
        saved = outcome_store.get(stored.forecast_id)
        assert saved == results[0]
        assert len(saved.daily) == stored.horizon_trading_days
        # only the forecast window was requested, never later prices
        assert set(source.calls) == {
            (stored.resolved_symbol, stored.as_of_date, stored.target_date),
            (NIFTY_50_SYMBOL, stored.as_of_date, stored.target_date),
        }
        # scored once: a second review finds nothing to do
        assert OutcomeReviewer(store, outcome_store, source).review_matured(NOW_MATURED) == []

    def test_unmatured_forecast_is_left_alone(self, stored, store, outcome_store):
        source = _source(stored)
        before_close = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)
        assert OutcomeReviewer(store, outcome_store, source).review_matured(before_close) == []
        assert source.calls == []

    def test_unresolved_is_retried_until_data_arrives(self, stored, store, outcome_store):
        empty = FakePriceSource({})
        first = OutcomeReviewer(store, outcome_store, empty).review_matured(NOW_MATURED)
        assert [r.status for r in first] == [OutcomeStatus.UNRESOLVED]
        assert outcome_store.get(stored.forecast_id) is None

        later = OutcomeReviewer(store, outcome_store, _source(stored)).review_matured(NOW_MATURED)
        assert [r.status for r in later] == [OutcomeStatus.SCORED]

    def test_invalid_is_stored_and_not_retried(self, stored, store, outcome_store):
        jump = _source(stored, [0, -45, -45, -45, -45])
        results = OutcomeReviewer(store, outcome_store, jump).review_matured(NOW_MATURED)
        assert outcome_store.get(stored.forecast_id).status == OutcomeStatus.INVALID
        assert results[0].invalid_reason
        assert (
            OutcomeReviewer(store, outcome_store, _source(stored)).review_matured(NOW_MATURED) == []
        )

    def test_only_original_forecasts_are_evaluated(self, stored, store, outcome_store):
        store.record_correction(
            stored.forecast_id, reason="name fix", corrected_at=datetime(2026, 9, 30, tzinfo=UTC)
        )
        results = OutcomeReviewer(store, outcome_store, _source(stored)).review_matured(NOW_MATURED)
        assert [r.forecast_id for r in results] == [stored.forecast_id]

    def test_ticker_filter(self, stored, store, outcome_store):
        reviewer = OutcomeReviewer(store, outcome_store, _source(stored))
        assert reviewer.review_matured(NOW_MATURED, ticker="TCS") == []
        assert len(reviewer.review_matured(NOW_MATURED, ticker="RELIANCE")) == 1


class TestOutcomeStorage:
    def test_outcomes_are_immutable(self, stored, store, outcome_store, migrated_db):
        OutcomeReviewer(store, outcome_store, _source(stored)).review_matured(NOW_MATURED)
        for statement in (
            "UPDATE forecast_outcomes SET actual_close = 1",
            "DELETE FROM forecast_outcomes",
            "UPDATE outcome_daily SET close = 1",
        ):
            with pytest.raises(sqlite3.DatabaseError, match="immutable"):
                migrated_db.execute(statement)
        with pytest.raises(OutcomeStoreError):
            outcome_store.save(outcome_store.get(stored.forecast_id))

    def test_unresolved_cannot_be_stored(self, stored, store, outcome_store):
        unresolved = OutcomeReviewer(store, outcome_store, FakePriceSource({})).review_matured(
            NOW_MATURED
        )[0]
        with pytest.raises(OutcomeStoreError, match="not stored"):
            outcome_store.save(unresolved)

    def test_track_record_counts_scored_forecasts(self, stored, store, outcome_store, memory_store):
        before = memory_store.track_record(
            "RELIANCE", before=NOW_MATURED, as_of_date=date(2026, 10, 8)
        )
        assert (before.forecasts_scored, before.forecasts_awaiting_outcome) == (0, 1)

        OutcomeReviewer(store, outcome_store, _source(stored)).review_matured(NOW_MATURED)
        after = memory_store.track_record(
            "RELIANCE", before=NOW_MATURED + timedelta(minutes=1), as_of_date=date(2026, 10, 8)
        )
        assert (after.forecasts_scored, after.forecasts_awaiting_outcome) == (1, 0)


class TestLastForecastVsActual:
    def test_scored_section(self, stored, store, outcome_store):
        outcome = OutcomeReviewer(store, outcome_store, _source(stored)).review_matured(
            NOW_MATURED
        )[0]
        text = render_last_forecast_vs_actual(stored, outcome)
        final = stored.final_forecast

        assert "## Last forecast vs actual" in text
        assert f"| P50 / close | ₹{final.p50_price:.2f} | ₹{outcome.actual_close:.2f} |" in text
        assert f"₹{final.p10_price:.2f}–₹{final.p90_price:.2f}" in text
        assert f"P(up) {final.prob_up * 100:.1f}%" in text
        assert "| Weekly return |" in text and "+3.00% |" in text
        assert "| Nifty 50 return | — | +1.00% |" in text
        assert "| Stock vs Nifty | — | +2.00% |" in text
        assert f"Baseline loss (brier): {outcome.baseline_loss:.4f}" in text
        assert f"Final loss (brier): {outcome.final_loss:.4f}" in text
        assert "LLM value added: +" in text and "lower loss" in text
        assert SINGLE_OBSERVATION_NOTE in text

    def test_invalid_section_shows_reason(self, stored, store, outcome_store):
        outcome = OutcomeReviewer(
            store, outcome_store, _source(stored, [0, -45, -45, -45, -45])
        ).review_matured(NOW_MATURED)[0]
        text = render_last_forecast_vs_actual(stored, outcome)
        assert f"Not evaluated: {outcome.invalid_reason}" in text
        assert "Baseline loss" not in text


def test_cli_review_scores_matured_forecasts(adjusted_state, store, monkeypatch):
    """End to end through `stock-analysis review` with a forecast whose target has passed."""
    from stock_analysis.cli import app
    from stock_analysis.database import close_database

    price = {**adjusted_state.price_snapshot, "last_date": "2026-09-15"}
    old = build_forecast_snapshot(
        adjusted_state.model_copy(update={"price_snapshot": price}), made_at=MADE_AT
    )
    store.save(old)
    source = _source(old)
    monkeypatch.setattr(YFinancePriceSource, "fetch", lambda self, *a: source.fetch(*a))

    close_database()
    try:
        result = CliRunner().invoke(app, ["review"])
        again = CliRunner().invoke(app, ["review"])
    finally:
        close_database()

    assert result.exit_code == 0, result.output
    assert "Last forecast vs actual" in result.output
    assert old.forecast_id in result.output
    assert "No matured forecasts awaiting evaluation." in again.output
