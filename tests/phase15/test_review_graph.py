"""Plan.md Phase 15 "done when": the review graph works independently of the main graph."""

import sqlite3

import pytest
from typer.testing import CliRunner

from stock_analysis.database import LearningStore
from stock_analysis.langgraph.review_graph import (
    REVIEW_STEPS,
    ReviewState,
    build_review_graph,
    compile_review_graph,
    run_review,
)
from stock_analysis.review import NIFTY_50_SYMBOL, YFinancePriceSource
from stock_analysis.schemas.outcome import OutcomeStatus
from stock_analysis.snapshots import build_forecast_snapshot
from tests.forecast_helpers import MADE_AT
from tests.learning_helpers import (
    WEEKDAYS,
    FakePostmortemLLM,
    evaluated_at,
    snapshot_as_of,
    week,
    wiggle_path,
)
from tests.outcome_helpers import FakePriceSource, nifty_series, path_series

DIAGNOSIS = {
    "primary_cause": "volatility_underestimated",
    "explanation": "Swings were larger than the forecast allowed.",
    "knowable_at_forecast_time": [{"statement": "Quant weekly vol 2.71%", "fact_ids": ["F3"]}],
    "only_in_hindsight": [{"statement": "Realised vol far higher", "fact_ids": ["H4"]}],
    "lessons": [
        {
            "text": "Realised weekly volatility for this ticker can exceed the EWMA estimate "
            "when all analysts agree on a bullish trend",
            "scope": "ticker",
            "category": "volatility_underestimated",
            "evidence_refs": ["F3"],
        }
    ],
    "confidence": "medium",
}


@pytest.fixture
def learning_store(migrated_db):
    return LearningStore(migrated_db)


@pytest.fixture
def weeks(adjusted_state, store):
    """Two stored, non-overlapping forecasts with large swings (not expected noise)."""
    snaps = [snapshot_as_of(adjusted_state, week(n)) for n in (0, 1)]
    for snap in snaps:
        store.save(snap)
    return snaps


class WindowPrices:
    """Actual prices per forecast window: a large-swing path for the stock, flat Nifty."""

    def __init__(self, snaps, final_pct=6.0, amplitude=2.5):
        self.series = {}
        for snap in snaps:
            path = path_series(snap, wiggle_path(final_pct, amplitude))
            self.series[snap.resolved_symbol, snap.as_of_date] = path
            self.series[NIFTY_50_SYMBOL, snap.as_of_date] = nifty_series(snap, 0.0)

    def fetch(self, symbol, start, end):
        return self.series[symbol, start]


@pytest.fixture
def review(store, outcome_store, memory_store, learning_store):
    def _review(snaps, run_at, *, ticker=None, source=None, llm=None):
        return run_review(
            run_at,
            ticker=ticker,
            snapshot_store=store,
            outcome_store=outcome_store,
            memory_store=memory_store,
            learning_store=learning_store,
            price_source=source or WindowPrices(snaps),
            calendar=WEEKDAYS,
            llm_factory=llm or FakePostmortemLLM(DIAGNOSIS),
        )

    return _review


class TestTopology:
    def test_steps_run_in_the_plan_order(self, store, outcome_store, memory_store, learning_store):
        graph = compile_review_graph(store, outcome_store, memory_store, learning_store)
        edges = {(e.source, e.target) for e in graph.get_graph().edges}
        chain = ["__start__", *REVIEW_STEPS, "__end__"]
        assert edges == set(zip(chain, chain[1:], strict=False))
        assert REVIEW_STEPS == (
            "find_matured",
            "score",
            "postmortem",
            "learning",
            "calibration",
            "scorecards",
            "track_record",
        )


class TestReviewGraph:
    def test_full_review(self, review, weeks, store, outcome_store, learning_store, memory_store):
        now = evaluated_at(weeks[1])
        state = review(weeks, now)

        assert state.completed_steps == list(REVIEW_STEPS)
        assert state.errors == []
        ids = [s.forecast_id for s in weeks]
        # find matured -> score
        assert sorted(state.matured_forecast_ids) == sorted(ids)
        assert {o.status for o in state.outcomes} == {OutcomeStatus.SCORED}
        assert all(outcome_store.get(i) is not None for i in ids)
        # postmortem -> learning: diagnosed oldest first, stored, decisions and lessons recorded
        assert [p.forecast_id for p in state.postmortems] == ids
        assert all(learning_store.get_postmortem(i) is not None for i in ids)
        assert state.decision_outcomes_recorded == sum(len(s.decisions) for s in weeks)
        assert [a.action for a in state.lesson_actions] == ["created", "confirmed"]
        # calibration: re-estimated (too few samples to change) and judged on the benchmarks
        (update,) = state.calibration_updates
        assert update.ticker == "RELIANCE" and not update.changed
        assert state.benchmarks["RELIANCE"].n == 2
        # scorecards and track record cover the newly scored forecasts
        assert state.scorecards.n_forecasts == 2
        assert state.evaluation.track_records[0].n_forecasts == 2
        all_time = state.evaluation.track_records[0].windows[-1]
        assert all_time.weeks is None and all_time.n == 2

    def test_second_review_has_nothing_new_but_still_summarizes(self, review, weeks):
        now = evaluated_at(weeks[1])
        review(weeks, now)
        again = review(weeks, now)

        assert again.errors == []
        assert again.matured_forecast_ids == [] and again.outcomes == []
        assert again.postmortems == [] and again.decision_outcomes_recorded == 0
        assert again.scorecards.n_forecasts == 2
        assert again.evaluation.track_records[0].n_forecasts == 2

    def test_point_in_time(self, review, weeks, outcome_store):
        """At week 0's evaluation time only week 0 is reviewed; week 1 has not matured."""
        state = review(weeks, evaluated_at(weeks[0]))

        assert state.matured_forecast_ids == [weeks[0].forecast_id]
        assert [p.forecast_id for p in state.postmortems] == [weeks[0].forecast_id]
        assert outcome_store.get(weeks[1].forecast_id) is None
        assert state.scorecards.n_forecasts == 1
        assert state.evaluation.track_records[0].n_forecasts == 1

    def test_unresolved_is_not_learned_from(self, review, weeks, outcome_store):
        """Prices not yet published: nothing is stored, learned or counted; retried next run."""
        state = review(weeks, evaluated_at(weeks[0]), source=FakePriceSource({}))

        assert [o.status for o in state.outcomes] == [OutcomeStatus.UNRESOLVED]
        assert outcome_store.get(weeks[0].forecast_id) is None
        assert state.postmortems == [] and state.calibration_updates == []
        assert state.scorecards.n_forecasts == 0

    def test_ticker_filter(self, review, weeks, outcome_store):
        state = review(weeks, evaluated_at(weeks[1]), ticker="TCS")

        assert state.matured_forecast_ids == [] and state.postmortems == []
        assert outcome_store.get(weeks[0].forecast_id) is None
        assert state.scorecards.ticker == "TCS" and state.scorecards.n_forecasts == 0
        # the requested ticker is still recalibrated (nothing to learn from yet)
        assert [u.ticker for u in state.calibration_updates] == ["TCS"]

    def test_failed_postmortem_llm_falls_back_to_rules(self, review, weeks, learning_store):
        state = review(
            weeks, evaluated_at(weeks[1]), llm=FakePostmortemLLM(error=TimeoutError("slow"))
        )
        assert state.errors == []
        assert all(p.lessons == [] for p in state.postmortems)
        assert all(learning_store.get_postmortem(s.forecast_id) for s in weeks)


class TestFailureIsolation:
    def test_failing_step_is_reported_and_later_steps_still_run(
        self, review, weeks, learning_store, monkeypatch
    ):
        def broken(*args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(learning_store, "list_decision_outcomes", broken)
        state = review(weeks, evaluated_at(weeks[1]))

        assert state.errors == ["scorecards: disk I/O error"]
        assert "scorecards" not in state.completed_steps and state.scorecards is None
        assert state.completed_steps[-1] == "track_record"
        assert state.evaluation.track_records[0].n_forecasts == 2

    def test_one_bad_forecast_does_not_block_the_others(
        self, review, weeks, learning_store, monkeypatch
    ):
        save = learning_store.save_postmortem

        def save_all_but_first(postmortem):
            if postmortem.forecast_id == weeks[0].forecast_id:
                raise sqlite3.OperationalError("locked")
            save(postmortem)

        monkeypatch.setattr(learning_store, "save_postmortem", save_all_but_first)
        state = review(weeks, evaluated_at(weeks[1]))

        assert state.errors == [f"{weeks[0].forecast_id}: locked"]
        # only the stored postmortem is reported; the failed one is retried next run
        assert [p.forecast_id for p in state.postmortems] == [weeks[1].forecast_id]
        assert learning_store.pending_postmortems(before=evaluated_at(weeks[1])) == [
            weeks[0].forecast_id
        ]

    def test_one_unscorable_forecast_is_reported_and_others_are_scored(
        self, review, weeks, outcome_store, monkeypatch
    ):
        save = outcome_store.save

        def save_all_but_first(outcome):
            if outcome.forecast_id == weeks[0].forecast_id:
                raise sqlite3.OperationalError("database is locked")
            save(outcome)

        monkeypatch.setattr(outcome_store, "save", save_all_but_first)
        state = review(weeks, evaluated_at(weeks[1]))

        assert state.errors == [f"{weeks[0].forecast_id}: database is locked"]
        assert "score" in state.completed_steps
        assert [o.forecast_id for o in state.outcomes] == [weeks[1].forecast_id]
        assert outcome_store.get(weeks[0].forecast_id) is None  # retried next run
        assert [p.forecast_id for p in state.postmortems] == [weeks[1].forecast_id]

    def test_benchmark_failure_keeps_the_calibration_update(self, review, weeks, monkeypatch):
        import stock_analysis.langgraph.review_graph as review_graph

        def broken(*args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(review_graph, "compare_benchmarks", broken)
        state = review(weeks, evaluated_at(weeks[1]))

        assert state.errors == ["benchmarks RELIANCE: disk I/O error"]
        assert "calibration" in state.completed_steps
        assert [u.ticker for u in state.calibration_updates] == ["RELIANCE"]
        assert state.benchmarks == {}

    def test_requested_ticker_is_recalibrated_when_postmortems_fail(
        self, review, weeks, learning_store, monkeypatch
    ):
        def broken(*args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(learning_store, "pending_postmortems", broken)
        state = review(weeks, evaluated_at(weeks[1]), ticker="RELIANCE")

        assert state.errors == ["postmortem: disk I/O error"]
        assert [u.ticker for u in state.calibration_updates] == ["RELIANCE"]
        assert state.completed_steps[-1] == "track_record"


class TestIndependence:
    def test_review_never_builds_the_forecast_graph(self, review, weeks, monkeypatch):
        import stock_analysis.langgraph.workflow as workflow

        def forbidden(*args, **kwargs):
            raise AssertionError("the review graph must not run the forecast graph")

        monkeypatch.setattr(workflow, "build_workflow", forbidden)
        monkeypatch.setattr(workflow, "compile_graph", forbidden)
        state = review(weeks, evaluated_at(weeks[1]))
        assert state.completed_steps == list(REVIEW_STEPS)

    def test_state_is_validated(self, store, outcome_store, memory_store, learning_store):
        graph = build_review_graph(store, outcome_store, memory_store, learning_store)
        assert graph.state_schema is ReviewState
        with pytest.raises(ValueError):
            ReviewState(run_at=MADE_AT, unexpected=1)


def test_cli_review_runs_the_review_graph(adjusted_state, store, monkeypatch):
    """`stock-analysis review` shows the review plus the scorecards and track record."""
    from stock_analysis.cli import app
    from stock_analysis.database import close_database

    price = {**adjusted_state.price_snapshot, "last_date": "2026-09-15"}
    old = build_forecast_snapshot(
        adjusted_state.model_copy(update={"price_snapshot": price}), made_at=MADE_AT
    )
    store.save(old)
    source = FakePriceSource(
        {
            old.resolved_symbol: path_series(old, [0.5, 1.0, 1.8, 2.4, 3.0]),
            NIFTY_50_SYMBOL: nifty_series(old, 1.0),
        }
    )
    monkeypatch.setattr(YFinancePriceSource, "fetch", lambda self, *a: source.fetch(*a))

    close_database()
    try:
        result = CliRunner().invoke(app, ["review", "--no-llm", "--ticker", "RELIANCE.NS"])
    finally:
        close_database()

    assert result.exit_code == 0, result.output
    assert "Last forecast vs actual" in result.output
    assert "## Postmortem" in result.output
    assert "System adaptation: RELIANCE" in result.output
    assert "## Scorecards for RELIANCE" in result.output
    assert "## Forecast track record for RELIANCE" in result.output
    assert "Review error" not in result.output
