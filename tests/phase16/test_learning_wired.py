"""Plan.md Phase 16 "done when": a second run after a forecast matures visibly shows the
previous forecast review and any eligible adaptation."""

from datetime import UTC, datetime, timedelta

import pytest

from stock_analysis.config.settings import get_settings
from stock_analysis.database import Database, LearningStore
from stock_analysis.langgraph.runner import run_forecast
from stock_analysis.langgraph.workflow import compile_graph, format_prior_context
from stock_analysis.review import NIFTY_50_SYMBOL
from stock_analysis.schemas.graph_state import GraphState
from stock_analysis.schemas.learning import CalibrationParams
from stock_analysis.schemas.memory import MemoryContext, OutcomeMetrics, PreRunReview, WindowMetrics
from stock_analysis.schemas.scorecard import HitRate
from stock_analysis.snapshots import find_unsupported_numbers
from stock_analysis.versions import CALIBRATOR_VERSION
from tests.backtest_helpers import BacktestLLM
from tests.forecast_helpers import MADE_AT, FakeLLM
from tests.outcome_helpers import NOW_MATURED, FakePriceSource, nifty_series, path_series

# The first forecast's window: the stock rallies far beyond its P90, Nifty barely moves
RALLY_PCT = [2.0, 4.0, 6.0, 8.0, 9.0]


@pytest.fixture
def learning_store(migrated_db):
    return LearningStore(migrated_db)


@pytest.fixture
def stores(store, memory_store, learning_store, outcome_store):
    return {
        "store": store,
        "memory_store": memory_store,
        "learning_store": learning_store,
        "outcome_store": outcome_store,
    }


def _run(initial_state, stores, run_at, llm=None, **options):
    state = initial_state.model_copy(update={"run_at": run_at})
    options.setdefault("news_source", None)  # no RSS hindsight news in tests
    return run_forecast(state, llm_factory=llm or BacktestLLM(), **stores, **options)


def _rally_prices(snapshot):
    return FakePriceSource(
        {
            snapshot.resolved_symbol: path_series(snapshot, RALLY_PCT),
            NIFTY_50_SYMBOL: nifty_series(snapshot, 0.5),
        }
    )


def _calibration(created_at):
    return CalibrationParams(
        ticker="RELIANCE",
        version=1,
        vol_multiplier=1.18,
        p50_bias_shift_pct=0.25,
        previous_vol_multiplier=1.0,
        previous_p50_bias_shift_pct=0.0,
        n_samples=8,
        reason=["Realized moves were larger than predicted in 6 of 8 forecasts"],
        evidence={"forecast_ids": []},
        calibrator_version=CALIBRATOR_VERSION,
        created_at=created_at,
    )


@pytest.mark.usefixtures("price_history")
class TestSecondRunShowsReview:
    @pytest.fixture
    def runs(self, initial_state, stores):
        first = _run(initial_state, stores, MADE_AT, price_source=FakePriceSource({}))
        snapshot = stores["store"].get(first.forecast_id)
        llm = BacktestLLM()
        second = _run(initial_state, stores, NOW_MATURED, llm, price_source=_rally_prices(snapshot))
        return first, second, llm

    def test_first_run_has_nothing_to_review(self, runs):
        first, _, _ = runs
        review = PreRunReview.model_validate(first.review_summary)
        assert review.matured == 0 and review.errors == []
        context = MemoryContext.model_validate(first.memory_context)
        assert context.last_review is None
        assert context.adaptation.version == 0
        assert "No earlier forecast of this ticker has been evaluated yet." in first.forecast_report

    def test_review_runs_before_the_forecast(self, runs, outcome_store, learning_store):
        first, second, _ = runs
        review = PreRunReview.model_validate(second.review_summary)
        assert (review.matured, review.scored, review.postmortems) == (1, 1, 1)
        assert review.errors == []
        assert outcome_store.get(first.forecast_id) is not None
        assert learning_store.get_postmortem(first.forecast_id) is not None

    def test_memory_holds_the_last_review(self, runs, outcome_store, learning_store):
        first, second, _ = runs
        context = MemoryContext.model_validate(second.memory_context)
        outcome = outcome_store.get(first.forecast_id)
        postmortem = learning_store.get_postmortem(first.forecast_id)
        last = context.last_review
        assert last.forecast_id == first.forecast_id
        assert last.status == "scored"
        assert last.actual_close == pytest.approx(outcome.actual_close_on_forecast_basis)
        assert last.signed_error_pct == pytest.approx(outcome.signed_error_pct)
        assert last.in_80pct_band is False and last.realized_direction == "up"
        assert last.primary_cause == postmortem.primary_cause.value
        assert last.cause_explanation == postmortem.explanation
        # the track record now carries rolling metrics, and the scorecards the regime
        metrics = context.track_record.outcome_metrics
        assert metrics.n_forecasts == 1
        assert metrics.windows[0].n == 1
        assert context.scorecards.n_forecasts == 1
        assert [g.group for g in context.scorecards.regimes] == [last_regime(first)]

    def test_predictor_receives_review_calibration_and_track_record(self, runs):
        first, second, llm = runs
        prompt = llm.predictor_prompts[-1]
        last = MemoryContext.model_validate(second.memory_context).last_review
        assert "LAST REVIEW of the forecast as of" in prompt
        assert f"actual close {last.actual_close:.2f}" in prompt
        assert "OUTSIDE the P10-P90 band" in prompt
        assert f"Postmortem cause: {last.primary_cause}" in prompt
        assert "CALIBRATION: none yet" in prompt
        assert "8 weeks = 26 weeks = All time: 1 forecasts" in prompt
        assert "SCORECARDS (1 independent scored forecasts" in prompt
        assert f"Regime {last_regime(first)} (this run's regime)" in prompt
        assert "prefer a wider band" in prompt
        assert "Prefer wider bands when uncertainty increases" in prompt

    def test_report_shows_review_and_adaptation(self, runs, store):
        first, second, _ = runs
        report = second.forecast_report
        last = MemoryContext.model_validate(second.memory_context).last_review
        assert "## Last forecast vs actual" in report
        assert f"- Forecast `{first.forecast_id}`" in report
        assert (
            f"| P50 / close | ₹{last.baseline_p50_price:.2f} | ₹{last.p50_price:.2f} "
            f"| ₹{last.actual_close:.2f} |"
        ) in report
        assert f"₹{last.p90_price:.2f} (OUTSIDE)" in report
        assert f"| {last.signed_error_pct:+.2f}% |" in report
        assert f"- Primary cause: **{last.primary_cause}**" in report
        assert "## System adaptation" in report
        assert "No calibration yet" in report and "(1 so far)" in report
        assert "## Track record" in report
        assert "- **8 weeks = 26 weeks = All time**: 1 forecasts; direction" in report
        assert "- Review before this run: 1 matured forecasts; 1 scored" in report
        assert report.index("## Last forecast vs actual") < report.index("## Current analysis")
        snapshot = store.get(second.forecast_id)
        assert snapshot.data_inputs["review"]["scored"] == 1
        assert find_unsupported_numbers(report, snapshot) == []


def last_regime(state: GraphState) -> str:
    return state.decision.result


@pytest.mark.usefixtures("price_history")
class TestAdaptation:
    def test_new_calibration_is_shown_and_applied(self, initial_state, stores, learning_store):
        first = _run(initial_state, stores, MADE_AT, review=False)
        learning_store.save_calibration(_calibration(MADE_AT + timedelta(days=2)))
        llm = BacktestLLM()
        second = _run(initial_state, stores, NOW_MATURED, llm, review=False)

        adaptation = MemoryContext.model_validate(second.memory_context).adaptation
        assert adaptation.version == 1 and adaptation.changed_since_last_forecast is True
        assert second.calibration["version"] == 1
        prompt = llm.predictor_prompts[-1]
        assert "CALIBRATION v1 (estimated on 8 scored forecasts): volatility multiplier " in prompt
        assert "1.00 -> 1.18" in prompt and "do not apply it again" in prompt
        report = second.forecast_report
        assert "- Calibration v1: volatility multiplier 1.00 → 1.18; P50 shift +0.00% → +0.25%" in (
            report
        )
        assert "- New since the previous forecast of this ticker" in report
        assert "- Reason: Realized moves were larger than predicted in 6 of 8 forecasts" in report
        assert "- Applied to this forecast (v1)" in report
        snapshot = stores["store"].get(second.forecast_id)
        assert find_unsupported_numbers(report, snapshot) == []
        assert first.forecast_id != second.forecast_id

    def test_unchanged_calibration(self, initial_state, stores, learning_store):
        learning_store.save_calibration(_calibration(MADE_AT - timedelta(days=1)))
        _run(initial_state, stores, MADE_AT, review=False)
        second = _run(initial_state, stores, NOW_MATURED, review=False)
        assert "- Unchanged since the previous forecast of this ticker" in second.forecast_report

    def test_disabled_learning_is_reported(
        self, initial_state, stores, learning_store, monkeypatch
    ):
        monkeypatch.setenv("LEARNING_ENABLED", "false")
        get_settings.cache_clear()
        learning_store.save_calibration(_calibration(MADE_AT - timedelta(days=1)))
        state = _run(initial_state, stores, MADE_AT, review=False)
        assert state.calibration is None
        assert "- Not applied: learning is disabled" in state.forecast_report


@pytest.mark.usefixtures("price_history")
class TestNeverBlocks:
    def test_failing_review_is_recorded(self, initial_state, store, memory_store, outcome_store):
        def reviewer(run_at, ticker):
            raise RuntimeError("price feed down")

        graph = compile_graph(
            FakeLLM("valid"), store, memory_store, outcome_store=outcome_store, reviewer=reviewer
        )
        state = GraphState(**graph.invoke(initial_state))
        assert state.snapshot_persisted is True
        review = PreRunReview.model_validate(state.review_summary)
        assert review.errors == ["review failed: price feed down"]
        assert "- Review error: review failed: price feed down" in state.forecast_report

    def test_broken_learning_store_only_drops_its_parts(
        self, initial_state, store, memory_store, outcome_store, tmp_path
    ):
        broken = LearningStore(Database(tmp_path / "no_tables.db"))
        graph = compile_graph(
            FakeLLM("valid"), store, memory_store, broken, outcome_store=outcome_store
        )
        state = GraphState(**graph.invoke(initial_state))
        assert state.snapshot_persisted is True
        context = MemoryContext.model_validate(state.memory_context)
        assert context.loaded is True and context.adaptation is None
        assert any(w.startswith("adaptation could not be loaded") for w in context.warnings)
        # nothing was evaluated yet, so the review never needed the learning store
        assert context.learning_loaded == ["last_review", "track_record"]
        report = state.forecast_report
        assert "- Warning: adaptation could not be loaded" in report
        assert "## System adaptation\n\n- Not available for this run" in report
        assert "No earlier forecast of this ticker has been evaluated yet." in report

    def test_unexpected_error_in_a_part_never_blocks(
        self, initial_state, store, memory_store, learning_store, outcome_store, monkeypatch
    ):
        def broken(*args, **kwargs):
            raise KeyError("regime")

        monkeypatch.setattr("stock_analysis.learning.context.build_scorecards", broken)
        graph = compile_graph(
            FakeLLM("valid"), store, memory_store, learning_store, outcome_store=outcome_store
        )
        state = GraphState(**graph.invoke(initial_state))
        assert state.snapshot_persisted is True
        context = MemoryContext.model_validate(state.memory_context)
        assert context.scorecards is None
        assert "scorecards could not be loaded: KeyError: 'regime'" in context.warnings
        assert sorted(context.learning_loaded) == ["adaptation", "last_review", "track_record"]

    def test_without_outcome_store_no_review_is_loaded(self, initial_state, store, memory_store):
        graph = compile_graph(FakeLLM("valid"), store, memory_store)
        state = GraphState(**graph.invoke(initial_state))
        context = MemoryContext.model_validate(state.memory_context)
        assert context.last_review is None and context.scorecards is None
        assert context.learning_loaded == []
        assert state.review_summary is None
        assert "review" not in store.get(state.forecast_id).data_inputs
        # nothing was loaded, so the report does not claim there is nothing to review
        assert "## Last forecast vs actual" not in state.forecast_report
        assert "## System adaptation" not in state.forecast_report


@pytest.mark.usefixtures("price_history")
class TestRunnerReviewDefaults:
    @pytest.fixture
    def captured(self, monkeypatch):
        import stock_analysis.langgraph.runner as runner

        calls = []
        real = runner.make_pre_run_reviewer

        def spy(*stores, **kwargs):
            calls.append(kwargs)
            return real(*stores, **{**kwargs, "news_source": None})

        monkeypatch.setattr(runner, "make_pre_run_reviewer", spy)
        return calls

    def test_llm_postmortems_get_hindsight_news(self, initial_state, stores, captured):
        from stock_analysis.learning import RssHindsightNewsSource

        llm = BacktestLLM()
        run_forecast(initial_state, llm_factory=llm, price_source=FakePriceSource({}), **stores)
        assert captured[0]["llm_factory"] is llm
        assert isinstance(captured[0]["news_source"], RssHindsightNewsSource)

    def test_rules_only_postmortems_need_no_news(self, initial_state, stores, captured):
        run_forecast(
            initial_state,
            llm_factory=BacktestLLM(),
            review_llm=False,
            price_source=FakePriceSource({}),
            **stores,
        )
        assert captured[0]["llm_factory"] is None
        assert "news_source" not in captured[0]

    def test_review_can_be_skipped(self, initial_state, stores, captured):
        state = run_forecast(initial_state, llm_factory=BacktestLLM(), review=False, **stores)
        assert captured == [] and state.review_summary is None


class TestPointInTime:
    def test_review_evaluated_after_the_clock_is_not_seen(
        self, adjusted_state, store, outcome_store
    ):
        from tests.learning_helpers import save_scored, snapshot_as_of, week

        snapshot = snapshot_as_of(adjusted_state, week(0))
        outcome = save_scored(store, outcome_store, snapshot, 4.0)
        before = outcome.evaluated_at - timedelta(seconds=1)
        assert outcome_store.latest_evaluated("RELIANCE", before=before) is None
        found = outcome_store.latest_evaluated("RELIANCE", before=outcome.evaluated_at)
        assert found.forecast_id == snapshot.forecast_id


class TestPromptFormatting:
    def test_empty_learning_context_adds_nothing(self):
        context = MemoryContext(loaded=True).model_dump(mode="json")
        assert format_prior_context(context, regime="trending_up") == []

    def test_invalid_review(self):
        context = {
            "loaded": True,
            "last_review": {
                "forecast_id": "f1",
                "as_of_date": "2026-01-05",
                "target_date": "2026-01-12",
                "evaluated_at": datetime(2026, 1, 13, tzinfo=UTC).isoformat(),
                "status": "invalid",
                "invalid_reason": "unexplained 40% daily move",
                "last_close": 100.0,
                "prob_up": 0.5,
                "prob_flat": 0.2,
                "prob_down": 0.3,
                "p10_price": 95.0,
                "p50_price": 100.0,
                "p90_price": 105.0,
                "adjustment_applied": False,
                "calibration_version": 0,
            },
        }
        lines = format_prior_context(context)
        assert any("could not be evaluated (unexplained 40% daily move)" in line for line in lines)


class TestDistinctWindows:
    def _window(self, label, n):
        return WindowMetrics(label=label, n=n, direction=HitRate(n=n, hits=0), finding="x")

    def test_nested_windows_with_the_same_forecasts_are_merged(self):
        metrics = OutcomeMetrics(
            n_forecasts=3,
            min_samples=20,
            windows=[
                self._window("8 weeks", 2),
                self._window("26 weeks", 3),
                self._window("All time", 3),
            ],
        )
        assert [(label, w.n) for label, w in metrics.distinct_windows()] == [
            ("8 weeks", 2),
            ("26 weeks = All time", 3),
        ]

    def test_empty_windows_are_left_out(self):
        metrics = OutcomeMetrics(
            n_forecasts=1,
            min_samples=20,
            windows=[self._window("8 weeks", 0), self._window("All time", 1)],
        )
        assert [label for label, _ in metrics.distinct_windows()] == ["All time"]


@pytest.fixture(scope="module")
def backtest(tmp_path_factory):
    """12 weekly forecasts; each one's memory comes from the reviews of earlier weeks."""
    from stock_analysis.backtest import run_backtest
    from tests.backtest_helpers import LAST_TARGET, NOW, RecordingLoader, synthetic_history

    path = tmp_path_factory.mktemp("bt16") / "backtest.db"
    llm = BacktestLLM()
    result = run_backtest(
        "RELIANCE",
        12,
        database_path=path,
        llm_factory=llm,
        use_llm=True,
        end=LAST_TARGET,
        now=NOW,
        history_loader=RecordingLoader(synthetic_history(weekly_swing_pct=5.0)),
    )
    db = Database(path)
    yield result, llm, db
    db.close()


class TestBacktestUsesTheReview:
    def test_each_week_sees_the_previous_weeks_review(self, backtest):
        from stock_analysis.database import ForecastSnapshotStore

        result, _, db = backtest
        store = ForecastSnapshotStore(db)
        previous = None
        for week in result.weeks:
            ctx = MemoryContext.model_validate(store.get(week.forecast_id).data_inputs["memory"])
            if previous is None:
                assert ctx.last_review is None
            else:
                assert ctx.last_review.forecast_id == previous.forecast_id
                assert ctx.last_review.primary_cause == previous.primary_cause
            assert ctx.adaptation.version == week.calibration_version
            previous = week

    def test_adaptation_is_visible_once_calibration_changes(self, backtest):
        from stock_analysis.database import ForecastSnapshotStore

        result, llm, db = backtest
        store = ForecastSnapshotStore(db)
        first_calibrated = next(w for w in result.weeks if w.calibration_version)
        snapshot = store.get(first_calibrated.forecast_id)
        report = snapshot_report(snapshot)
        assert f"- Calibration v{first_calibrated.calibration_version}: volatility multiplier" in (
            report
        )
        assert "- New since the previous forecast of this ticker" in report
        assert any(
            f"CALIBRATION v{first_calibrated.calibration_version} " in p
            for p in llm.predictor_prompts
        )


def snapshot_report(snapshot):
    from stock_analysis.snapshots import render_forecast_report

    return render_forecast_report(snapshot)
