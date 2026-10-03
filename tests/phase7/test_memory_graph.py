"""Plan.md Phase 7 "done when": a second run receives prior context."""

from datetime import UTC, datetime, timedelta

import pytest

from stock_analysis.database import Database, ForecastSnapshotStore, MemoryStore
from stock_analysis.langgraph.runner import run_forecast
from stock_analysis.langgraph.workflow import format_prior_context
from stock_analysis.schemas.memory import LessonCandidate, LessonEvidenceKind, MemoryContext
from stock_analysis.snapshots import find_unsupported_numbers
from tests.forecast_helpers import FakeLLM, run_graph


@pytest.mark.usefixtures("price_history")
class TestSecondRunReceivesPriorContext:
    def test_first_run_has_empty_memory_and_writes_insight(
        self, initial_state, store, memory_store
    ):
        llm = FakeLLM("valid")
        first = run_graph(llm, initial_state, store, memory_store)

        context = MemoryContext.model_validate(first.memory_context)
        assert context.loaded and not context.prior_forecasts and not context.active_lessons
        assert context.track_record.forecasts_made == 0
        assert first.memory_written is True
        assert "PRIOR CONTEXT" not in llm.predictor_prompts[0]

    def test_second_run_sees_first_forecast(self, initial_state, store, memory_store):
        first = run_graph(FakeLLM("valid"), initial_state, store, memory_store)
        llm = FakeLLM("valid")
        second = run_graph(llm, initial_state, store, memory_store)

        context = MemoryContext.model_validate(second.memory_context)
        assert [f.forecast_id for f in context.prior_forecasts] == [first.forecast_id]
        assert context.track_record.forecasts_made == 1

        prompt = llm.predictor_prompts[0]
        prior = context.prior_forecasts[0]
        assert "PRIOR CONTEXT" in prompt
        assert f"{prior.p50_price:.2f}" in prompt
        assert "Do not treat a previous forecast as evidence" in prompt

        snapshot = store.get(second.forecast_id)
        assert (
            snapshot.data_inputs["memory"]["prior_forecasts"][0]["forecast_id"] == first.forecast_id
        )
        report = second.forecast_report
        assert f"Previous forecast `{first.forecast_id}`" in report
        assert (
            "Track record: earlier forecasts 1; scored against actual prices 0; "
            "awaiting evaluation 0."
        ) in report
        assert find_unsupported_numbers(report, snapshot) == []

    def test_active_lesson_reaches_predictor_and_report(self, initial_state, store, memory_store):
        runs = [run_graph(FakeLLM("valid"), initial_state, store, memory_store) for _ in range(3)]
        lesson = memory_store.add_lesson(
            LessonCandidate(
                text="Weekly realized volatility exceeded the model estimate in trending regimes",
                scope="ticker",
                category="volatility",
                ticker="RELIANCE",
            ),
            source_forecast_id=runs[0].forecast_id,
            first_seen=datetime.now(UTC) - timedelta(minutes=5),
        )
        for run in runs[1:]:
            memory_store.record_evidence(
                lesson.lesson_id,
                LessonEvidenceKind.CONFIRMED,
                forecast_id=run.forecast_id,
                recorded_at=datetime.now(UTC) - timedelta(minutes=1),
            )

        llm = FakeLLM("valid")
        state = run_graph(llm, initial_state, store, memory_store)

        assert f"lesson:{lesson.lesson_id}" in llm.predictor_prompts[0]
        assert "Active lesson (ticker, 3 confirmations)" in state.forecast_report
        snapshot = store.get(state.forecast_id)
        assert find_unsupported_numbers(state.forecast_report, snapshot) == []

    def test_broken_memory_never_blocks_the_forecast(self, initial_state, store, tmp_path):
        broken = MemoryStore(Database(tmp_path / "no_tables.db"))
        state = run_graph(FakeLLM("valid"), initial_state, store, broken)

        assert state.snapshot_persisted is True
        context = MemoryContext.model_validate(state.memory_context)
        assert context.loaded is False and "no such table" in context.error
        assert state.memory_written is False
        assert "Memory could not be loaded" in state.forecast_report

    def test_without_memory_store_nothing_is_loaded(self, initial_state):
        llm = FakeLLM("valid")
        state = run_graph(llm, initial_state)
        assert state.memory_context is None
        assert state.memory_written is False
        assert "Memory was not configured for this run." in state.forecast_report
        assert "PRIOR CONTEXT" not in llm.predictor_prompts[0]


@pytest.mark.usefixtures("price_history")
def test_run_forecast_carries_memory_between_runs(migrated_db, initial_state):
    from stock_analysis.database import close_database

    close_database()
    try:
        first = run_forecast(initial_state, llm_factory=FakeLLM("valid"))
        second = run_forecast(initial_state, llm_factory=FakeLLM("valid"))
    finally:
        close_database()

    assert first.memory_written and second.memory_written
    prior = MemoryContext.model_validate(second.memory_context).prior_forecasts
    assert [f.forecast_id for f in prior] == [first.forecast_id]
    assert ForecastSnapshotStore(migrated_db).get(second.forecast_id) is not None


class TestFormatPriorContext:
    def test_nothing_to_show(self):
        assert format_prior_context(None) == []
        empty = MemoryContext(loaded=True).model_dump(mode="json")
        assert format_prior_context(empty) == []
