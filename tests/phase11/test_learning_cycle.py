"""The learning loop after review: postmortems, decision outcomes, audit trail."""

import sqlite3

import pytest

from stock_analysis.database import LearningStore
from stock_analysis.learning import LearningCycle
from stock_analysis.schemas.analyst_reports import DecisionType
from stock_analysis.schemas.learning import CauseCategory, PostmortemMethod
from tests.learning_helpers import (
    FakePostmortemLLM,
    evaluated_at,
    save_scored,
    snapshot_as_of,
    week,
)

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
def cycle(store, outcome_store, memory_store, learning_store):
    def _cycle(llm=None):
        return LearningCycle(
            store,
            outcome_store,
            memory_store,
            learning_store,
            llm_factory=llm or FakePostmortemLLM(DIAGNOSIS),
        )

    return _cycle


@pytest.fixture
def bad_week(adjusted_state, store, outcome_store):
    snap = snapshot_as_of(adjusted_state, week(0))
    outcome = save_scored(store, outcome_store, snap, 6.0, amplitude=2.5)
    return snap, outcome


class TestLearningCycle:
    def test_processes_each_scored_forecast_once(self, cycle, bad_week, learning_store):
        snap, _ = bad_week
        now = evaluated_at(snap)
        report = cycle().run(now)

        assert [p.forecast_id for p in report.postmortems] == [snap.forecast_id]
        assert learning_store.get_postmortem(snap.forecast_id) == report.postmortems[0]
        assert [a.action for a in report.lesson_actions] == ["created"]
        assert report.errors == []
        again = cycle().run(now)
        assert again.postmortems == [] and again.lesson_actions == []

    def test_point_in_time(self, cycle, bad_week):
        snap, outcome = bad_week
        # before the outcome was evaluated there is nothing to diagnose
        before = outcome.evaluated_at.replace(year=2025)
        assert cycle().run(before).postmortems == []

    def test_one_bad_forecast_does_not_change_the_system(
        self, cycle, bad_week, learning_store, memory_store
    ):
        snap, _ = bad_week
        now = evaluated_at(snap)
        report = cycle().run(now)

        assert report.postmortems[0].lessons  # a candidate was proposed...
        assert memory_store.active_lessons("RELIANCE", None, as_of=now, limit=5) == []
        (update,) = report.calibration_updates  # ...but nothing the next forecast uses changed
        assert not update.changed and update.estimate.vol_multiplier == 1.0
        assert learning_store.latest_calibration("RELIANCE", as_of=now) is None

    def test_failed_llm_still_records_a_postmortem(self, cycle, bad_week, learning_store):
        snap, _ = bad_week
        cycle(FakePostmortemLLM(error=TimeoutError("slow"))).run(evaluated_at(snap))
        stored = learning_store.get_postmortem(snap.forecast_id)
        assert stored.method == PostmortemMethod.DETERMINISTIC_FALLBACK
        assert stored.lessons == []


class TestDecisionOutcomes:
    def test_every_decision_is_recorded_with_the_result(self, cycle, bad_week, learning_store):
        snap, outcome = bad_week
        report = cycle().run(evaluated_at(snap))

        recorded = learning_store.decision_outcomes(snap.forecast_id)
        assert report.decision_outcomes_recorded == len(snap.decisions) == len(recorded)
        assert {r.decision_type for r in recorded} == {
            d.decision_type.value for d in snap.decisions
        }
        gate = next(
            r for r in recorded if r.decision_type == DecisionType.FORECAST_ADJUSTMENT_GATE.value
        )
        assert gate.decision == snap.decision(DecisionType.FORECAST_ADJUSTMENT_GATE).decision
        assert gate.llm_value_added == outcome.llm_value_added
        assert gate.uncalibrated_loss == outcome.uncalibrated_loss
        assert gate.primary_cause == CauseCategory.VOLATILITY_UNDERESTIMATED

    def test_summary_is_for_evaluation_only(
        self, cycle, adjusted_state, store, outcome_store, learning_store
    ):
        snaps = [snapshot_as_of(adjusted_state, week(n)) for n in range(3)]
        for snap, final in zip(snaps, (6.0, -6.0, 0.5), strict=True):
            save_scored(store, outcome_store, snap, final, amplitude=2.5)
        cycle().run(evaluated_at(snaps[-1]))

        summary = learning_store.decision_outcome_summary(before=evaluated_at(snaps[-1]))
        gate = [s for s in summary if s.decision_type == "forecast_adjustment_gate"]
        assert [(s.decision, s.n) for s in gate] == [("allow_adjustment", 3)]
        assert gate[0].direction_hit_rate is not None
        # the decision engine is untouched: a new forecast makes the same gate decision
        later = snapshot_as_of(adjusted_state, week(4))
        assert (
            later.decision(DecisionType.FORECAST_ADJUSTMENT_GATE).decision
            == snaps[0].decision(DecisionType.FORECAST_ADJUSTMENT_GATE).decision
        )


class TestAuditability:
    @pytest.mark.parametrize(
        "sql",
        [
            "UPDATE forecast_postmortems SET primary_cause = 'data_issue'",
            "DELETE FROM forecast_postmortems",
            "UPDATE decision_outcomes SET llm_value_added = 1.0",
            "DELETE FROM decision_outcomes",
        ],
    )
    def test_learning_records_are_immutable(self, cycle, bad_week, migrated_db, sql):
        cycle().run(evaluated_at(bad_week[0]))
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            migrated_db.execute(sql)

    def test_postmortem_keeps_the_facts_it_was_given(self, cycle, bad_week, learning_store):
        snap, _ = bad_week
        cycle().run(evaluated_at(snap))
        stored = learning_store.get_postmortem(snap.forecast_id)
        assert stored.facts.forecast_time and stored.facts.hindsight
        assert stored.prompt_version.startswith("prompt=sha256:")


class TestRobustness:
    def test_one_failing_forecast_does_not_block_the_others(
        self, cycle, adjusted_state, store, outcome_store, learning_store, monkeypatch
    ):
        import stock_analysis.learning.cycle as cycle_module

        snaps = [snapshot_as_of(adjusted_state, week(n)) for n in range(2)]
        for snap in snaps:
            save_scored(store, outcome_store, snap, 6.0, amplitude=2.5)
        real = cycle_module.run_postmortem

        def flaky(snapshot, outcome, **kwargs):
            if snapshot.forecast_id == snaps[0].forecast_id:
                raise ValueError("corrupt record")
            return real(snapshot, outcome, **kwargs)

        monkeypatch.setattr(cycle_module, "run_postmortem", flaky)
        report = cycle().run(evaluated_at(snaps[-1]))

        assert [p.forecast_id for p in report.postmortems] == [snaps[1].forecast_id]
        assert report.errors == [f"{snaps[0].forecast_id}: corrupt record"]
        assert [u.ticker for u in report.calibration_updates] == ["RELIANCE"]
        # the failed forecast is retried on the next run
        assert learning_store.pending_postmortems(before=evaluated_at(snaps[-1])) == [
            snaps[0].forecast_id
        ]

    def test_hindsight_news_reaches_the_postmortem(self, cycle, bad_week, learning_store):
        from datetime import timedelta

        from stock_analysis.schemas.learning import HindsightNewsItem

        snap, _ = bad_week
        calls = []

        class News:
            def fetch(self, symbol, start, end):
                calls.append((symbol, start, end))
                return [
                    HindsightNewsItem(
                        published_at=start + timedelta(days=1), headline="Tariff ruling"
                    )
                ]

        learning = cycle()
        learning.news_source = News()
        learning.run(evaluated_at(snap))
        assert calls[0][:2] == (snap.resolved_symbol, snap.made_at)
        facts = learning_store.get_postmortem(snap.forecast_id).facts
        assert any(f.source == "news" and "Tariff ruling" in f.text for f in facts.hindsight)
