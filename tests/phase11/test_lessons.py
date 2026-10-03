"""Plan.md §20 / §64: lesson lifecycle candidate -> active -> retired."""

from datetime import timedelta

import pytest

from stock_analysis.learning import apply_lessons, retire_stale_lessons, run_postmortem
from stock_analysis.schemas.memory import LessonStatus
from tests.learning_helpers import (
    FakePostmortemLLM,
    evaluated_at,
    save_scored,
    snapshot_as_of,
    week,
)

VOL_LESSON = {
    "text": "Realised weekly volatility for this ticker can exceed the EWMA estimate "
    "when all analysts agree on a bullish trend",
    "scope": "ticker",
    "category": "volatility_underestimated",
    "evidence_refs": ["F3"],
}
EVENT_LESSON = {
    "text": "Earnings weeks for this ticker move more than the quant band allows",
    "scope": "ticker",
    "category": "earnings_or_corporate_event",
    "evidence_refs": ["F3"],
}


def _llm(lesson):
    return FakePostmortemLLM(
        {
            "primary_cause": "volatility_underestimated",
            "explanation": "Swings were larger than the forecast allowed.",
            "knowable_at_forecast_time": [],
            "only_in_hindsight": [{"statement": "Volatility spiked", "fact_ids": ["H4"]}],
            "lessons": [lesson],
            "confidence": "medium",
        }
    )


@pytest.fixture
def evaluate(adjusted_state, store, outcome_store, memory_store):
    """Store, score and diagnose a forecast for week ``n``; apply its lessons."""

    def _evaluate(n, final_pct=6.0, amplitude=2.5, llm=None, as_of=None, **scenario):
        snap = snapshot_as_of(adjusted_state, as_of or week(n))
        outcome = save_scored(
            store, outcome_store, snap, final_pct, amplitude=amplitude, **scenario
        )
        now = evaluated_at(snap)
        pm = run_postmortem(snap, outcome, llm_factory=llm or _llm(VOL_LESSON), now=now)
        return snap, apply_lessons(memory_store, snap, outcome, pm, now=now)

    return _evaluate


def _lesson(memory_store, actions, now):
    return memory_store.get_lesson(actions[0].lesson_id, as_of=now)


class TestActivation:
    def test_one_bad_forecast_creates_only_a_candidate(self, evaluate, memory_store):
        snap, actions = evaluate(0)
        assert [a.action for a in actions] == ["created"]
        lesson = _lesson(memory_store, actions, evaluated_at(snap))
        assert lesson.status == LessonStatus.CANDIDATE
        assert lesson.evidence_count == 1
        # candidates are never shown to the predictor
        assert (
            memory_store.active_lessons("RELIANCE", None, as_of=evaluated_at(snap), limit=5) == []
        )

    def test_active_after_three_independent_confirmations(self, evaluate, memory_store):
        _, first = evaluate(0)
        _, second = evaluate(1)
        assert [a.action for a in second] == ["confirmed"]
        assert second[0].status_after == LessonStatus.CANDIDATE
        snap, third = evaluate(2)
        assert third[0].status_after == LessonStatus.ACTIVE

        lesson = _lesson(memory_store, first, evaluated_at(snap))
        assert lesson.evidence_count == 3 and len(set(lesson.source_forecast_ids)) == 3
        active = memory_store.active_lessons("RELIANCE", None, as_of=evaluated_at(snap), limit=5)
        assert [a.lesson_id for a in active] == [lesson.lesson_id]

    def test_discrete_event_lesson_needs_two(self, evaluate):
        _, first = evaluate(0, llm=_event_llm(), dividend_on_day=2)
        assert first[0].status_after == LessonStatus.CANDIDATE
        _, second = evaluate(1, llm=_event_llm(), dividend_on_day=2)
        assert second[0].status_after == LessonStatus.ACTIVE

    def test_overlapping_forecasts_do_not_count_twice(self, evaluate, memory_store):
        _, first = evaluate(0)
        # made two days later: its window overlaps the first forecast's window
        snap, overlap = evaluate(0, as_of=week(0) + timedelta(days=2))
        assert [a.action for a in overlap] == ["skipped"]
        assert "overlaps" in overlap[0].note
        assert _lesson(memory_store, first, evaluated_at(snap)).evidence_count == 1

    def test_noise_forecasts_do_not_confirm(self, evaluate, memory_store):
        _, first = evaluate(0)
        snap, actions = evaluate(1, final_pct=0.5, amplitude=0.6)
        # vol ratio ~0.9 is neither a confirmation (>= 1.2) nor a postmortem lesson,
        # but it is evidence against "volatility is underestimated"
        assert [a.action for a in actions] == ["contradicted"]
        assert _lesson(memory_store, first, evaluated_at(snap)).evidence_count == 1


class TestRetirement:
    def test_contradictions_retire_a_lesson(self, evaluate, memory_store):
        _, first = evaluate(0)
        _, second = evaluate(1, final_pct=0.5, amplitude=0.6)  # calm week: contradiction
        assert second[0].action == "contradicted"
        assert second[0].status_after == LessonStatus.RETIRED  # 1 confirmation, 1 contradiction
        snap, later = evaluate(2)
        # a retired lesson is not revived: the same pattern starts a new candidate
        assert [a.action for a in later] == ["created"]
        assert later[0].lesson_id != first[0].lesson_id

    def test_active_lesson_retires_once_contradictions_catch_up(self, evaluate, memory_store):
        for n in range(3):
            _, actions = evaluate(n)
        assert actions[0].status_after == LessonStatus.ACTIVE
        statuses = [evaluate(n, final_pct=0.5, amplitude=0.6)[1][0].status_after for n in (3, 4, 5)]
        assert statuses == [LessonStatus.ACTIVE, LessonStatus.ACTIVE, LessonStatus.RETIRED]

    def test_stale_lessons_are_retired(self, evaluate, memory_store):
        snap, actions = evaluate(0)
        now = evaluated_at(snap)
        assert retire_stale_lessons(memory_store, now=now + timedelta(days=30)) == []
        retired = retire_stale_lessons(memory_store, now=now + timedelta(days=200))
        assert [a.action for a in retired] == ["retired"]
        assert "stale" in retired[0].note
        lesson = _lesson(memory_store, actions, now + timedelta(days=200))
        assert lesson.status == LessonStatus.RETIRED
        # point in time: before the retirement it was still a candidate
        assert _lesson(memory_store, actions, now).status == LessonStatus.CANDIDATE


def _event_llm():
    """An earnings lesson whose cause cites the corporate-action fact (hindsight only)."""

    def respond(prompt):
        action_id = next(
            line.split()[0] for line in prompt.splitlines() if "[corporate_action]" in line
        )
        return {
            "primary_cause": "earnings_or_corporate_event",
            "explanation": "A dividend fell in the window.",
            "knowable_at_forecast_time": [],
            "only_in_hindsight": [{"statement": "Dividend in window", "fact_ids": [action_id]}],
            "lessons": [EVENT_LESSON],
            "confidence": "medium",
        }

    return FakePostmortemLLM(respond)
