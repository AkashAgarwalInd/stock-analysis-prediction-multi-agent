"""Plan.md Phase 7: memory store — insights, lesson lifecycle and track record."""

import sqlite3
from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from stock_analysis.database import MemoryStoreError
from stock_analysis.schemas.memory import (
    ForecastInsight,
    LessonCandidate,
    LessonEvidenceKind,
    LessonStatus,
)
from stock_analysis.snapshots import build_forecast_snapshot

T0 = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture
def forecasts(adjusted_state, store):
    """Three persisted original forecasts made a week apart."""
    snaps = [
        build_forecast_snapshot(adjusted_state, made_at=T0 + timedelta(days=7 * i))
        for i in range(3)
    ]
    for snap in snaps:
        store.save(snap)
    return snaps


def _ticker_lesson(**overrides):
    data = {
        "text": "Weekly realized volatility exceeded the model estimate in high-volatility regimes",
        "scope": "ticker",
        "category": "volatility",
        "ticker": "RELIANCE",
    }
    return LessonCandidate(**{**data, **overrides})


class TestInsights:
    def test_insight_copies_snapshot_numbers(self, snapshot):
        insight = ForecastInsight.from_snapshot(snapshot)
        final = snapshot.final_forecast
        assert insight.forecast_id == snapshot.forecast_id
        assert (insight.prob_up, insight.p10_price, insight.p50_price, insight.p90_price) == (
            final.prob_up,
            final.p10_price,
            final.p50_price,
            final.p90_price,
        )
        assert insight.last_close == snapshot.last_close
        assert insight.key_drivers == final.evidence[:3]

    def test_recent_insights_newest_first_with_cutoff(self, memory_store, forecasts):
        for snap in forecasts:
            memory_store.save_insight(ForecastInsight.from_snapshot(snap))

        newest = memory_store.recent_insights("RELIANCE", before=T0 + timedelta(days=30), limit=2)
        assert [i.forecast_id for i in newest] == [
            forecasts[2].forecast_id,
            forecasts[1].forecast_id,
        ]

        # point-in-time: a run on day 8 must not see the forecast made on day 14
        early = memory_store.recent_insights("RELIANCE", before=T0 + timedelta(days=8), limit=5)
        assert [i.forecast_id for i in early] == [
            forecasts[1].forecast_id,
            forecasts[0].forecast_id,
        ]
        assert memory_store.recent_insights("TCS", before=T0 + timedelta(days=30), limit=5) == []

    def test_insight_requires_stored_forecast(self, memory_store, snapshot):
        with pytest.raises(MemoryStoreError):
            memory_store.save_insight(ForecastInsight.from_snapshot(snapshot))

    def test_one_insight_per_forecast(self, memory_store, store, snapshot):
        store.save(snapshot)
        memory_store.save_insight(ForecastInsight.from_snapshot(snapshot))
        with pytest.raises(MemoryStoreError):
            memory_store.save_insight(ForecastInsight.from_snapshot(snapshot))

    @pytest.mark.parametrize(
        "statement",
        [
            "UPDATE memory_insights SET insight_json = '{}'",
            "DELETE FROM memory_insights",
            "UPDATE lessons SET text = 'rewritten lesson text here'",
            "DELETE FROM lesson_evidence",
        ],
    )
    def test_memory_tables_are_append_only(self, memory_store, forecasts, migrated_db, statement):
        memory_store.save_insight(ForecastInsight.from_snapshot(forecasts[0]))
        memory_store.add_lesson(_ticker_lesson(), source_forecast_id=forecasts[0].forecast_id)
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            migrated_db.execute(statement)


class TestLessonLifecycle:
    def test_candidate_becomes_active_after_three_confirmations(self, memory_store, forecasts):
        lesson = memory_store.add_lesson(
            _ticker_lesson(), source_forecast_id=forecasts[0].forecast_id, first_seen=T0
        )
        assert lesson.status == LessonStatus.CANDIDATE
        assert lesson.evidence_count == 1

        lesson = memory_store.record_evidence(
            lesson.lesson_id,
            LessonEvidenceKind.CONFIRMED,
            forecast_id=forecasts[1].forecast_id,
            recorded_at=T0 + timedelta(days=7),
        )
        assert lesson.status == LessonStatus.CANDIDATE

        lesson = memory_store.record_evidence(
            lesson.lesson_id,
            LessonEvidenceKind.CONFIRMED,
            forecast_id=forecasts[2].forecast_id,
            recorded_at=T0 + timedelta(days=14),
        )
        assert lesson.status == LessonStatus.ACTIVE
        assert lesson.source_forecast_ids == [s.forecast_id for s in forecasts]
        assert lesson.last_confirmed == T0 + timedelta(days=14)

        # point-in-time: before the third confirmation it was still a candidate
        past = memory_store.get_lesson(lesson.lesson_id, as_of=T0 + timedelta(days=10))
        assert past.status == LessonStatus.CANDIDATE

    def test_event_lessons_activate_after_two(self, memory_store, forecasts):
        lesson = memory_store.add_lesson(
            _ticker_lesson(category="earnings"), source_forecast_id=forecasts[0].forecast_id
        )
        lesson = memory_store.record_evidence(
            lesson.lesson_id, LessonEvidenceKind.CONFIRMED, forecast_id=forecasts[1].forecast_id
        )
        assert lesson.status == LessonStatus.ACTIVE

    def test_contradictions_retire_a_lesson(self, memory_store, forecasts):
        lesson = memory_store.add_lesson(
            _ticker_lesson(), source_forecast_id=forecasts[0].forecast_id
        )
        lesson = memory_store.record_evidence(
            lesson.lesson_id, LessonEvidenceKind.CONTRADICTED, forecast_id=forecasts[1].forecast_id
        )
        assert lesson.status == LessonStatus.RETIRED
        assert lesson.contradicted_count == 1
        with pytest.raises(MemoryStoreError, match="retired"):
            memory_store.record_evidence(
                lesson.lesson_id, LessonEvidenceKind.CONFIRMED, forecast_id=forecasts[2].forecast_id
            )

    def test_explicit_retirement_needs_reason(self, memory_store, forecasts):
        lesson = memory_store.add_lesson(
            _ticker_lesson(), source_forecast_id=forecasts[0].forecast_id
        )
        with pytest.raises(MemoryStoreError, match="reason"):
            memory_store.record_evidence(lesson.lesson_id, LessonEvidenceKind.RETIRED)
        retired = memory_store.record_evidence(
            lesson.lesson_id, LessonEvidenceKind.RETIRED, note="Stale: regime ended"
        )
        assert retired.status == LessonStatus.RETIRED

    def test_confirmation_must_cite_forecast(self, memory_store, forecasts):
        lesson = memory_store.add_lesson(
            _ticker_lesson(), source_forecast_id=forecasts[0].forecast_id
        )
        with pytest.raises(MemoryStoreError, match="cite a forecast"):
            memory_store.record_evidence(lesson.lesson_id, LessonEvidenceKind.CONFIRMED)

    def test_candidate_validation(self):
        with pytest.raises(ValidationError, match="needs a ticker"):
            LessonCandidate(text="x" * 30, scope="ticker", category="volatility")
        with pytest.raises(ValidationError):
            _ticker_lesson(text="too short")


class TestActiveLessons:
    def _activate(self, memory_store, forecasts, candidate):
        lesson = memory_store.add_lesson(
            candidate, source_forecast_id=forecasts[0].forecast_id, first_seen=T0
        )
        for snap in forecasts[1:]:
            lesson = memory_store.record_evidence(
                lesson.lesson_id,
                LessonEvidenceKind.CONFIRMED,
                forecast_id=snap.forecast_id,
                recorded_at=T0 + timedelta(days=1),
            )
        return lesson

    def test_scope_matching_and_order(self, memory_store, forecasts):
        general = self._activate(
            memory_store,
            forecasts,
            LessonCandidate(
                text="Earnings weeks need wider bands than the EWMA estimate",
                scope="general",
                category="volatility",
            ),
        )
        sector = self._activate(
            memory_store,
            forecasts,
            LessonCandidate(
                text="Energy names track crude moves more than the model assumes",
                scope="sector",
                category="context",
                sector="Energy",
            ),
        )
        ticker = self._activate(memory_store, forecasts, _ticker_lesson())
        self._activate(memory_store, forecasts, _ticker_lesson(ticker="TCS"))
        memory_store.add_lesson(
            _ticker_lesson(text="A candidate lesson that is not yet supported by evidence"),
            source_forecast_id=forecasts[0].forecast_id,
            first_seen=T0,
        )

        as_of = T0 + timedelta(days=2)
        active = memory_store.active_lessons("RELIANCE", "Energy", as_of=as_of, limit=5)
        assert [lesson.lesson_id for lesson in active] == [
            ticker.lesson_id,
            sector.lesson_id,
            general.lesson_id,
        ]
        assert memory_store.active_lessons("RELIANCE", None, as_of=as_of, limit=1) == [ticker]
        assert memory_store.active_lessons("RELIANCE", "Energy", as_of=T0, limit=5) == []


class TestTrackRecord:
    def test_counts_original_forecasts_only(self, memory_store, store, forecasts):
        store.record_correction(
            forecasts[0].forecast_id, reason="typo", corrected_at=T0 + timedelta(days=1)
        )
        record = memory_store.track_record(
            "RELIANCE", before=T0 + timedelta(days=30), as_of_date=date(2026, 9, 20)
        )
        assert record.forecasts_made == 3
        # targets: about 5 trading days after each as_of; all as_of dates are 2026-09-29
        assert record.forecasts_awaiting_outcome == 0
        assert record.outcome_metrics is None

        later = memory_store.track_record(
            "RELIANCE", before=T0 + timedelta(days=30), as_of_date=date(2026, 10, 20)
        )
        assert later.forecasts_awaiting_outcome == 3

    def test_respects_cutoff(self, memory_store, forecasts):
        record = memory_store.track_record(
            "RELIANCE", before=T0 + timedelta(days=1), as_of_date=date(2026, 10, 20)
        )
        assert record.forecasts_made == 1

    def test_requires_schema(self, temp_db_path):
        from stock_analysis.database import Database, MemoryStore

        with pytest.raises(MemoryStoreError, match="stock-analysis migrate"):
            MemoryStore(Database(temp_db_path)).ensure_schema()
