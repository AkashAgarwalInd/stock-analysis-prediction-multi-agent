"""Long-term memory store: forecast insights, lessons and track record (Plan.md §36.2).

Lives in the application database next to the forecast snapshots it refers
to. Every table is append-only (alembic revision 003). Reads take an ``as_of``
cutoff so a backtest only sees memory that existed at forecast time;
timestamps are compared with SQLite ``julianday`` (millisecond resolution),
which accepts both the ``Z`` and ``+00:00`` UTC suffixes in use.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, date, datetime
from typing import Optional
from uuid import uuid4

from stock_analysis.config.settings import get_settings
from stock_analysis.database.database import Database
from stock_analysis.logging import get_logger
from stock_analysis.schemas.memory import (
    ForecastInsight,
    Lesson,
    LessonCandidate,
    LessonEvidenceKind,
    LessonStatus,
    TrackRecord,
    lesson_status,
)
from stock_analysis.schemas.snapshot import canonical_json

logger = get_logger(__name__)

_REQUIRED_TABLES = ("memory_insights", "lessons", "lesson_evidence", "forecast_outcomes")
_SCOPE_PRIORITY = {"ticker": 0, "sector": 1, "general": 2}


class MemoryStoreError(Exception):
    """Memory could not be read or written as requested."""


def _utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise MemoryStoreError("memory timestamps must be timezone-aware")
    return dt.astimezone(UTC).isoformat()


class MemoryStore:
    """Append-only long-term memory for forecasts."""

    def __init__(self, db: Database):
        self.db = db
        self._lock = threading.RLock()

    def ensure_schema(self) -> None:
        """Fail fast with a clear message if the memory migration has not been applied."""
        rows = self.db.fetchall("SELECT name FROM sqlite_master WHERE type = 'table'")
        missing = sorted(set(_REQUIRED_TABLES) - {r["name"] for r in rows})
        if missing:
            raise MemoryStoreError(
                f"Memory tables are missing ({', '.join(missing)}); "
                "run `stock-analysis migrate` (alembic upgrade head)"
            )

    # -- insights ---------------------------------------------------------

    def save_insight(self, insight: ForecastInsight) -> None:
        """Store the insight for a persisted forecast (one per forecast)."""
        try:
            with self._lock, self.db.transaction() as conn:
                conn.execute(
                    "INSERT INTO memory_insights (forecast_id, ticker, made_at, insight_json) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        insight.forecast_id,
                        insight.ticker,
                        _utc_iso(insight.made_at),
                        canonical_json(insight.model_dump(mode="json")),
                    ),
                )
        except sqlite3.IntegrityError as err:
            raise MemoryStoreError(
                f"Cannot store insight for forecast {insight.forecast_id}: {err}"
            ) from err
        logger.info("memory_insight_saved", forecast_id=insight.forecast_id, ticker=insight.ticker)

    def recent_insights(
        self, ticker: str, *, before: datetime, limit: int
    ) -> list[ForecastInsight]:
        """Insights for ``ticker`` from forecasts made strictly before ``before``, newest first."""
        rows = self.db.fetchall(
            "SELECT insight_json FROM memory_insights WHERE ticker = ? "
            "AND julianday(made_at) < julianday(?) ORDER BY julianday(made_at) DESC LIMIT ?",
            (ticker, _utc_iso(before), limit),
        )
        return [ForecastInsight.model_validate(json.loads(r["insight_json"])) for r in rows]

    # -- lessons ----------------------------------------------------------

    def add_lesson(
        self,
        candidate: LessonCandidate,
        *,
        source_forecast_id: str,
        first_seen: Optional[datetime] = None,
    ) -> Lesson:
        """Record a new candidate lesson; its source forecast counts as the first confirmation."""
        lesson_id = uuid4().hex
        seen = _utc_iso(first_seen or datetime.now(UTC))
        try:
            with self._lock, self.db.transaction() as conn:
                conn.execute(
                    "INSERT INTO lessons (lesson_id, scope, ticker, sector, text, category, "
                    "source_forecast_id, first_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        lesson_id,
                        candidate.scope,
                        candidate.ticker,
                        candidate.sector,
                        candidate.text,
                        candidate.category,
                        source_forecast_id,
                        seen,
                    ),
                )
                conn.execute(
                    "INSERT INTO lesson_evidence (lesson_id, kind, forecast_id, recorded_at) "
                    "VALUES (?, ?, ?, ?)",
                    (lesson_id, LessonEvidenceKind.CONFIRMED.value, source_forecast_id, seen),
                )
        except sqlite3.IntegrityError as err:
            raise MemoryStoreError(f"Cannot store lesson: {err}") from err
        return self._require_lesson(lesson_id)

    def record_evidence(
        self,
        lesson_id: str,
        kind: LessonEvidenceKind,
        *,
        forecast_id: Optional[str] = None,
        note: Optional[str] = None,
        recorded_at: Optional[datetime] = None,
    ) -> Lesson:
        """Append a confirmation, contradiction or retirement to a lesson's evidence log."""
        with self._lock:
            lesson = self.get_lesson(lesson_id)
            if lesson is None:
                raise MemoryStoreError(f"Unknown lesson {lesson_id}")
            if lesson.status == LessonStatus.RETIRED:
                raise MemoryStoreError(f"Lesson {lesson_id} is retired")
            if kind != LessonEvidenceKind.RETIRED and not forecast_id:
                raise MemoryStoreError("confirmations and contradictions must cite a forecast")
            if kind == LessonEvidenceKind.RETIRED and not note:
                raise MemoryStoreError("retiring a lesson requires a reason")
            try:
                self.db.execute(
                    "INSERT INTO lesson_evidence (lesson_id, kind, forecast_id, note, recorded_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        lesson_id,
                        kind.value,
                        forecast_id,
                        note,
                        _utc_iso(recorded_at or datetime.now(UTC)),
                    ),
                )
            except sqlite3.IntegrityError as err:
                raise MemoryStoreError(f"Cannot record lesson evidence: {err}") from err
            return self._require_lesson(lesson_id)

    def get_lesson(self, lesson_id: str, *, as_of: Optional[datetime] = None) -> Optional[Lesson]:
        """A lesson with its status derived from evidence recorded up to ``as_of``."""
        cutoff = _utc_iso(as_of or datetime.now(UTC))
        row = self.db.fetchone(
            "SELECT * FROM lessons WHERE lesson_id = ? AND julianday(first_seen) <= julianday(?)",
            (lesson_id, cutoff),
        )
        if row is None:
            return None
        evidence = self.db.fetchall(
            "SELECT kind, forecast_id, recorded_at FROM lesson_evidence WHERE lesson_id = ? "
            "AND julianday(recorded_at) <= julianday(?) ORDER BY julianday(recorded_at), id",
            (lesson_id, cutoff),
        )
        confirmed = [e for e in evidence if e["kind"] == LessonEvidenceKind.CONFIRMED.value]
        contradicted = sum(e["kind"] == LessonEvidenceKind.CONTRADICTED.value for e in evidence)
        retired = any(e["kind"] == LessonEvidenceKind.RETIRED.value for e in evidence)
        settings = get_settings()
        return Lesson(
            lesson_id=row["lesson_id"],
            text=row["text"],
            scope=row["scope"],
            category=row["category"],
            ticker=row["ticker"],
            sector=row["sector"],
            status=lesson_status(
                category=row["category"],
                evidence_count=len(confirmed),
                contradicted_count=contradicted,
                retired=retired,
                min_evidence=settings.lesson_activation_min_evidence,
                event_min_evidence=settings.lesson_event_activation_min_evidence,
            ),
            evidence_count=len(confirmed),
            contradicted_count=contradicted,
            source_forecast_ids=[e["forecast_id"] for e in confirmed],
            first_seen=row["first_seen"],
            last_confirmed=confirmed[-1]["recorded_at"] if confirmed else None,
        )

    def _require_lesson(self, lesson_id: str) -> Lesson:
        lesson = self.get_lesson(lesson_id)
        if lesson is None:
            raise MemoryStoreError(f"Lesson {lesson_id} was not found after writing it")
        return lesson

    def active_lessons(
        self, ticker: str, sector: Optional[str], *, as_of: datetime, limit: int
    ) -> list[Lesson]:
        """Active lessons that apply to ``ticker``/``sector``, most specific and best supported first."""
        rows = self.db.fetchall(
            "SELECT lesson_id FROM lessons WHERE julianday(first_seen) <= julianday(?) AND ("
            "scope = 'general' OR (scope = 'ticker' AND ticker = ?) "
            "OR (scope = 'sector' AND sector = ?))",
            (_utc_iso(as_of), ticker, sector),
        )
        lessons = [
            lesson
            for r in rows
            if (lesson := self.get_lesson(r["lesson_id"], as_of=as_of)) is not None
            and lesson.status == LessonStatus.ACTIVE
        ]
        lessons.sort(
            key=lambda lesson: (
                _SCOPE_PRIORITY[lesson.scope],
                -lesson.evidence_count,
                lesson.contradicted_count,
                lesson.lesson_id,
            )
        )
        return lessons[:limit]

    # -- track record -----------------------------------------------------

    def track_record(self, ticker: str, *, before: datetime, as_of_date: date) -> TrackRecord:
        """Counts of original forecasts made, scored and awaiting evaluation before ``before``.

        Only outcomes evaluated before the cutoff count, so the record is point-in-time.
        """
        cutoff = _utc_iso(before)
        row = self.db.fetchone(
            "SELECT COUNT(*) AS made, "
            "COALESCE(SUM(o.status = 'scored'), 0) AS scored, "
            "COALESCE(SUM(s.target_date < ? AND o.forecast_id IS NULL), 0) AS awaiting "
            "FROM forecast_snapshots s LEFT JOIN forecast_outcomes o "
            "ON o.forecast_id = s.forecast_id AND julianday(o.evaluated_at) < julianday(?) "
            "WHERE s.ticker = ? AND s.version = 1 AND julianday(s.made_at) < julianday(?)",
            (as_of_date.isoformat(), cutoff, ticker, cutoff),
        )
        return TrackRecord(
            ticker=ticker,
            forecasts_made=row["made"] if row else 0,
            forecasts_scored=row["scored"] if row else 0,
            forecasts_awaiting_outcome=row["awaiting"] if row else 0,
        )
