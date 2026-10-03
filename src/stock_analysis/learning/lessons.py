"""Plan.md §19-20: turn postmortem lessons into evidence on the lesson lifecycle.

candidate -> active -> retired, all derived from the append-only evidence log
in ``MemoryStore``:

* A postmortem lesson either creates a candidate or confirms the open lesson
  with the same scope and category (lessons are matched on what they are about,
  not on the LLM's wording).
* Ticker-scoped volatility lessons also get deterministic feedback from later
  outcomes: realised/predicted volatility confirms or contradicts them.
* One forecast counts at most once per lesson, and never when its window
  overlaps a forecast already counted: overlapping forecasts see the same
  market move, so they are not independent evidence.
* Lessons not confirmed for ``LESSON_STALE_AFTER_DAYS`` are retired.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from stock_analysis.config.settings import get_settings
from stock_analysis.database.memory_store import MemoryStore
from stock_analysis.logging import get_logger
from stock_analysis.schemas.learning import CauseCategory, PostMortem
from stock_analysis.schemas.memory import Lesson, LessonEvidenceKind, LessonStatus
from stock_analysis.schemas.outcome import ForecastOutcome
from stock_analysis.schemas.snapshot import ForecastSnapshot

logger = get_logger(__name__)

# Realised/predicted volatility that confirms / contradicts a volatility lesson
_CONFIRM_UNDER, _CONTRADICT_UNDER = 1.2, 1.0
_CONFIRM_OVER, _CONTRADICT_OVER = 0.8, 1.0


class LessonAction(BaseModel):
    """What one evaluated forecast did to one lesson."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lesson_id: str
    action: Literal["created", "confirmed", "contradicted", "retired", "skipped"]
    forecast_id: Optional[str] = None
    status_after: LessonStatus
    note: Optional[str] = None


def _sector(snapshot: ForecastSnapshot) -> Optional[str]:
    sector = (snapshot.data_inputs.get("fundamentals") or {}).get("sector")
    return sector if isinstance(sector, str) and sector else None


def _overlaps(a_start: date, a_end: date, b_start: date, b_end: date) -> bool:
    return a_start < b_end and b_start < a_end


def volatility_verdict(lesson: Lesson, outcome: ForecastOutcome) -> Optional[LessonEvidenceKind]:
    """Deterministic evidence a scored outcome gives a ticker-scoped volatility lesson."""
    ratio = outcome.vol_ratio
    if ratio is None or lesson.scope != "ticker":
        return None
    if lesson.category == CauseCategory.VOLATILITY_UNDERESTIMATED.value:
        if ratio >= _CONFIRM_UNDER:
            return LessonEvidenceKind.CONFIRMED
        if ratio <= _CONTRADICT_UNDER:
            return LessonEvidenceKind.CONTRADICTED
    if lesson.category == CauseCategory.VOLATILITY_OVERESTIMATED.value:
        if ratio <= _CONFIRM_OVER:
            return LessonEvidenceKind.CONFIRMED
        if ratio >= _CONTRADICT_OVER:
            return LessonEvidenceKind.CONTRADICTED
    return None


def _record(
    store: MemoryStore,
    lesson: Lesson,
    kind: LessonEvidenceKind,
    snapshot: ForecastSnapshot,
    *,
    now: datetime,
) -> LessonAction:
    """Append evidence unless this forecast (or an overlapping one) already counted."""
    for prior in store.evidence_windows(lesson.lesson_id):
        if prior["forecast_id"] == snapshot.forecast_id:
            note = "this forecast already counted for the lesson"
        elif _overlaps(
            snapshot.as_of_date,
            snapshot.target_date,
            date.fromisoformat(prior["as_of_date"]),
            date.fromisoformat(prior["target_date"]),
        ):
            note = f"window overlaps forecast {prior['forecast_id']}, already counted"
        else:
            continue
        return LessonAction(
            lesson_id=lesson.lesson_id,
            action="skipped",
            forecast_id=snapshot.forecast_id,
            status_after=lesson.status,
            note=note,
        )
    updated = store.record_evidence(
        lesson.lesson_id, kind, forecast_id=snapshot.forecast_id, recorded_at=now
    )
    return LessonAction(
        lesson_id=lesson.lesson_id,
        action="confirmed" if kind == LessonEvidenceKind.CONFIRMED else "contradicted",
        forecast_id=snapshot.forecast_id,
        status_after=updated.status,
    )


def apply_lessons(
    store: MemoryStore,
    snapshot: ForecastSnapshot,
    outcome: ForecastOutcome,
    postmortem: PostMortem,
    *,
    now: datetime,
) -> list[LessonAction]:
    """Record what one evaluated forecast says about new and existing lessons."""
    sector = _sector(snapshot)
    open_lessons = store.open_lessons(snapshot.ticker, sector, as_of=now)
    actions: list[LessonAction] = []
    handled: set[str] = set()

    for candidate in postmortem.lessons:
        match = next(
            (
                lesson
                for lesson in open_lessons
                if lesson.scope == candidate.scope
                and lesson.category == candidate.category
                and lesson.ticker == candidate.ticker
                and lesson.sector == candidate.sector
            ),
            None,
        )
        if match is None:
            created = store.add_lesson(
                candidate, source_forecast_id=snapshot.forecast_id, first_seen=now
            )
            open_lessons.append(created)
            actions.append(
                LessonAction(
                    lesson_id=created.lesson_id,
                    action="created",
                    forecast_id=snapshot.forecast_id,
                    status_after=created.status,
                )
            )
            handled.add(created.lesson_id)
        elif match.lesson_id not in handled:
            actions.append(_record(store, match, LessonEvidenceKind.CONFIRMED, snapshot, now=now))
            handled.add(match.lesson_id)

    for lesson in open_lessons:
        if lesson.lesson_id in handled:
            continue
        verdict = volatility_verdict(lesson, outcome)
        if verdict is not None and lesson.ticker == snapshot.ticker:
            actions.append(_record(store, lesson, verdict, snapshot, now=now))

    for action in actions:
        logger.info("lesson_evidence", **action.model_dump(mode="json"))
    return actions


def retire_stale_lessons(
    store: MemoryStore, *, now: datetime, stale_after_days: Optional[int] = None
) -> list[LessonAction]:
    """Retire open lessons with no confirmation in the last ``stale_after_days``."""
    days = get_settings().lesson_stale_after_days if stale_after_days is None else stale_after_days
    cutoff = now - timedelta(days=days)
    actions = []
    for lesson in store.open_lessons(None, None, as_of=now):
        last = lesson.last_confirmed or lesson.first_seen
        if last >= cutoff:
            continue
        note = f"stale: not confirmed since {last.date().isoformat()} ({days}-day limit)"
        updated = store.record_evidence(
            lesson.lesson_id, LessonEvidenceKind.RETIRED, note=note, recorded_at=now
        )
        actions.append(
            LessonAction(
                lesson_id=lesson.lesson_id, action="retired", status_after=updated.status, note=note
            )
        )
        logger.info("lesson_retired", lesson_id=lesson.lesson_id, note=note)
    return actions
