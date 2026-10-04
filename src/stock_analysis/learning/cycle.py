"""The learning loop run after outcome review (Plan.md §71).

For every scored original forecast without a postmortem, oldest first:

1. diagnose it (``run_postmortem``; ``diagnose``);
2. record its decision-engine decisions with how the forecast did (for later
   evaluation; no decision rule is changed automatically) and apply any lessons
   to the lesson lifecycle (``apply``);
3. store the postmortem last, which marks the forecast as processed.

Then stale lessons are retired and each affected ticker's calibration is
re-estimated (``recalibrate``). Steps 1-2 are idempotent per forecast, so a run
interrupted before step 3 is safely repeated. ``run`` does all of it; the review
graph (``langgraph.review_graph``) runs the steps as separate nodes.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field

from stock_analysis.database.forecast_store import ForecastSnapshotError, ForecastSnapshotStore
from stock_analysis.database.learning_store import LearningStore, LearningStoreError
from stock_analysis.database.memory_store import MemoryStore, MemoryStoreError
from stock_analysis.database.outcome_store import OutcomeStore
from stock_analysis.learning.calibration import update_calibration
from stock_analysis.learning.lessons import LessonAction, apply_lessons, retire_stale_lessons
from stock_analysis.learning.postmortem import run_postmortem
from stock_analysis.logging import get_logger
from stock_analysis.schemas.learning import (
    CalibrationUpdate,
    CauseCategory,
    DecisionOutcome,
    HindsightNewsItem,
    PostMortem,
)
from stock_analysis.schemas.outcome import ForecastOutcome
from stock_analysis.schemas.snapshot import ForecastSnapshot

logger = get_logger(__name__)


class HindsightNewsSource(Protocol):
    """News published between two instants (only items inside the window are used)."""

    def fetch(self, symbol: str, start: datetime, end: datetime) -> Iterable[HindsightNewsItem]: ...


def decision_outcomes_for(
    snapshot: ForecastSnapshot,
    outcome: ForecastOutcome,
    *,
    cause: Optional[CauseCategory],
    now: datetime,
) -> list[DecisionOutcome]:
    """One record per decision in the snapshot, joined with the scored outcome."""
    final = snapshot.final_forecast
    return [
        DecisionOutcome(
            forecast_id=snapshot.forecast_id,
            decision_type=d.decision_type.value,
            ticker=snapshot.ticker,
            as_of_date=snapshot.as_of_date,
            target_date=snapshot.target_date,
            decision_engine=d.decision_engine,
            decision_model=d.decision_model,
            decision_model_version=d.decision_model_version,
            decision=d.decision,
            confidence=d.confidence,
            adjustment_applied=final.adjustment_applied,
            fallback_to_quant=final.fallback_to_quant,
            calibration_version=snapshot.calibration_version,
            direction_correct=outcome.direction_correct,
            in_80pct_band=outcome.in_80pct_band,
            final_loss=outcome.final_loss,
            baseline_loss=outcome.baseline_loss,
            uncalibrated_loss=outcome.uncalibrated_loss,
            llm_value_added=outcome.llm_value_added,
            llm_value_added_pinball=outcome.llm_value_added_pinball,
            calibration_value_added=outcome.calibration_value_added,
            primary_cause=cause,
            recorded_at=now,
        )
        for d in snapshot.decisions
    ]


class LearningReport(BaseModel):
    """What one learning pass did."""

    model_config = ConfigDict(extra="forbid")

    postmortems: list[PostMortem] = Field(default_factory=list)
    lesson_actions: list[LessonAction] = Field(default_factory=list)
    calibration_updates: list[CalibrationUpdate] = Field(default_factory=list)
    decision_outcomes_recorded: int = 0
    errors: list[str] = Field(default_factory=list)


class LearningCycle:
    """Postmortems, lessons, decision outcomes and calibration after outcome review."""

    def __init__(
        self,
        snapshot_store: ForecastSnapshotStore,
        outcome_store: OutcomeStore,
        memory_store: MemoryStore,
        learning_store: LearningStore,
        *,
        llm_factory: Any = None,
        news_source: Optional[HindsightNewsSource] = None,
    ):
        self.snapshot_store = snapshot_store
        self.outcome_store = outcome_store
        self.memory_store = memory_store
        self.learning_store = learning_store
        self.llm_factory = llm_factory
        self.news_source = news_source

    def _news(self, snapshot: ForecastSnapshot) -> list[HindsightNewsItem]:
        if self.news_source is None:
            return []
        from stock_analysis.snapshots.builder import IST, NSE_CLOSE_IST

        end = datetime.combine(snapshot.target_date, NSE_CLOSE_IST, tzinfo=IST)
        try:
            return list(self.news_source.fetch(snapshot.resolved_symbol, snapshot.made_at, end))
        except Exception as err:  # news is optional context; never block the postmortem
            logger.warning(
                "hindsight_news_failed", forecast_id=snapshot.forecast_id, error=str(err)
            )
            return []

    def run(self, now: datetime, *, ticker: Optional[str] = None) -> LearningReport:
        """Process every scored forecast without a postmortem, then recalibrate."""
        postmortems, tickers, errors = self.diagnose(now, ticker=ticker)
        report = self.apply(postmortems, now)
        updates, calibration_errors = self.recalibrate(sorted(tickers), now)
        return report.model_copy(
            update={
                "calibration_updates": updates,
                "errors": errors + report.errors + calibration_errors,
            }
        )

    def diagnose(
        self, now: datetime, *, ticker: Optional[str] = None
    ) -> tuple[list[PostMortem], set[str], list[str]]:
        """Postmortems (not yet stored) for every scored forecast without one, oldest first.

        Returns the postmortems, the tickers to recalibrate (``ticker`` plus every
        ticker with a pending forecast, even if its postmortem failed) and errors.
        """
        postmortems: list[PostMortem] = []
        tickers: set[str] = {ticker} if ticker else set()
        errors: list[str] = []
        for forecast_id in self.learning_store.pending_postmortems(before=now, ticker=ticker):
            try:
                snapshot = self.snapshot_store.get(forecast_id)
                outcome = self.outcome_store.get(forecast_id)
                if snapshot is None or outcome is None:
                    continue
                # Recalibrate this ticker even if its postmortem fails below
                tickers.add(snapshot.ticker)
                postmortems.append(
                    run_postmortem(
                        snapshot,
                        outcome,
                        llm_factory=self.llm_factory,
                        now=now,
                        news=self._news(snapshot),
                    )
                )
            except (
                ForecastSnapshotError,
                LearningStoreError,
                sqlite3.Error,
                ValueError,  # includes pydantic ValidationError and PostmortemError
            ) as err:
                # One bad record is reported and retried next run; it never blocks the others
                logger.error("learning_failed", forecast_id=forecast_id, error=str(err))
                errors.append(f"{forecast_id}: {err}")
        return postmortems, tickers, errors

    def apply(self, postmortems: list[PostMortem], now: datetime) -> LearningReport:
        """Record decision outcomes, apply lessons and store each postmortem, then retire
        stale lessons. ``report.postmortems`` holds only the postmortems that were stored."""
        report = LearningReport()
        for postmortem in postmortems:
            forecast_id = postmortem.forecast_id
            try:
                snapshot = self.snapshot_store.get(forecast_id)
                outcome = self.outcome_store.get(forecast_id)
                if snapshot is None or outcome is None:
                    raise LearningStoreError("forecast or its outcome is no longer readable")
                report.decision_outcomes_recorded += self.learning_store.record_decision_outcomes(
                    decision_outcomes_for(
                        snapshot, outcome, cause=postmortem.primary_cause, now=now
                    )
                )
                report.lesson_actions += apply_lessons(
                    self.memory_store, snapshot, outcome, postmortem, now=now
                )
                # Stored last: the postmortem marks the forecast as processed
                self.learning_store.save_postmortem(postmortem)
            except (
                ForecastSnapshotError,
                LearningStoreError,
                MemoryStoreError,
                sqlite3.Error,
                ValueError,
            ) as err:
                logger.error("learning_failed", forecast_id=forecast_id, error=str(err))
                report.errors.append(f"{forecast_id}: {err}")
                continue
            report.postmortems.append(postmortem)

        try:
            report.lesson_actions += retire_stale_lessons(self.memory_store, now=now)
        except (MemoryStoreError, sqlite3.Error) as err:
            report.errors.append(f"lesson retirement: {err}")
        return report

    def recalibrate(
        self, tickers: Iterable[str], now: datetime
    ) -> tuple[list[CalibrationUpdate], list[str]]:
        """Re-estimate each ticker's calibration; returns the updates and errors."""
        updates: list[CalibrationUpdate] = []
        errors: list[str] = []
        for name in tickers:
            try:
                updates.append(
                    update_calibration(
                        name,
                        now=now,
                        snapshot_store=self.snapshot_store,
                        outcome_store=self.outcome_store,
                        learning_store=self.learning_store,
                    )
                )
            except (ForecastSnapshotError, LearningStoreError, sqlite3.Error, ValueError) as err:
                logger.error("calibration_failed", ticker=name, error=str(err))
                errors.append(f"calibration {name}: {err}")
        return updates, errors
