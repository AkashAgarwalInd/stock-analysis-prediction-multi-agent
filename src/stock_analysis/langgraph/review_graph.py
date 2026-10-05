"""Plan.md Phase 15: the review graph, run independently of the forecast graph.

Topology (one LangGraph node per step, strictly sequential)::

    find_matured → score → postmortem → learning → calibration
      → scorecards → track_record → END

- ``find_matured``: original forecasts whose target session has completed and
  that have no outcome yet (``OutcomeReviewer.find_matured``).
- ``score``: fetch actual prices for each forecast window and store scored or
  invalid outcomes; unresolved ones (and any that failed) are retried next run.
- ``postmortem``: diagnose every scored forecast without a postmortem (oldest
  first, not only those scored in this run).
- ``learning``: record decision outcomes, apply lessons and store each
  postmortem last (the "processed" marker), then retire stale lessons.
- ``calibration``: re-estimate the calibration of every affected ticker and
  compare the benchmarks it is judged on.
- ``scorecards`` / ``track_record``: recomputed from the immutable records as
  of the review time (nothing is stored; the forecast graph's memory loader
  recomputes them as of its own clock).

Every step is point-in-time (``ReviewState.run_at`` replaces the wall clock),
and a failing step records its error in ``ReviewState.errors`` instead of
raising, so a review never blocks whatever runs after it.
"""

from __future__ import annotations

import itertools
import sqlite3
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional, Protocol

import structlog
from langgraph.graph import END, StateGraph
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from stock_analysis.database import (
    ForecastSnapshotError,
    ForecastSnapshotStore,
    LearningStore,
    LearningStoreError,
    LLMCallStore,
    MemoryStore,
    MemoryStoreError,
    OutcomeStore,
    OutcomeStoreError,
    get_database,
)
from stock_analysis.langgraph.observability import instrument_node
from stock_analysis.learning import (
    HindsightNewsSource,
    LearningCycle,
    LessonAction,
    build_evaluation,
    build_scorecards,
    compare_benchmarks,
)
from stock_analysis.llm.usage import (
    LLMUsageSummary,
    LLMUsageTracker,
    current_tracker,
    track_llm_usage,
)
from stock_analysis.logging import get_logger
from stock_analysis.market.calendar import TradingCalendar
from stock_analysis.review import OutcomeReviewer, PriceSource
from stock_analysis.schemas.learning import BenchmarkComparison, CalibrationUpdate, PostMortem
from stock_analysis.schemas.memory import PreRunReview
from stock_analysis.schemas.outcome import ForecastOutcome, OutcomeStatus
from stock_analysis.schemas.scorecard import Scorecards
from stock_analysis.schemas.track_record import EvaluationReport

logger = get_logger(__name__)

REVIEW_STEPS = (
    "find_matured",
    "score",
    "postmortem",
    "learning",
    "calibration",
    "scorecards",
    "track_record",
)

# Errors a review step reports instead of raising (ValueError includes pydantic's
# ValidationError and PostmortemError)
_STEP_ERRORS = (
    ForecastSnapshotError,
    OutcomeStoreError,
    LearningStoreError,
    MemoryStoreError,
    sqlite3.Error,
    ValueError,
)


class ReviewState(BaseModel):
    """State of one review run. Inputs are ``run_at`` and the optional ``ticker``."""

    model_config = ConfigDict(extra="forbid")

    run_at: AwareDatetime
    ticker: Optional[str] = None

    matured_forecast_ids: list[str] = Field(default_factory=list)
    # Newly scored, invalid and unresolved (not stored, retried next run) outcomes
    outcomes: list[ForecastOutcome] = Field(default_factory=list)
    # Diagnosed by ``postmortem``; after ``learning`` only those that were stored
    postmortems: list[PostMortem] = Field(default_factory=list)
    lesson_actions: list[LessonAction] = Field(default_factory=list)
    decision_outcomes_recorded: int = 0
    recalibrate_tickers: list[str] = Field(default_factory=list)
    calibration_updates: list[CalibrationUpdate] = Field(default_factory=list)
    benchmarks: dict[str, BenchmarkComparison] = Field(default_factory=dict)
    scorecards: Optional[Scorecards] = None
    evaluation: Optional[EvaluationReport] = None

    completed_steps: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    # Set by ``run_review``: the postmortems' LLM calls, tokens and estimated cost
    llm_usage: Optional[LLMUsageSummary] = None

    def summary(self) -> PreRunReview:
        """The compact record a forecast run keeps of the review it ran first."""
        statuses = [o.status for o in self.outcomes]
        stored = [
            u.stored.version
            for u in self.calibration_updates
            if u.stored is not None and (self.ticker is None or u.ticker == self.ticker)
        ]
        return PreRunReview(
            run_at=self.run_at,
            matured=len(self.matured_forecast_ids),
            scored=statuses.count(OutcomeStatus.SCORED),
            invalid=statuses.count(OutcomeStatus.INVALID),
            unresolved=statuses.count(OutcomeStatus.UNRESOLVED),
            postmortems=len(self.postmortems),
            lessons_created=sum(a.action == "created" for a in self.lesson_actions),
            lessons_updated=sum(
                a.action in ("confirmed", "contradicted", "retired") for a in self.lesson_actions
            ),
            calibration_version=stored[-1] if stored else None,
            errors=[e[:300] for e in self.errors],
        )


class Node(Protocol):
    """A review graph node (LangGraph passes the state as ``state``)."""

    def __call__(self, state: ReviewState) -> dict[str, Any]: ...


def _step(name: str, body: Callable[[ReviewState], dict[str, Any]]) -> Node:
    """Wrap a step so a failure is recorded in ``errors`` and the review continues."""

    def node(state: ReviewState) -> dict[str, Any]:
        try:
            update = body(state)
        except _STEP_ERRORS as err:
            logger.error("review_step_failed", step=name, ticker=state.ticker, error=str(err))
            return {"errors": [*state.errors, f"{name}: {err}"]}
        errors = update.pop("errors", [])
        return {
            **update,
            "completed_steps": [*state.completed_steps, name],
            "errors": [*state.errors, *errors],
        }

    node.__name__ = f"{name}_node"
    return node


def build_review_graph(
    snapshot_store: ForecastSnapshotStore,
    outcome_store: OutcomeStore,
    memory_store: MemoryStore,
    learning_store: LearningStore,
    *,
    price_source: Optional[PriceSource] = None,
    calendar: Optional[TradingCalendar] = None,
    llm_factory: Any = None,
    news_source: Optional[HindsightNewsSource] = None,
) -> StateGraph[ReviewState]:
    """Build the review graph over the given stores.

    ``price_source`` defaults to yfinance and ``calendar`` to the NSE calendar;
    without ``llm_factory`` postmortems are rules only (no LLM call), and
    ``news_source`` adds hindsight news to them.
    """
    reviewer = OutcomeReviewer(snapshot_store, outcome_store, price_source, calendar)
    cycle = LearningCycle(
        snapshot_store,
        outcome_store,
        memory_store,
        learning_store,
        llm_factory=llm_factory,
        news_source=news_source,
    )

    def find_matured(state: ReviewState) -> dict[str, Any]:
        return {"matured_forecast_ids": reviewer.find_matured(state.run_at, ticker=state.ticker)}

    def score(state: ReviewState) -> dict[str, Any]:
        outcomes, errors = reviewer.score(state.matured_forecast_ids, state.run_at)
        return {"outcomes": outcomes, "errors": errors}

    def postmortem(state: ReviewState) -> dict[str, Any]:
        postmortems, tickers, errors = cycle.diagnose(state.run_at, ticker=state.ticker)
        return {
            "postmortems": postmortems,
            "recalibrate_tickers": sorted(tickers),
            "errors": errors,
        }

    def learning(state: ReviewState) -> dict[str, Any]:
        report = cycle.apply(state.postmortems, state.run_at)
        return {
            "postmortems": report.postmortems,
            "lesson_actions": report.lesson_actions,
            "decision_outcomes_recorded": report.decision_outcomes_recorded,
            "errors": report.errors,
        }

    def calibration(state: ReviewState) -> dict[str, Any]:
        # The requested ticker is recalibrated even when the postmortem step failed
        tickers = set(state.recalibrate_tickers) | ({state.ticker} if state.ticker else set())
        updates, errors = cycle.recalibrate(sorted(tickers), state.run_at)
        benchmarks = {}
        for update in updates:
            try:
                ids = learning_store.scored_forecast_ids(
                    update.ticker, before=state.run_at, limit=10_000
                )
                benchmarks[update.ticker] = compare_benchmarks(
                    [o for fid in ids if (o := outcome_store.get(fid)) is not None]
                )
            except _STEP_ERRORS as err:
                # The update is already stored; only its benchmark comparison is missing
                logger.error("benchmarks_failed", ticker=update.ticker, error=str(err))
                errors.append(f"benchmarks {update.ticker}: {err}")
        return {"calibration_updates": updates, "benchmarks": benchmarks, "errors": errors}

    def scorecards(state: ReviewState) -> dict[str, Any]:
        cards = build_scorecards(
            outcome_store, learning_store, as_of=state.run_at, ticker=state.ticker
        )
        return {"scorecards": cards}

    def track_record(state: ReviewState) -> dict[str, Any]:
        evaluation = build_evaluation(
            snapshot_store, outcome_store, as_of=state.run_at, ticker=state.ticker
        )
        return {"evaluation": evaluation}

    bodies = {
        "find_matured": find_matured,
        "score": score,
        "postmortem": postmortem,
        "learning": learning,
        "calibration": calibration,
        "scorecards": scorecards,
        "track_record": track_record,
    }
    graph = StateGraph(ReviewState)
    for name in REVIEW_STEPS:
        graph.add_node(name, instrument_node(name, _step(name, bodies[name])))
    graph.set_entry_point(REVIEW_STEPS[0])
    for current, following in itertools.pairwise(REVIEW_STEPS):
        graph.add_edge(current, following)
    graph.add_edge(REVIEW_STEPS[-1], END)
    return graph


def compile_review_graph(
    snapshot_store: ForecastSnapshotStore,
    outcome_store: OutcomeStore,
    memory_store: MemoryStore,
    learning_store: LearningStore,
    **kwargs: Any,
) -> Any:
    """Compile ``build_review_graph`` (same arguments)."""
    return build_review_graph(
        snapshot_store, outcome_store, memory_store, learning_store, **kwargs
    ).compile()


def invoke_review(graph: Any, run_at: datetime, *, ticker: Optional[str] = None) -> ReviewState:
    """Run a compiled review graph at ``run_at`` and return its final state."""
    result = graph.invoke(ReviewState(run_at=run_at, ticker=ticker))
    return result if isinstance(result, ReviewState) else ReviewState(**result)


def run_review(
    run_at: datetime,
    *,
    ticker: Optional[str] = None,
    snapshot_store: Optional[ForecastSnapshotStore] = None,
    outcome_store: Optional[OutcomeStore] = None,
    memory_store: Optional[MemoryStore] = None,
    learning_store: Optional[LearningStore] = None,
    usage_tracker: Optional[LLMUsageTracker] = None,
    **kwargs: Any,
) -> ReviewState:
    """Production entry point: review matured forecasts and summarize the track record.

    Stores default to the application database; ``kwargs`` go to
    ``build_review_graph`` (price source, calendar, LLM factory, news source).
    The postmortems' LLM calls are counted by ``usage_tracker`` (default: the
    active one, else a new ``review-…`` run logged to ``llm_calls`` when that
    table exists) and summarized in ``llm_usage``.

    Raises:
        ForecastSnapshotError, OutcomeStoreError, MemoryStoreError, LearningStoreError:
            the corresponding tables are missing.
    """
    snapshot_store = snapshot_store or ForecastSnapshotStore(get_database())
    outcome_store = outcome_store or OutcomeStore(get_database())
    memory_store = memory_store or MemoryStore(get_database())
    learning_store = learning_store or LearningStore(get_database())
    for store in (snapshot_store, outcome_store, memory_store, learning_store):
        store.ensure_schema()
    graph = compile_review_graph(
        snapshot_store, outcome_store, memory_store, learning_store, **kwargs
    )
    tracker = usage_tracker or current_tracker()
    if tracker is None:
        tracker = LLMUsageTracker(run_id=f"review-{uuid.uuid4().hex}", ticker=ticker)
        call_store = LLMCallStore(snapshot_store.db)
        if call_store.available():
            tracker.sink = call_store.record
    context = {"run_id": tracker.run_id, **({"ticker": ticker} if ticker else {})}
    already = len(tracker.records)  # a shared tracker may hold earlier calls
    with track_llm_usage(tracker), structlog.contextvars.bound_contextvars(**context):
        state = invoke_review(graph, run_at, ticker=ticker)
    usage = LLMUsageSummary.of(tracker.records[already:])
    state = state.model_copy(update={"llm_usage": usage})
    logger.info(
        "review_completed",
        ticker=ticker,
        matured=len(state.matured_forecast_ids),
        postmortems=len(state.postmortems),
        calibration_changed=sum(u.changed for u in state.calibration_updates),
        errors=len(state.errors),
    )
    return state


def make_pre_run_reviewer(
    snapshot_store: ForecastSnapshotStore,
    outcome_store: OutcomeStore,
    memory_store: MemoryStore,
    learning_store: LearningStore,
    **kwargs: Any,
) -> Callable[[datetime, str], PreRunReview]:
    """Plan.md §5.2 / Phase 16: a reviewer for the forecast graph's ``review_matured`` node.

    The review graph is compiled once (``kwargs`` as for ``build_review_graph``)
    and run for the forecast's ticker at the forecast's clock.
    """
    graph = compile_review_graph(
        snapshot_store, outcome_store, memory_store, learning_store, **kwargs
    )

    def review(run_at: datetime, ticker: str) -> PreRunReview:
        state = invoke_review(graph, run_at, ticker=ticker)
        logger.info(
            "pre_run_review_completed",
            ticker=ticker,
            matured=len(state.matured_forecast_ids),
            errors=len(state.errors),
        )
        return state.summary()

    return review
