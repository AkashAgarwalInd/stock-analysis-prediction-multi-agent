"""Plan.md §32 / §68: a sequential, point-in-time historical backtest.

Weeks are processed strictly in order, each on the simulated clock::

    week k forecast (as of its as-of close, using only data up to then,
                     with the memory and calibration that existed then)
      -> actual prices for week k
      -> review graph: score -> postmortem -> lessons -> eligible calibration
         update -> scorecards -> track record
      -> week k+1 forecast ...

so no forecast is ever made with lessons or calibration learned later.
Consecutive windows share only an endpoint (the next as-of is the previous
target), so every week is independent evidence for learning.

A backtest writes to its own database by default, keeping simulated history
out of the live track record unless a database is chosen explicitly. Because
every record is written with its simulated (past) timestamp, a database that
already holds later history is refused: inserting the past behind it would
change what point-in-time reads of that later history return.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import structlog
from pydantic import BaseModel, ConfigDict, Field

from stock_analysis.backtest.data import (
    INDIA_VIX_SYMBOL,
    QUANT_HISTORY_DAYS,
    HistoricalMarketData,
    load_yfinance_history,
)
from stock_analysis.backtest.inputs import (
    DISABLED_IN_BACKTEST,
    market_context_summary,
    technical_summary,
)
from stock_analysis.config.settings import get_settings
from stock_analysis.database import (
    Database,
    ForecastSnapshotStore,
    LearningStore,
    LLMCallStore,
    MemoryStore,
    OutcomeStore,
)
from stock_analysis.database.forecast_store import ForecastSnapshotError
from stock_analysis.database.migrations import upgrade_database
from stock_analysis.langgraph.review_graph import compile_review_graph, invoke_review
from stock_analysis.langgraph.workflow import compile_graph
from stock_analysis.learning import build_evaluation, build_scorecards, compare_benchmarks
from stock_analysis.llm.factory import DisabledLLM
from stock_analysis.llm.usage import LLMUsageSummary, LLMUsageTracker, track_llm_usage
from stock_analysis.logging import get_logger
from stock_analysis.market.calendar import TradingCalendar, get_trading_calendar
from stock_analysis.market.resolver import resolve_nse_ticker
from stock_analysis.review import last_completed_trading_date
from stock_analysis.review.prices import NIFTY_50_SYMBOL
from stock_analysis.schemas.graph_state import GraphState
from stock_analysis.schemas.learning import BenchmarkComparison, CalibrationParams
from stock_analysis.schemas.outcome import OutcomeStatus
from stock_analysis.schemas.scorecard import Scorecards
from stock_analysis.schemas.track_record import EvaluationReport
from stock_analysis.snapshots.builder import IST, NSE_CLOSE_IST

logger = get_logger(__name__)

# Forecasts are made a few minutes after the as-of session's end-of-day data is out;
# the previous week is evaluated at that data-publication time, just before.
_FORECAST_AFTER_EVALUATION = timedelta(minutes=5)
# History needed before the first as-of date: the quant/indicator span plus a margin
_HISTORY_MARGIN_DAYS = 30
ALL_ANALYSTS = ("technical", "fundamental", "sentiment", "context")


class BacktestError(RuntimeError):
    """The backtest cannot run as requested."""


# Timestamped, append-only records a backtest writes, by table and time column
_HISTORY_COLUMNS = {
    "forecast_snapshots": "made_at",
    "forecast_outcomes": "evaluated_at",
    "lessons": "first_seen",
    "lesson_evidence": "recorded_at",
    "calibration_params": "created_at",
    "forecast_postmortems": "created_at",
}


def later_history(db: Database, since: datetime) -> dict[str, int]:
    """Count records timestamped at or after ``since``, per table (empty if none)."""
    cutoff = since.astimezone(UTC).isoformat()
    counts = {}
    for table, column in _HISTORY_COLUMNS.items():
        row = db.fetchone(
            f"SELECT COUNT(*) AS n FROM {table} WHERE julianday({column}) >= julianday(?)",
            (cutoff,),
        )
        if row and row["n"]:
            counts[table] = row["n"]
    return counts


class BacktestWeek(BaseModel):
    """What happened in one simulated week."""

    model_config = ConfigDict(extra="forbid")

    as_of_date: date
    target_date: date
    forecast_made_at: datetime
    evaluated_at: datetime
    forecast_id: Optional[str] = None
    calibration_version: int = 0
    status: str = Field(description="scored, invalid, unresolved or no_forecast")
    reason: Optional[str] = None
    last_close: Optional[float] = None
    p10_price: Optional[float] = None
    p50_price: Optional[float] = None
    p90_price: Optional[float] = None
    prob_up: Optional[float] = None
    actual_close: Optional[float] = None
    actual_return_pct: Optional[float] = None
    in_80pct_band: Optional[bool] = None
    direction_correct: Optional[bool] = None
    final_loss: Optional[float] = None
    baseline_loss: Optional[float] = None
    uncalibrated_loss: Optional[float] = None
    adjustment_applied: bool = False
    primary_cause: Optional[str] = None
    lessons_created: int = 0
    calibration_changed: bool = False


class BacktestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    resolved_symbol: str
    database_path: str
    llm_enabled: bool
    disabled_analysts: dict[str, str]
    weeks: list[BacktestWeek]
    calibration_history: list[CalibrationParams]
    active_lessons: int
    benchmarks: BenchmarkComparison
    scorecards: Scorecards
    evaluation: EvaluationReport
    errors: list[str] = Field(default_factory=list)
    llm_usage: LLMUsageSummary = Field(default_factory=LLMUsageSummary)


def evaluation_time(session: date) -> datetime:
    """When ``session``'s end-of-day data is available (close + configured delay)."""
    close = datetime.combine(session, NSE_CLOSE_IST, tzinfo=IST)
    return close + timedelta(minutes=get_settings().outcome_data_delay_minutes)


def forecast_time(as_of: date) -> datetime:
    return evaluation_time(as_of) + _FORECAST_AFTER_EVALUATION


def _shift_trading_days(day: date, n: int, calendar: TradingCalendar) -> date:
    for _ in range(n):
        prev = calendar.previous_trading_day(day)
        if prev is None:
            raise BacktestError(f"No trading day before {day}")
        day = prev
    return day


def backtest_schedule(
    weeks: int, last_target: date, *, horizon: int, calendar: TradingCalendar
) -> list[tuple[date, date]]:
    """``weeks`` back-to-back (as_of, target) windows ending on ``last_target``, oldest first."""
    if weeks < 1:
        raise BacktestError("weeks must be at least 1")
    if not calendar.is_trading_day(last_target):
        prev = calendar.previous_trading_day(last_target)
        if prev is None:
            raise BacktestError(f"No trading day on or before {last_target}")
        last_target = prev
    windows = []
    target = last_target
    for _ in range(weeks):
        as_of = _shift_trading_days(target, horizon, calendar)
        windows.append((as_of, target))
        target = as_of
    return windows[::-1]


def default_database_path(ticker: str, now: datetime) -> Path:
    return Path("data/backtests") / f"{ticker}-{now:%Y%m%d-%H%M%S}.db"


def run_backtest(
    ticker: str,
    weeks: int,
    *,
    database_path: Path,
    llm_factory: Any = None,
    use_llm: bool = True,
    end: Optional[date] = None,
    now: Optional[datetime] = None,
    calendar: Optional[TradingCalendar] = None,
    history_loader: Callable[..., HistoricalMarketData] = load_yfinance_history,
    progress: Optional[Callable[[BacktestWeek], None]] = None,
) -> BacktestResult:
    """Simulate ``weeks`` weekly forecasts for ``ticker`` ending at ``end`` (or the latest
    week whose target session has completed), learning between weeks.

    Raises:
        BacktestError: an invalid request, or the database already holds forecasts for
            this ticker inside the backtest period (a re-run would duplicate history).
    """
    settings = get_settings()
    calendar = calendar or get_trading_calendar()
    real_now = now or datetime.now(IST)
    latest = last_completed_trading_date(real_now, calendar)
    if end is not None and end > latest:
        raise BacktestError(f"End {end} is after the last completed session {latest}")
    schedule = backtest_schedule(
        weeks, end or latest, horizon=settings.forecast_horizon_days, calendar=calendar
    )
    base, symbol, company = resolve_nse_ticker(ticker)

    upgrade_database(database_path)
    db = Database(database_path)
    # Every LLM call of the backtest is logged to its database (Plan.md §51); no budget
    tracker = LLMUsageTracker(
        run_id=f"backtest-{uuid.uuid4().hex}", ticker=base, sink=LLMCallStore(db).record
    )
    try:
        with (
            track_llm_usage(tracker),
            structlog.contextvars.bound_contextvars(run_id=tracker.run_id, ticker=base),
        ):
            result = _run(
                base,
                symbol,
                company,
                schedule,
                db=db,
                database_path=database_path,
                llm_factory=llm_factory,
                use_llm=use_llm,
                calendar=calendar,
                history_loader=history_loader,
                progress=progress,
            )
        return result.model_copy(update={"llm_usage": tracker.summary()})
    finally:
        db.close()


def _run(
    base: str,
    symbol: str,
    company: str,
    schedule: list[tuple[date, date]],
    *,
    db: Database,
    database_path: Path,
    llm_factory: Any,
    use_llm: bool,
    calendar: TradingCalendar,
    history_loader: Callable[..., HistoricalMarketData],
    progress: Optional[Callable[[BacktestWeek], None]],
) -> BacktestResult:
    snapshots, outcomes = ForecastSnapshotStore(db), OutcomeStore(db)
    memory, learning = MemoryStore(db), LearningStore(db)
    for store in (snapshots, outcomes, memory, learning):
        store.ensure_schema()

    first_as_of = schedule[0][0]
    later = later_history(db, forecast_time(first_as_of))
    if later:
        found = ", ".join(f"{n} in {table}" for table, n in later.items())
        raise BacktestError(
            f"{database_path} already has history from {first_as_of} onwards ({found}); "
            "a backtest can only add history after what a database already holds — "
            "use a new database"
        )

    data = history_loader(
        [symbol, NIFTY_50_SYMBOL, INDIA_VIX_SYMBOL],
        first_as_of - timedelta(days=QUANT_HISTORY_DAYS + _HISTORY_MARGIN_DAYS),
        schedule[-1][1],
    )
    if data.last_date(symbol) is None:
        raise BacktestError(f"No price history could be loaded for {symbol}")
    if data.last_date(NIFTY_50_SYMBOL) is None:
        logger.warning("backtest_no_benchmark", symbol=NIFTY_50_SYMBOL)
    disabled = (
        dict(DISABLED_IN_BACKTEST)
        if use_llm
        else dict.fromkeys(ALL_ANALYSTS, "quant-only backtest (LLM disabled)")
    )
    graph_llm = llm_factory if use_llm else DisabledLLM()
    postmortem_llm = llm_factory if use_llm else None
    if use_llm and graph_llm is None:
        from stock_analysis.llm.factory import get_llm_factory

        graph_llm = postmortem_llm = get_llm_factory()

    prices = data.price_source()
    review_graph = compile_review_graph(
        snapshots,
        outcomes,
        memory,
        learning,
        price_source=prices,
        calendar=calendar,
        llm_factory=postmortem_llm,
    )
    results: list[BacktestWeek] = []
    errors: list[str] = []

    for as_of, target in schedule:
        week = BacktestWeek(
            as_of_date=as_of,
            target_date=target,
            forecast_made_at=forecast_time(as_of),
            evaluated_at=evaluation_time(target),
            status="no_forecast",
        )
        if not data.has_bar(symbol, as_of):
            week.reason = f"no {symbol} price bar on {as_of}"
        else:
            try:
                _forecast_week(
                    week,
                    base,
                    symbol,
                    company,
                    data,
                    disabled,
                    graph_llm,
                    snapshots,
                    learning,
                    memory,
                    outcomes,
                )
            except (ForecastSnapshotError, ValueError, sqlite3.Error) as err:
                # One failed week is reported; the simulation continues with the next
                logger.error("backtest_forecast_failed", as_of=str(as_of), error=str(err))
                week.reason = f"forecast failed: {err}"[:300]
                errors.append(f"{as_of}: {err}")
        # Review (score, learn, recalibrate) at the target session's data time,
        # before the next forecast
        prices.available_until = target
        review = invoke_review(review_graph, week.evaluated_at, ticker=base)
        for outcome in review.outcomes:
            if outcome.forecast_id == week.forecast_id:
                _record_outcome(week, outcome)
        errors += review.errors
        for pm in review.postmortems:
            if pm.forecast_id == week.forecast_id:
                week.primary_cause = pm.primary_cause.value
        week.lessons_created = sum(a.action == "created" for a in review.lesson_actions)
        week.calibration_changed = any(u.changed for u in review.calibration_updates)
        results.append(week)
        logger.info("backtest_week", **week.model_dump(mode="json"))
        if progress is not None:
            progress(week)

    # A week left unresolved (late data) may have been scored during a later week's review
    for week in results:
        if week.status == "unresolved" and week.forecast_id:
            stored = outcomes.get(week.forecast_id)
            if stored is not None:
                _record_outcome(week, stored)
    scored = [o for w in results if w.forecast_id and (o := outcomes.get(w.forecast_id))]
    final_time = results[-1].evaluated_at
    return BacktestResult(
        ticker=base,
        resolved_symbol=symbol,
        database_path=str(database_path),
        llm_enabled=use_llm,
        disabled_analysts=disabled,
        weeks=results,
        calibration_history=learning.calibration_history(base),
        active_lessons=len(memory.active_lessons(base, None, as_of=final_time, limit=1000)),
        benchmarks=compare_benchmarks(scored),
        scorecards=build_scorecards(outcomes, learning, as_of=final_time, ticker=base),
        evaluation=build_evaluation(snapshots, outcomes, as_of=final_time, ticker=base),
        errors=errors,
    )


def _forecast_week(
    week: BacktestWeek,
    base: str,
    symbol: str,
    company: str,
    data: HistoricalMarketData,
    disabled: dict[str, str],
    llm: Any,
    snapshots: ForecastSnapshotStore,
    learning: LearningStore,
    memory: MemoryStore,
    outcomes: OutcomeStore,
) -> None:
    """Make week k's forecast from the point-in-time view only."""
    view = data.as_of(week.as_of_date)
    state = GraphState(
        symbol=base,
        resolved_symbol=symbol,
        company_name=company,
        run_at=week.forecast_made_at,
        disabled_analysts=disabled,
        technical_indicators_summary=technical_summary(view, symbol),
        market_context_summary=market_context_summary(view, symbol),
    )
    # The week's review already ran (at the previous target's data time); the
    # graph only loads what it stored, as of this forecast's clock
    graph = compile_graph(
        llm,
        snapshots,
        memory,
        learning,
        price_fetcher=view.quant_prices,
        outcome_store=outcomes,
    )
    final = GraphState(**graph.invoke(state))
    if not final.snapshot_persisted or final.forecast_id is None:
        forecast = final.final_forecast or {}
        week.reason = str(forecast.get("error") or "forecast was not completed")
        return
    snapshot = snapshots.get(final.forecast_id)
    if snapshot is None:
        week.reason = "forecast snapshot missing after the run"
        return
    f = snapshot.final_forecast
    week.forecast_id = snapshot.forecast_id
    week.calibration_version = snapshot.calibration_version
    week.status = "unresolved"
    week.last_close = snapshot.last_close
    week.p10_price, week.p50_price, week.p90_price = f.p10_price, f.p50_price, f.p90_price
    week.prob_up = f.prob_up
    week.adjustment_applied = f.adjustment_applied


def _record_outcome(week: BacktestWeek, outcome: Any) -> None:
    week.status = outcome.status.value
    week.reason = outcome.invalid_reason
    if outcome.status == OutcomeStatus.SCORED:
        week.actual_close = outcome.actual_close_on_forecast_basis
        week.actual_return_pct = outcome.actual_return_pct
        week.in_80pct_band = outcome.in_80pct_band
        week.direction_correct = outcome.direction_correct
        week.final_loss = outcome.final_loss
        week.baseline_loss = outcome.baseline_loss
        week.uncalibrated_loss = outcome.uncalibrated_loss
