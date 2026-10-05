"""Production entry point for one forecast run: reviews the ticker's matured forecasts
first, always persists the snapshot and memory, and applies the ticker's current
calibration."""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

import structlog

from stock_analysis.config.settings import get_settings
from stock_analysis.database import (
    ForecastSnapshotError,
    ForecastSnapshotStore,
    LearningStore,
    LLMCallStore,
    MemoryStore,
    OutcomeStore,
    get_database,
)
from stock_analysis.langgraph.review_graph import make_pre_run_reviewer
from stock_analysis.langgraph.workflow import PriceFetcher, compile_graph
from stock_analysis.learning import RssHindsightNewsSource
from stock_analysis.llm.usage import LLMUsageTracker, track_llm_usage
from stock_analysis.logging import get_logger
from stock_analysis.market.resolver import SymbolResolver
from stock_analysis.schemas.graph_state import GraphState

logger = get_logger(__name__)


def _with_company_name(state: GraphState) -> GraphState:
    """Fill ``company_name`` from the offline symbol index when the caller did not."""
    if state.company_name:
        return state
    try:
        resolved = SymbolResolver(enable_yfinance_validation=False).resolve(state.symbol)
    except Exception as err:  # best effort: the snapshot falls back to the symbol
        logger.warning("company_name_lookup_failed", symbol=state.symbol, error=str(err))
        return state
    if resolved is None or not resolved.name:
        return state
    return state.model_copy(update={"company_name": resolved.name})


def _usage_tracker(
    tracker: Optional[LLMUsageTracker], store: ForecastSnapshotStore, symbol: str
) -> tuple[LLMUsageTracker, Optional[LLMCallStore]]:
    """The run's usage tracker, writing to ``llm_calls`` in the forecast database when the
    table exists (an older database only keeps the totals in memory)."""
    tracker = tracker or LLMUsageTracker(max_calls=get_settings().llm_max_calls_per_run)
    tracker.run_id = tracker.run_id or uuid.uuid4().hex
    tracker.ticker = tracker.ticker or symbol
    call_store = LLMCallStore(store.db)
    if not call_store.available():
        logger.warning("llm_call_log_unavailable", hint="run `stock-analysis migrate`")
        return tracker, None
    if tracker.sink is None:
        tracker.sink = call_store.record
    return tracker, call_store


def run_forecast(
    initial_state: GraphState,
    *,
    llm_factory: Any = None,
    store: Optional[ForecastSnapshotStore] = None,
    memory_store: Optional[MemoryStore] = None,
    learning_store: Optional[LearningStore] = None,
    outcome_store: Optional[OutcomeStore] = None,
    price_fetcher: Optional[PriceFetcher] = None,
    review: bool = True,
    review_llm: bool = True,
    usage_tracker: Optional[LLMUsageTracker] = None,
    **review_options: Any,
) -> GraphState:
    """Run the forecast graph and guarantee every completed forecast is stored.

    All stores default to the configured application database. The run first
    reviews the ticker's matured forecasts (Plan.md §5.2: score, postmortem,
    lessons, calibration; skipped with ``review=False``, and a failing review
    never blocks the forecast), then starts from prior context: earlier
    forecasts, the last forecast review, the calibration, active lessons,
    scorecards and the track record. It applies the latest calibration for the
    ticker (the uncalibrated baseline is stored alongside as the shadow
    forecast), and ends by writing this forecast's insight; a memory failure is
    logged (``memory_written``) but never fails the run.

    Postmortems use the forecast's LLM with hindsight news from RSS, as
    ``stock-analysis review`` does; ``review_llm=False`` makes them rules only
    (no LLM call, no news). ``review_options`` go to the review graph
    (``price_source``, ``calendar``, ``news_source``). ``price_fetcher``
    replaces the live price collector for the quant baseline, e.g. to reuse the
    bars the technical indicators were computed from.

    Every LLM call of the run (including the pre-run review's postmortems) is
    counted by ``usage_tracker`` (default: a new one) against
    ``LLM_MAX_CALLS_PER_RUN`` and stored in ``llm_calls`` when that table
    exists, linked to the stored forecast. Logs inside the run carry its
    ``run_id`` and ``ticker``.

    Raises:
        ForecastSnapshotError: the snapshot tables are missing, or a completed
            forecast was not persisted together with its price history.
        MemoryStoreError: the memory tables are missing.
        LearningStoreError: the learning tables are missing.
        OutcomeStoreError: the outcome tables are missing.
    """
    store = store or ForecastSnapshotStore(get_database())
    memory_store = memory_store or MemoryStore(get_database())
    learning_store = learning_store or LearningStore(get_database())
    outcome_store = outcome_store or OutcomeStore(get_database())
    for required in (store, memory_store, learning_store, outcome_store):
        required.ensure_schema()

    reviewer = None
    if review:
        postmortem_llm = None
        if review_llm:
            if llm_factory is None:
                from stock_analysis.llm.factory import get_llm_factory

                llm_factory = get_llm_factory()
            postmortem_llm = llm_factory
            review_options.setdefault("news_source", RssHindsightNewsSource())
        reviewer = make_pre_run_reviewer(
            store,
            outcome_store,
            memory_store,
            learning_store,
            llm_factory=postmortem_llm,
            **review_options,
        )
    graph = compile_graph(
        llm_factory,
        store,
        memory_store,
        learning_store,
        price_fetcher=price_fetcher,
        outcome_store=outcome_store,
        reviewer=reviewer,
    )
    tracker, call_store = _usage_tracker(usage_tracker, store, initial_state.symbol)
    started = time.perf_counter()
    with (
        track_llm_usage(tracker),
        structlog.contextvars.bound_contextvars(run_id=tracker.run_id, ticker=initial_state.symbol),
    ):
        result = graph.invoke(_with_company_name(initial_state))
        final = result if isinstance(result, GraphState) else GraphState(**result)
        usage = tracker.summary()
        logger.info(
            "forecast_run_completed",
            forecast_id=final.forecast_id,
            latency_ms=int((time.perf_counter() - started) * 1000),
            status="ok" if final.forecast_id else "incomplete",
            llm_calls=usage.calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            estimated_cost=round(usage.cost_est, 6),
        )
    if call_store is not None and final.forecast_id and tracker.run_id:
        try:
            call_store.attach_forecast(tracker.run_id, final.forecast_id)
        except Exception as err:  # telemetry never fails a stored forecast
            logger.warning("llm_calls_not_linked", error=str(err))

    forecast = final.final_forecast
    if not forecast or "error" in forecast:
        logger.warning("forecast_not_completed", symbol=final.symbol)
        return final

    if not final.snapshot_persisted or final.forecast_id is None:
        raise ForecastSnapshotError(f"Completed forecast for {final.symbol} was not persisted")
    sha = (final.price_snapshot or {}).get("history_sha256")
    if not sha or store.get_price_history(sha) is None:
        raise ForecastSnapshotError(
            f"Price history behind forecast {final.forecast_id} was not persisted"
        )
    if not final.memory_written:
        logger.warning("memory_insight_not_written", forecast_id=final.forecast_id)
    return final
