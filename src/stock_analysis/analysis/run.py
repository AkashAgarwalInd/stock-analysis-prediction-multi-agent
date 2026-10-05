"""``stock-analysis analyze``: one live forecast with its review, adaptation and track
record (Plan.md Phase 17 / §56 part 6).

The forecast itself runs through ``langgraph.runner.run_forecast``, so matured
forecasts are reviewed first, the calibration is applied and the snapshot is
stored; the report (``snapshots.report``) is the demo output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from stock_analysis.analysis.inputs import InputSource, LiveInputSource, collect_inputs
from stock_analysis.config.settings import get_settings
from stock_analysis.database import (
    Database,
    ForecastSnapshotStore,
    LearningStore,
    MemoryStore,
    OutcomeStore,
)
from stock_analysis.langgraph.runner import run_forecast
from stock_analysis.llm.factory import DisabledLLM
from stock_analysis.llm.usage import LLMUsageSummary, LLMUsageTracker
from stock_analysis.logging import get_logger
from stock_analysis.market.resolver import resolve_nse_ticker
from stock_analysis.schemas.graph_state import GraphState

logger = get_logger(__name__)

ANALYSTS = ("technical", "fundamental", "sentiment", "context")
QUANT_ONLY_REASON = "quant-only run (LLM disabled)"


@dataclass
class AnalysisResult:
    """A finished ``analyze`` run; ``report`` is None when no forecast was completed."""

    ticker: str
    resolved_symbol: str
    company_name: str
    state: GraphState
    unavailable_inputs: dict[str, str] = field(default_factory=dict)
    llm_usage: LLMUsageSummary = field(default_factory=LLMUsageSummary)

    @property
    def report(self) -> Optional[str]:
        return self.state.forecast_report

    @property
    def error(self) -> Optional[str]:
        """Why no forecast was completed (None when one was)."""
        if self.report is not None:
            return None
        forecast = self.state.final_forecast or {}
        quant = self.state.quant_baseline or {}
        # the quant baseline's error is the root cause when it failed (e.g. no prices)
        return str(quant.get("error") or forecast.get("error") or "the forecast was not completed")


def run_analysis(
    query: str,
    db: Database,
    *,
    use_llm: bool = True,
    review: bool = True,
    llm_factory: Any = None,
    source: Optional[InputSource] = None,
    run_at: Optional[datetime] = None,
    **review_options: Any,
) -> AnalysisResult:
    """Resolve ``query`` (ticker or company name), collect live inputs and run the forecast.

    Everything is stored in ``db`` (its snapshot, memory, outcome and learning
    tables must exist). ``use_llm=False`` is a quant-only run: all four
    analysts are disabled, nothing calls an LLM, and postmortems are rules only.
    ``run_at`` sets the run's clock (default: now). ``review_options`` go to the
    review graph (``price_source``, ``calendar``, ``news_source``).

    Raises:
        ForecastSnapshotError, MemoryStoreError, LearningStoreError,
        OutcomeStoreError: the database is missing tables, or a completed
            forecast could not be stored.
    """
    ticker, symbol, company = resolve_nse_ticker(query)
    disabled = {} if use_llm else dict.fromkeys(ANALYSTS, QUANT_ONLY_REASON)
    inputs = collect_inputs(
        ticker,
        symbol,
        company,
        source or LiveInputSource(db),
        run_at=run_at,
        disabled_analysts=disabled,
        calendar=review_options.get("calendar"),
    )
    bars = inputs.bars

    def prices(fetch_symbol: str) -> list[Any]:
        # The quant baseline uses the bars the technical indicators were computed from
        if fetch_symbol != symbol:
            raise ValueError(f"Unexpected price request for {fetch_symbol}")
        if not bars:
            raise ValueError(inputs.unavailable.get("prices", "no price history was returned"))
        return bars

    logger.info("analysis_started", query=query, symbol=symbol, llm=use_llm)
    tracker = LLMUsageTracker(ticker=ticker, max_calls=get_settings().llm_max_calls_per_run)
    final = run_forecast(
        inputs.state,
        llm_factory=llm_factory if use_llm else DisabledLLM(),
        store=ForecastSnapshotStore(db),
        memory_store=MemoryStore(db),
        learning_store=LearningStore(db),
        outcome_store=OutcomeStore(db),
        price_fetcher=prices,
        review=review,
        review_llm=use_llm,
        usage_tracker=tracker,
        **review_options,
    )
    return AnalysisResult(
        ticker=ticker,
        resolved_symbol=symbol,
        company_name=company,
        state=final,
        unavailable_inputs=inputs.unavailable,
        llm_usage=tracker.summary(),
    )
