"""Production entry point for one forecast run: always persists the snapshot and memory."""

from __future__ import annotations

from typing import Any, Optional

from stock_analysis.database import (
    ForecastSnapshotError,
    ForecastSnapshotStore,
    MemoryStore,
    get_database,
)
from stock_analysis.langgraph.workflow import compile_graph
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


def run_forecast(
    initial_state: GraphState,
    *,
    llm_factory: Any = None,
    store: Optional[ForecastSnapshotStore] = None,
    memory_store: Optional[MemoryStore] = None,
) -> GraphState:
    """Run the forecast graph and guarantee every completed forecast is stored.

    Both stores default to the configured application database. The run starts
    with prior context from memory and ends by writing this forecast's insight;
    a memory failure is logged (``memory_written``) but never fails the run.

    Raises:
        ForecastSnapshotError: the snapshot tables are missing, or a completed
            forecast was not persisted together with its price history.
        MemoryStoreError: the memory tables are missing.
    """
    store = store or ForecastSnapshotStore(get_database())
    memory_store = memory_store or MemoryStore(get_database())
    store.ensure_schema()
    memory_store.ensure_schema()

    result = compile_graph(llm_factory, store, memory_store).invoke(
        _with_company_name(initial_state)
    )
    final = result if isinstance(result, GraphState) else GraphState(**result)

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
