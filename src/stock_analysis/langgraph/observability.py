"""Node-level observability shared by the forecast and review graphs (Plan.md §51)."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast

import structlog

from stock_analysis.llm.usage import llm_node
from stock_analysis.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def node_status(result: Any) -> str:
    """``error`` when a node returned an error marker in one of its outputs, else ``ok``."""
    if isinstance(result, dict):
        for value in result.values():
            if isinstance(value, dict) and value.get("error"):
                return "error"
    return "ok"


def instrument_node(name: str, fn: F) -> F:
    """Wrap a node: its logs and LLM calls are tagged with ``node``, and one structured
    ``node_completed`` / ``node_failed`` line records latency and status (Plan.md §51)."""

    @functools.wraps(fn)
    def node(state: Any) -> Any:
        started = time.perf_counter()
        ticker = getattr(state, "symbol", None) or getattr(state, "ticker", None)
        # a review of all tickers has none; keep the run's bound ticker then
        tag = {"ticker": ticker} if ticker else {}
        with llm_node(name), structlog.contextvars.bound_contextvars(node=name):
            try:
                result = fn(state)
            except Exception as err:
                logger.error(
                    "node_failed",
                    **tag,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    status="error",
                    error=f"{type(err).__name__}: {err}"[:300],
                )
                raise
            logger.info(
                "node_completed",
                **tag,
                latency_ms=int((time.perf_counter() - started) * 1000),
                status=node_status(result),
            )
        return result

    return cast(F, node)
