"""LLM call accounting: tokens, latency, estimated cost and a per-run call budget
(Plan.md §50-51, table ``llm_calls`` §37.16).

``LLMFactory`` reports every model call to the tracker active in the current
context (:func:`track_llm_usage`), tagged with the graph node that made it
(:func:`llm_node`). LangGraph runs each node in a copy of the caller's
context, so a tracker set around ``graph.invoke`` sees the calls of parallel
nodes too. Without a tracker, calls are still logged.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Optional

from pydantic import BaseModel, Field

from stock_analysis.config.settings import get_settings
from stock_analysis.logging import get_logger

logger = get_logger(__name__)


class LLMBudgetExceeded(RuntimeError):
    """The run already made its allowed number of LLM calls (``LLM_MAX_CALLS_PER_RUN``)."""


@dataclass(frozen=True)
class LLMCallRecord:
    """One call to one model (retries of the same call are counted in ``attempts``)."""

    node: Optional[str]
    role: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    cost_est: float
    status: str  # "ok" | "error"
    attempts: int = 1
    purpose: str = "generate"  # "generate" | "repair"
    error: Optional[str] = None
    run_id: Optional[str] = None
    ticker: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class LLMUsageSummary(BaseModel):
    """Totals over a set of LLM calls."""

    calls: int = 0
    failed_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_est: float = 0.0
    by_node: dict[str, int] = Field(default_factory=dict)  # node -> calls

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @classmethod
    def of(cls, records: list[LLMCallRecord]) -> LLMUsageSummary:
        by_node: dict[str, int] = {}
        for r in records:
            by_node[r.node or "unknown"] = by_node.get(r.node or "unknown", 0) + 1
        return cls(
            calls=len(records),
            failed_calls=sum(r.status != "ok" for r in records),
            prompt_tokens=sum(r.prompt_tokens for r in records),
            completion_tokens=sum(r.completion_tokens for r in records),
            cost_est=round(sum(r.cost_est for r in records), 6),
            by_node=by_node,
        )

    def describe(self) -> str:
        """One line for the CLI."""
        if not self.calls:
            return "LLM usage: no calls"
        failed = f" ({self.failed_calls} failed)" if self.failed_calls else ""
        return (
            f"LLM usage: {self.calls} calls{failed}, {self.prompt_tokens:,} prompt + "
            f"{self.completion_tokens:,} completion tokens, estimated cost ${self.cost_est:.4f}"
        )


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimated USD cost from the configured per-million-token prices."""
    s = get_settings()
    return (
        prompt_tokens * s.llm_input_cost_per_million_tokens
        + completion_tokens * s.llm_output_cost_per_million_tokens
    ) / 1_000_000


class LLMUsageTracker:
    """Collects the LLM calls of one run and enforces its call budget.

    ``sink`` (e.g. ``LLMCallStore.record``) receives each record as it is made;
    a failing sink is logged and never fails the LLM call.
    """

    def __init__(
        self,
        *,
        run_id: Optional[str] = None,
        ticker: Optional[str] = None,
        max_calls: Optional[int] = None,
        sink: Optional[Callable[[LLMCallRecord], None]] = None,
    ) -> None:
        self.run_id = run_id
        self.ticker = ticker
        self.max_calls = max_calls or None
        self.sink = sink
        self._records: list[LLMCallRecord] = []
        self._started = 0
        self._lock = threading.Lock()

    @property
    def records(self) -> list[LLMCallRecord]:
        with self._lock:
            return list(self._records)

    def start_call(self) -> None:
        """Reserve one call from the budget; raises ``LLMBudgetExceeded`` when spent."""
        with self._lock:
            if self.max_calls is not None and self._started >= self.max_calls:
                raise LLMBudgetExceeded(
                    f"LLM call budget of {self.max_calls} calls for this run is spent"
                )
            self._started += 1

    def record(self, record: LLMCallRecord) -> None:
        record = _with_run(record, self.run_id, self.ticker)
        with self._lock:
            self._records.append(record)
        if self.sink is not None:
            try:
                self.sink(record)
            except Exception as err:  # accounting must never break a forecast
                logger.warning("llm_call_not_stored", error=str(err))

    def summary(self) -> LLMUsageSummary:
        return LLMUsageSummary.of(self.records)


def _with_run(record: LLMCallRecord, run_id: Optional[str], ticker: Optional[str]) -> LLMCallRecord:
    if record.run_id == run_id and record.ticker == ticker:
        return record
    return replace(record, run_id=record.run_id or run_id, ticker=record.ticker or ticker)


_tracker: ContextVar[Optional[LLMUsageTracker]] = ContextVar("llm_usage_tracker", default=None)
_node: ContextVar[Optional[str]] = ContextVar("llm_node", default=None)


def current_tracker() -> Optional[LLMUsageTracker]:
    return _tracker.get()


def current_node() -> Optional[str]:
    return _node.get()


@contextmanager
def track_llm_usage(tracker: LLMUsageTracker) -> Iterator[LLMUsageTracker]:
    """Report every LLM call made inside the block (and in nodes it runs) to ``tracker``."""
    token = _tracker.set(tracker)
    try:
        yield tracker
    finally:
        _tracker.reset(token)


@contextmanager
def llm_node(node: str) -> Iterator[None]:
    """Tag LLM calls made inside the block with graph node ``node``."""
    token = _node.set(node)
    try:
        yield
    finally:
        _node.reset(token)
