"""Persistence for the LLM call log (Plan.md §37.16 ``llm_calls``)."""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Optional

from stock_analysis.database.database import Database
from stock_analysis.llm.usage import LLMCallRecord, LLMUsageSummary
from stock_analysis.logging import get_logger

logger = get_logger(__name__)

_COLUMNS = (
    "run_id",
    "ticker",
    "node",
    "role",
    "model",
    "purpose",
    "prompt_tokens",
    "completion_tokens",
    "latency_ms",
    "cost_est",
    "status",
    "attempts",
    "error",
    "created_at",
)


class LLMCallStoreError(Exception):
    """The LLM call log is not available."""


class LLMCallStore:
    """Append-only log of LLM calls with per-run totals."""

    def __init__(self, db: Database):
        self.db = db
        self._lock = threading.RLock()

    def ensure_schema(self) -> None:
        """Raise ``LLMCallStoreError`` if migration 006 has not been applied."""
        row = self.db.fetchone(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'llm_calls'"
        )
        if row is None:
            raise LLMCallStoreError(
                "Table llm_calls is missing; run `stock-analysis migrate` (alembic upgrade head)"
            )

    def available(self) -> bool:
        try:
            self.ensure_schema()
        except LLMCallStoreError:
            return False
        return True

    def record(self, call: LLMCallRecord) -> None:
        """Append one call."""
        values: dict[str, Any] = {c: getattr(call, c) for c in _COLUMNS}
        values["created_at"] = call.created_at.isoformat()
        with self._lock:
            self.db.execute(
                f"INSERT INTO llm_calls ({', '.join(_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in _COLUMNS)})",
                tuple(values[c] for c in _COLUMNS),
            )

    def attach_forecast(self, run_id: str, forecast_id: str) -> None:
        """Link a run's calls to the forecast it produced."""
        with self._lock:
            self.db.execute(
                "UPDATE llm_calls SET forecast_id = ? WHERE run_id = ? AND forecast_id IS NULL",
                (forecast_id, run_id),
            )

    def calls(
        self,
        *,
        run_id: Optional[str] = None,
        ticker: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> list[LLMCallRecord]:
        """Stored calls, oldest first, optionally for one run, ticker or after ``since``."""
        clauses, params = [], []
        for column, value in (("run_id", run_id), ("ticker", ticker)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since.isoformat())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.fetchall(
            f"SELECT {', '.join(_COLUMNS)} FROM llm_calls{where} ORDER BY created_at, id",
            tuple(params),
        )
        return [
            LLMCallRecord(
                **{c: row[c] for c in _COLUMNS if c != "created_at"},
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def summary(self, **filters: Any) -> LLMUsageSummary:
        """Totals over :meth:`calls` with the same filters."""
        return LLMUsageSummary.of(self.calls(**filters))
