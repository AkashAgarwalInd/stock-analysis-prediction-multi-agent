"""Append-only persistence for forecast outcomes (Plan.md §37.7-37.8).

Only final results (scored or invalid) are stored; outcomes are never
rewritten (alembic revision 004 triggers reject UPDATE/DELETE).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date
from typing import Any, Optional

from stock_analysis.database.database import Database
from stock_analysis.logging import get_logger
from stock_analysis.schemas.outcome import ForecastOutcome, OutcomeDaily, OutcomeStatus
from stock_analysis.schemas.snapshot import canonical_json

logger = get_logger(__name__)

_REQUIRED_TABLES = ("forecast_outcomes", "outcome_daily")
_JSON_FIELDS = frozenset(
    {"validity_checks", "corporate_actions", "pinball", "baseline_pinball", "analyst_hits"}
)
_COLUMNS = tuple(name for name in ForecastOutcome.model_fields if name != "daily")
_DAILY_COLUMNS = tuple(OutcomeDaily.model_fields)


class OutcomeStoreError(Exception):
    """An outcome could not be stored or read."""


def _to_sql(name: str, value: Any) -> Any:
    if name in _JSON_FIELDS:
        return None if value is None else canonical_json(value)
    if isinstance(value, bool):
        return int(value)
    return value


class OutcomeStore:
    """Insert-only repository for ``ForecastOutcome`` records."""

    def __init__(self, db: Database):
        self.db = db
        self._lock = threading.RLock()

    def ensure_schema(self) -> None:
        """Fail fast with a clear message if the outcome migration has not been applied."""
        rows = self.db.fetchall("SELECT name FROM sqlite_master WHERE type = 'table'")
        missing = sorted(set(_REQUIRED_TABLES) - {r["name"] for r in rows})
        if missing:
            raise OutcomeStoreError(
                f"Outcome tables are missing ({', '.join(missing)}); "
                "run `stock-analysis migrate` (alembic upgrade head)"
            )

    def save(self, outcome: ForecastOutcome) -> None:
        """Store a scored or invalid outcome with its daily rows, atomically."""
        if outcome.status == OutcomeStatus.UNRESOLVED:
            raise OutcomeStoreError("Unresolved outcomes are not stored; retry the review later")
        data = outcome.model_dump(mode="json")
        try:
            with self._lock, self.db.transaction() as conn:
                conn.execute(
                    f"INSERT INTO forecast_outcomes ({', '.join(_COLUMNS)}) "
                    f"VALUES ({', '.join('?' for _ in _COLUMNS)})",
                    tuple(_to_sql(c, data[c]) for c in _COLUMNS),
                )
                conn.executemany(
                    f"INSERT INTO outcome_daily (forecast_id, {', '.join(_DAILY_COLUMNS)}) "
                    f"VALUES (?, {', '.join('?' for _ in _DAILY_COLUMNS)})",
                    [
                        (outcome.forecast_id, *(_to_sql(c, d[c]) for c in _DAILY_COLUMNS))
                        for d in data["daily"]
                    ],
                )
        except sqlite3.IntegrityError as err:
            raise OutcomeStoreError(
                f"Cannot store outcome for forecast {outcome.forecast_id}: {err}"
            ) from err
        logger.info(
            "forecast_outcome_saved", forecast_id=outcome.forecast_id, status=outcome.status.value
        )

    def get(self, forecast_id: str) -> Optional[ForecastOutcome]:
        """Load a stored outcome with its daily rows."""
        with self._lock:
            row = self.db.fetchone(
                f"SELECT {', '.join(_COLUMNS)} FROM forecast_outcomes WHERE forecast_id = ?",
                (forecast_id,),
            )
            if row is None:
                return None
            daily = self.db.fetchall(
                f"SELECT {', '.join(_DAILY_COLUMNS)} FROM outcome_daily "
                "WHERE forecast_id = ? ORDER BY date",
                (forecast_id,),
            )
        data = {
            c: (json.loads(row[c]) if c in _JSON_FIELDS and row[c] is not None else row[c])
            for c in _COLUMNS
        }
        data["daily"] = [{c: d[c] for c in _DAILY_COLUMNS} for d in daily]
        return ForecastOutcome.model_validate(data)

    def matured_unevaluated(self, completed: date, *, ticker: Optional[str] = None) -> list[str]:
        """Original forecasts whose target session is on or before ``completed`` and that
        have no stored outcome, oldest target first.

        Evaluation always refers to the original (version 1) forecast, never a correction.
        """
        ticker_filter = " AND s.ticker = ?" if ticker else ""
        params: tuple[Any, ...] = (
            (completed.isoformat(), ticker) if ticker else (completed.isoformat(),)
        )
        rows = self.db.fetchall(
            "SELECT s.forecast_id FROM forecast_snapshots s "
            "LEFT JOIN forecast_outcomes o ON o.forecast_id = s.forecast_id "
            f"WHERE o.forecast_id IS NULL AND s.version = 1 AND s.target_date <= ?{ticker_filter} "
            "ORDER BY s.target_date, s.made_at",
            params,
        )
        return [r["forecast_id"] for r in rows]

    def latest_for_ticker(self, ticker: str) -> Optional[ForecastOutcome]:
        """The most recent stored outcome (by target date) for an original forecast of ``ticker``."""
        row = self.db.fetchone(
            "SELECT o.forecast_id FROM forecast_outcomes o "
            "JOIN forecast_snapshots s ON s.forecast_id = o.forecast_id "
            "WHERE s.ticker = ? ORDER BY o.target_date DESC, s.made_at DESC LIMIT 1",
            (ticker,),
        )
        return self.get(row["forecast_id"]) if row else None
