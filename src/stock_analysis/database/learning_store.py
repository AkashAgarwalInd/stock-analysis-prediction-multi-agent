"""Append-only persistence for learning: calibration versions, postmortems, decision outcomes.

Tables come from alembic revision 005 and reject UPDATE/DELETE. Reads take a
``before``/``as_of`` cutoff so a backtest only sees what existed at the time;
timestamps are compared with SQLite ``julianday`` like the memory store.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Any, Optional

from stock_analysis.database.database import Database
from stock_analysis.logging import get_logger
from stock_analysis.schemas.learning import (
    CalibrationParams,
    DecisionOutcome,
    DecisionOutcomeSummary,
    PostMortem,
)
from stock_analysis.schemas.snapshot import canonical_json

logger = get_logger(__name__)

_REQUIRED_TABLES = (
    "forecast_snapshots",
    "forecast_outcomes",
    "shadow_forecasts",
    "calibration_params",
    "forecast_postmortems",
    "decision_outcomes",
)
_DECISION_OUTCOME_COLUMNS = tuple(DecisionOutcome.model_fields)


class LearningStoreError(Exception):
    """Learning records could not be read or written."""


def _utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise LearningStoreError("learning timestamps must be timezone-aware")
    return dt.astimezone(UTC).isoformat()


def _to_sql(value: Any) -> Any:
    return int(value) if isinstance(value, bool) else value


class LearningStore:
    """Insert-only repository for the learning loop."""

    def __init__(self, db: Database):
        self.db = db
        self._lock = threading.RLock()

    def ensure_schema(self) -> None:
        """Fail fast with a clear message if the learning migration has not been applied."""
        rows = self.db.fetchall("SELECT name FROM sqlite_master WHERE type = 'table'")
        missing = sorted(set(_REQUIRED_TABLES) - {r["name"] for r in rows})
        if missing:
            raise LearningStoreError(
                f"Learning tables are missing ({', '.join(missing)}); "
                "run `stock-analysis migrate` (alembic upgrade head)"
            )

    # -- calibration ------------------------------------------------------

    def save_calibration(self, params: CalibrationParams) -> None:
        """Store the next calibration version for a ticker (must be latest + 1)."""
        with self._lock:
            latest = self.latest_calibration(params.ticker)
            expected = (latest.version if latest else 0) + 1
            if params.version != expected:
                raise LearningStoreError(
                    f"Calibration for {params.ticker} must be version {expected}, "
                    f"got {params.version}"
                )
            try:
                self.db.execute(
                    "INSERT INTO calibration_params (ticker, version, vol_multiplier, "
                    "p50_bias_shift_pct, previous_vol_multiplier, previous_p50_bias_shift_pct, "
                    "n_samples, reason_json, evidence_json, calibrator_version, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        params.ticker,
                        params.version,
                        params.vol_multiplier,
                        params.p50_bias_shift_pct,
                        params.previous_vol_multiplier,
                        params.previous_p50_bias_shift_pct,
                        params.n_samples,
                        canonical_json(params.reason),
                        canonical_json(params.evidence),
                        params.calibrator_version,
                        _utc_iso(params.created_at),
                    ),
                )
            except sqlite3.IntegrityError as err:
                raise LearningStoreError(f"Cannot store calibration: {err}") from err
        logger.info(
            "calibration_saved",
            ticker=params.ticker,
            version=params.version,
            vol_multiplier=params.vol_multiplier,
            p50_bias_shift_pct=params.p50_bias_shift_pct,
        )

    def latest_calibration(
        self, ticker: str, *, as_of: Optional[datetime] = None
    ) -> Optional[CalibrationParams]:
        """The newest calibration for ``ticker`` created at or before ``as_of``."""
        cutoff = _utc_iso(as_of or datetime.now(UTC))
        row = self.db.fetchone(
            "SELECT * FROM calibration_params WHERE ticker = ? "
            "AND julianday(created_at) <= julianday(?) ORDER BY version DESC LIMIT 1",
            (ticker, cutoff),
        )
        return self._calibration(row) if row else None

    def calibration_history(self, ticker: str) -> list[CalibrationParams]:
        """Every calibration version for ``ticker``, oldest first."""
        rows = self.db.fetchall(
            "SELECT * FROM calibration_params WHERE ticker = ? ORDER BY version", (ticker,)
        )
        return [self._calibration(r) for r in rows]

    @staticmethod
    def _calibration(row: sqlite3.Row) -> CalibrationParams:
        data = dict(row)
        data["reason"] = json.loads(data.pop("reason_json"))
        data["evidence"] = json.loads(data.pop("evidence_json"))
        return CalibrationParams.model_validate(data)

    # -- postmortems ------------------------------------------------------

    def save_postmortem(self, postmortem: PostMortem) -> None:
        """Store the postmortem of an evaluated forecast (one per forecast)."""
        data = postmortem.model_dump(mode="json")
        try:
            with self._lock:
                self.db.execute(
                    "INSERT INTO forecast_postmortems (forecast_id, ticker, method, primary_cause, "
                    "confidence, expected_noise, knowable_at_forecast_time_json, hindsight_json, "
                    "lessons_json, postmortem_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        postmortem.forecast_id,
                        postmortem.ticker,
                        postmortem.method.value,
                        postmortem.primary_cause.value,
                        postmortem.confidence,
                        int(postmortem.expected_noise),
                        canonical_json(data["knowable_at_forecast_time"]),
                        canonical_json(data["only_in_hindsight"]),
                        canonical_json(data["lessons"]),
                        canonical_json(data),
                        _utc_iso(postmortem.created_at),
                    ),
                )
        except sqlite3.IntegrityError as err:
            raise LearningStoreError(
                f"Cannot store postmortem for forecast {postmortem.forecast_id}: {err}"
            ) from err
        logger.info(
            "postmortem_saved",
            forecast_id=postmortem.forecast_id,
            cause=postmortem.primary_cause.value,
            method=postmortem.method.value,
            lessons=len(postmortem.lessons),
        )

    def get_postmortem(self, forecast_id: str) -> Optional[PostMortem]:
        row = self.db.fetchone(
            "SELECT postmortem_json FROM forecast_postmortems WHERE forecast_id = ?", (forecast_id,)
        )
        return PostMortem.model_validate(json.loads(row["postmortem_json"])) if row else None

    def pending_postmortems(self, *, before: datetime, ticker: Optional[str] = None) -> list[str]:
        """Scored original forecasts (evaluated by ``before``) that have no postmortem yet."""
        ticker_filter = " AND s.ticker = ?" if ticker else ""
        params: tuple[Any, ...] = (_utc_iso(before), *((ticker,) if ticker else ()))
        rows = self.db.fetchall(
            "SELECT o.forecast_id FROM forecast_outcomes o "
            "JOIN forecast_snapshots s ON s.forecast_id = o.forecast_id "
            "LEFT JOIN forecast_postmortems p ON p.forecast_id = o.forecast_id "
            "WHERE o.status = 'scored' AND p.forecast_id IS NULL AND s.version = 1 "
            f"AND julianday(o.evaluated_at) <= julianday(?){ticker_filter} "
            "ORDER BY o.target_date, s.made_at",
            params,
        )
        return [r["forecast_id"] for r in rows]

    def scored_forecast_ids(self, ticker: str, *, before: datetime, limit: int) -> list[str]:
        """The most recent scored original forecasts for ``ticker`` evaluated by ``before``."""
        rows = self.db.fetchall(
            "SELECT o.forecast_id FROM forecast_outcomes o "
            "JOIN forecast_snapshots s ON s.forecast_id = o.forecast_id "
            "WHERE s.ticker = ? AND s.version = 1 AND o.status = 'scored' "
            "AND julianday(o.evaluated_at) <= julianday(?) "
            "ORDER BY o.target_date DESC, s.made_at DESC LIMIT ?",
            (ticker, _utc_iso(before), limit),
        )
        return [r["forecast_id"] for r in rows]

    # -- decision outcomes ------------------------------------------------

    def record_decision_outcomes(self, outcomes: list[DecisionOutcome]) -> int:
        """Store decision outcomes; ones already recorded for a forecast are left as they are."""
        rows = [
            tuple(_to_sql(o.model_dump(mode="json")[c]) for c in _DECISION_OUTCOME_COLUMNS)
            for o in outcomes
        ]
        with self._lock, self.db.transaction() as conn:
            cursor = conn.executemany(
                f"INSERT OR IGNORE INTO decision_outcomes ({', '.join(_DECISION_OUTCOME_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in _DECISION_OUTCOME_COLUMNS)})",
                rows,
            )
            return cursor.rowcount

    def decision_outcomes(self, forecast_id: str) -> list[DecisionOutcome]:
        rows = self.db.fetchall(
            f"SELECT {', '.join(_DECISION_OUTCOME_COLUMNS)} FROM decision_outcomes "
            "WHERE forecast_id = ? ORDER BY decision_type",
            (forecast_id,),
        )
        return [DecisionOutcome.model_validate(dict(r)) for r in rows]

    def decision_outcome_summary(
        self, *, before: Optional[datetime] = None, ticker: Optional[str] = None
    ) -> list[DecisionOutcomeSummary]:
        """Per decision type and value: count, mean LLM value added, loss and hit rate.

        For evaluation only; nothing in the system changes behaviour based on it.
        """
        ticker_filter = " AND ticker = ?" if ticker else ""
        params: tuple[Any, ...] = (
            _utc_iso(before or datetime.now(UTC)),
            *((ticker,) if ticker else ()),
        )
        rows = self.db.fetchall(
            "SELECT decision_type, decision, COUNT(*) AS n, "
            "AVG(llm_value_added) AS mean_llm_value_added, AVG(final_loss) AS mean_final_loss, "
            "AVG(direction_correct) AS direction_hit_rate FROM decision_outcomes "
            f"WHERE julianday(recorded_at) <= julianday(?){ticker_filter} "
            "GROUP BY decision_type, decision ORDER BY decision_type, decision",
            params,
        )
        return [DecisionOutcomeSummary.model_validate(dict(r)) for r in rows]
