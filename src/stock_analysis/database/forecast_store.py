"""Append-only persistence for immutable forecast snapshots.

There is deliberately no update or delete API: a finalized forecast is
corrected by inserting a new version (``record_correction``). The tables also
carry triggers (alembic revision 002) that reject UPDATE/DELETE, so raw SQL
cannot rewrite history either. Every load re-hashes the stored values and
compares them with the hash written at insert time.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Any, Optional

from pydantic import ValidationError

from stock_analysis.database.database import Database
from stock_analysis.logging import get_logger
from stock_analysis.schemas.snapshot import ForecastSnapshot, PriceHistorySnapshot, canonical_json

logger = get_logger(__name__)


class ForecastSnapshotError(Exception):
    """Base error for forecast snapshot persistence."""


class SnapshotExistsError(ForecastSnapshotError):
    """A snapshot with this forecast_id (or version in its lineage) already exists."""


class SnapshotLineageError(ForecastSnapshotError):
    """A correction does not extend the latest version of its forecast."""


class SnapshotIntegrityError(ForecastSnapshotError):
    """Stored snapshot content no longer matches the hash recorded at insert time."""


_SNAPSHOT_COLUMNS = (
    "forecast_id",
    "root_forecast_id",
    "version",
    "supersedes_forecast_id",
    "correction_reason",
    "corrected_at",
    "ticker",
    "resolved_symbol",
    "company_name",
    "made_at",
    "as_of_date",
    "target_date",
    "horizon_trading_days",
    "last_close",
    "prob_up",
    "prob_flat",
    "prob_down",
    "expected_return_pct",
    "p10_price",
    "p50_price",
    "p90_price",
    "adjustment_applied",
    "fallback_to_quant",
    "market_regime",
    "risk_category",
    "data_snapshot_id",
    "calibration_version",
    "model_versions_json",
    "prompt_versions_json",
    "quant_baseline_json",
    "final_forecast_json",
    "analyst_reports_json",
    "critic_json",
    "data_inputs_json",
    "data_quality_json",
)
_DECISION_COLUMNS = (
    "sequence",
    "decision_engine",
    "decision_model",
    "decision_model_version",
    "decision_type",
    "decision",
    "confidence",
    "rationale",
    "evidence_json",
)
_DAILY_COLUMNS = (
    "day_index",
    "target_date",
    "predicted_return_pct",
    "p10_price",
    "p50_price",
    "p90_price",
    "prob_up",
    "source",
)

Rows = list[tuple[Any, ...]]


def _snapshot_values(snapshot: ForecastSnapshot) -> tuple[Any, ...]:
    data = snapshot.model_dump(mode="json")
    final = data["final_forecast"]
    row = {
        **{k: data[k] for k in _SNAPSHOT_COLUMNS if k in data},
        **{k: final[k] for k in ("prob_up", "prob_flat", "prob_down", "expected_return_pct")},
        **{k: final[k] for k in ("p10_price", "p50_price", "p90_price")},
        # SQLite returns booleans as integers; store them that way so the hash round-trips
        "adjustment_applied": int(final["adjustment_applied"]),
        "fallback_to_quant": int(final["fallback_to_quant"]),
        "model_versions_json": canonical_json(data["model_versions"]),
        "prompt_versions_json": canonical_json(data["prompt_versions"]),
        "quant_baseline_json": canonical_json(data["quant_baseline"]),
        "final_forecast_json": canonical_json(final),
        "analyst_reports_json": canonical_json(data["analyst_reports"]),
        "critic_json": canonical_json(data["critic_verdict"]) if data["critic_verdict"] else None,
        "data_inputs_json": canonical_json(data["data_inputs"]),
        "data_quality_json": canonical_json(data["data_quality"]),
    }
    return tuple(row[c] for c in _SNAPSHOT_COLUMNS)


def _decision_values(snapshot: ForecastSnapshot) -> Rows:
    return [
        (
            seq,
            d.decision_engine,
            d.decision_model,
            d.decision_model_version,
            d.decision_type.value,
            d.decision,
            d.confidence,
            d.rationale,
            canonical_json(d.evidence),
        )
        for seq, d in enumerate(snapshot.decisions)
    ]


def _daily_values(snapshot: ForecastSnapshot) -> Rows:
    return [
        (
            p.day_index,
            p.target_date.isoformat(),
            p.predicted_return_pct,
            p.p10_price,
            p.p50_price,
            p.p90_price,
            p.prob_up,
            p.source,
        )
        for p in snapshot.daily_predictions
    ]


_SHADOW_COLUMNS = ("calibration_version", "calibration_json", "uncalibrated_baseline_json")


def _shadow_values(snapshot: ForecastSnapshot) -> Optional[tuple[Any, ...]]:
    if snapshot.uncalibrated_baseline is None:
        return None
    data = snapshot.model_dump(mode="json")
    return (
        snapshot.calibration_version,
        canonical_json(data["calibration"]) if data["calibration"] else None,
        canonical_json(data["uncalibrated_baseline"]),
    )


def _storage_hash(
    row: tuple[Any, ...], decisions: Rows, daily: Rows, shadow: Optional[tuple[Any, ...]] = None
) -> str:
    """SHA-256 over the exact values written to the snapshot tables.

    Hashing stored values (rather than a re-serialised model) keeps old
    snapshots verifiable even if model serialisation changes in a later release.
    The shadow-forecast row is included only when present, so snapshots stored
    before shadow forecasts existed still verify.
    """
    payload = [list(row), [list(d) for d in decisions], [list(p) for p in daily]]
    if shadow is not None:
        payload.append(list(shadow))
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _insert_sql(table: str, columns: tuple[str, ...]) -> str:
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})"


_REQUIRED_TABLES = (
    "forecast_snapshots",
    "forecast_decisions",
    "forecast_daily_predictions",
    "price_history_snapshots",
    "shadow_forecasts",
)


class ForecastSnapshotStore:
    """Insert-only repository for ``ForecastSnapshot`` records.

    A store serialises its own reads and writes, so concurrent graph runs that
    share one store (and its SQLite connection) cannot interleave transactions.
    """

    def __init__(self, db: Database):
        self.db = db
        self._lock = threading.RLock()

    def ensure_schema(self) -> None:
        """Fail fast with a clear message if the Phase 7 migration has not been applied."""
        rows = self.db.fetchall("SELECT name FROM sqlite_master WHERE type = 'table'")
        missing = sorted(set(_REQUIRED_TABLES) - {r["name"] for r in rows})
        if missing:
            raise ForecastSnapshotError(
                f"Forecast snapshot tables are missing ({', '.join(missing)}); "
                "run `stock-analysis migrate` (alembic upgrade head)"
            )

    def save_price_history(self, history: PriceHistorySnapshot) -> None:
        """Store a close series once per content hash (re-saving the same series is a no-op)."""
        fingerprint = history.fingerprint()
        with self._lock, self.db.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO price_history_snapshots (history_sha256, symbol, bars, "
                "first_date, last_date, dates_json, closes_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    history.history_sha256,
                    history.symbol,
                    fingerprint["bars"],
                    fingerprint["first_date"],
                    fingerprint["last_date"],
                    canonical_json(history.dates),
                    canonical_json(history.closes),
                ),
            )

    def get_price_history(self, history_sha256: str) -> Optional[PriceHistorySnapshot]:
        """Load a stored close series, verifying it still matches its hash."""
        with self._lock:
            row = self.db.fetchone(
                "SELECT symbol, dates_json, closes_json FROM price_history_snapshots "
                "WHERE history_sha256 = ?",
                (history_sha256,),
            )
        if row is None:
            return None
        try:
            return PriceHistorySnapshot(
                history_sha256=history_sha256,
                symbol=row["symbol"],
                dates=json.loads(row["dates_json"]),
                closes=json.loads(row["closes_json"]),
            )
        except ValidationError as err:
            raise SnapshotIntegrityError(
                f"Price history {history_sha256} does not match its hash"
            ) from err

    def save(self, snapshot: ForecastSnapshot) -> None:
        """Persist a snapshot with its decision audit and daily predictions atomically.

        Raises:
            SnapshotExistsError: the forecast_id, or its lineage version, is taken.
            SnapshotLineageError: a correction that does not extend the latest version.
        """
        with self._lock:
            self._save(snapshot)

    def _save(self, snapshot: ForecastSnapshot) -> None:
        if snapshot.version > 1:
            self._check_extends_latest(snapshot)

        row = _snapshot_values(snapshot)
        decisions = _decision_values(snapshot)
        daily = _daily_values(snapshot)
        shadow = _shadow_values(snapshot)
        digest = _storage_hash(row, decisions, daily, shadow)
        fid = snapshot.forecast_id
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    _insert_sql("forecast_snapshots", (*_SNAPSHOT_COLUMNS, "snapshot_hash")),
                    (*row, digest),
                )
                conn.executemany(
                    _insert_sql("forecast_decisions", ("forecast_id", *_DECISION_COLUMNS)),
                    [(fid, *d) for d in decisions],
                )
                conn.executemany(
                    _insert_sql("forecast_daily_predictions", ("forecast_id", *_DAILY_COLUMNS)),
                    [(fid, *p) for p in daily],
                )
                if shadow is not None:
                    conn.execute(
                        _insert_sql("shadow_forecasts", ("forecast_id", *_SHADOW_COLUMNS)),
                        (fid, *shadow),
                    )
        except sqlite3.IntegrityError as err:
            if "UNIQUE constraint failed: forecast_snapshots." in str(err):
                raise SnapshotExistsError(
                    f"Forecast {fid} (root {snapshot.root_forecast_id}, "
                    f"version {snapshot.version}) already exists"
                ) from err
            raise ForecastSnapshotError(f"Failed to persist forecast {fid}: {err}") from err

        logger.info(
            "forecast_snapshot_saved",
            forecast_id=snapshot.forecast_id,
            ticker=snapshot.ticker,
            version=snapshot.version,
            data_snapshot_id=snapshot.data_snapshot_id,
        )

    def get(self, forecast_id: str) -> Optional[ForecastSnapshot]:
        """Load one snapshot version, verifying its integrity hash."""
        with self._lock:
            row = self.db.fetchone(
                f"SELECT {', '.join(_SNAPSHOT_COLUMNS)}, snapshot_hash FROM forecast_snapshots "
                "WHERE forecast_id = ?",
                (forecast_id,),
            )
            return self._from_row(row) if row else None

    def get_latest(self, forecast_id: str) -> Optional[ForecastSnapshot]:
        """Return the newest version in the lineage that ``forecast_id`` belongs to."""
        root = self._root_of(forecast_id)
        if root is None:
            return None
        row = self.db.fetchone(
            "SELECT forecast_id FROM forecast_snapshots WHERE root_forecast_id = ? "
            "ORDER BY version DESC LIMIT 1",
            (root,),
        )
        return self.get(row["forecast_id"]) if row else None

    def list_versions(self, forecast_id: str) -> list[ForecastSnapshot]:
        """Return every version in the lineage, oldest first."""
        root = self._root_of(forecast_id)
        if root is None:
            return []
        rows = self.db.fetchall(
            "SELECT forecast_id FROM forecast_snapshots WHERE root_forecast_id = ? ORDER BY version",
            (root,),
        )
        return [s for r in rows if (s := self.get(r["forecast_id"])) is not None]

    def list_for_ticker(
        self, ticker: str, *, include_corrections: bool = False
    ) -> list[ForecastSnapshot]:
        """Return a ticker's forecasts, newest first.

        By default only original (version 1) forecasts are returned: evaluation
        must score what was actually forecast. Use ``get_latest`` for the
        corrected view, or ``include_corrections=True`` for every version.
        """
        version_filter = "" if include_corrections else " AND version = 1"
        rows = self.db.fetchall(
            f"SELECT forecast_id FROM forecast_snapshots WHERE ticker = ?{version_filter} "
            "ORDER BY made_at DESC, version DESC",
            (ticker,),
        )
        return [s for r in rows if (s := self.get(r["forecast_id"])) is not None]

    def record_correction(
        self,
        forecast_id: str,
        *,
        reason: str,
        corrected_at: Optional[datetime] = None,
        **updates: Any,
    ) -> ForecastSnapshot:
        """Persist a corrected version of the latest snapshot in ``forecast_id``'s lineage.

        The original rows are untouched; the new version links back via
        ``supersedes_forecast_id`` and keeps the original ``made_at``.
        """
        with self._lock:
            latest = self.get_latest(forecast_id)
            if latest is None:
                raise SnapshotLineageError(f"Unknown forecast {forecast_id}")
            corrected = latest.corrected(
                reason=reason, corrected_at=corrected_at or datetime.now(UTC), **updates
            )
            self.save(corrected)
            return corrected

    def _root_of(self, forecast_id: str) -> Optional[str]:
        row = self.db.fetchone(
            "SELECT root_forecast_id FROM forecast_snapshots WHERE forecast_id = ?", (forecast_id,)
        )
        return row["root_forecast_id"] if row else None

    def _check_extends_latest(self, snapshot: ForecastSnapshot) -> None:
        latest = self.db.fetchone(
            "SELECT forecast_id, version FROM forecast_snapshots WHERE root_forecast_id = ? "
            "ORDER BY version DESC LIMIT 1",
            (snapshot.root_forecast_id,),
        )
        if latest is None:
            raise SnapshotLineageError(f"Unknown root forecast {snapshot.root_forecast_id}")
        if (
            snapshot.supersedes_forecast_id != latest["forecast_id"]
            or snapshot.version != latest["version"] + 1
        ):
            raise SnapshotLineageError(
                f"Correction must supersede the latest version {latest['forecast_id']} "
                f"(v{latest['version']}) of {snapshot.root_forecast_id}"
            )

    def _from_row(self, row: sqlite3.Row) -> ForecastSnapshot:
        forecast_id = row["forecast_id"]
        decisions = self.db.fetchall(
            f"SELECT {', '.join(_DECISION_COLUMNS)} FROM forecast_decisions "
            "WHERE forecast_id = ? ORDER BY sequence",
            (forecast_id,),
        )
        daily = self.db.fetchall(
            f"SELECT {', '.join(_DAILY_COLUMNS)} FROM forecast_daily_predictions "
            "WHERE forecast_id = ? ORDER BY day_index",
            (forecast_id,),
        )
        shadow = self.db.fetchone(
            f"SELECT {', '.join(_SHADOW_COLUMNS)} FROM shadow_forecasts WHERE forecast_id = ?",
            (forecast_id,),
        )
        values = tuple(row[c] for c in _SNAPSHOT_COLUMNS)
        stored = _storage_hash(
            values,
            [tuple(d) for d in decisions],
            [tuple(p) for p in daily],
            tuple(shadow) if shadow is not None else None,
        )
        if stored != row["snapshot_hash"]:
            raise SnapshotIntegrityError(f"Snapshot {forecast_id} does not match its stored hash")

        json_columns = {
            "model_versions": "model_versions_json",
            "prompt_versions": "prompt_versions_json",
            "quant_baseline": "quant_baseline_json",
            "final_forecast": "final_forecast_json",
            "analyst_reports": "analyst_reports_json",
            "critic_verdict": "critic_json",
            "data_inputs": "data_inputs_json",
            "data_quality": "data_quality_json",
        }
        flattened_final = {
            "prob_up",
            "prob_flat",
            "prob_down",
            "expected_return_pct",
            "p10_price",
            "p50_price",
            "p90_price",
            "adjustment_applied",
            "fallback_to_quant",
        }
        scalar = {
            c: row[c]
            for c in _SNAPSHOT_COLUMNS
            if c not in flattened_final and c not in json_columns.values()
        }
        return ForecastSnapshot.model_validate(
            {
                **scalar,
                **{
                    field: json.loads(row[col]) if row[col] is not None else None
                    for field, col in json_columns.items()
                },
                "decisions": [
                    {
                        **{
                            k: d[k]
                            for k in _DECISION_COLUMNS
                            if k not in ("sequence", "evidence_json")
                        },
                        "evidence": json.loads(d["evidence_json"]),
                    }
                    for d in decisions
                ],
                "daily_predictions": [{k: p[k] for k in _DAILY_COLUMNS} for p in daily],
                "calibration": (
                    json.loads(shadow["calibration_json"])
                    if shadow is not None and shadow["calibration_json"]
                    else None
                ),
                "uncalibrated_baseline": (
                    json.loads(shadow["uncalibrated_baseline_json"]) if shadow is not None else None
                ),
            }
        )
