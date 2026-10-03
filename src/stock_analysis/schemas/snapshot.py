"""Phase 7 immutable forecast snapshot schemas.

A ``ForecastSnapshot`` is the auditable record of one completed forecast: the
numbers, the evidence they were derived from, the decision-engine results that
opened or closed the adjustment gate, and the model/prompt/data/calibration
versions that produced it. Snapshots are never edited; a correction is a new
snapshot that ``supersedes`` the previous version of the same forecast.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from stock_analysis.schemas.analyst_reports import DecisionType, MarketRegime, RiskCategory
from stock_analysis.schemas.forecast_pipeline import CriticResult, FinalForecast

# Calibration version 0 means "uncalibrated": no stored calibration was applied.
UNCALIBRATED_VERSION = 0


def canonical_json(value: Any) -> str:
    """Serialize ``value`` deterministically (sorted keys, no whitespace)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def compute_data_snapshot_id(data_inputs: dict[str, Any]) -> str:
    """Deterministic hash of the inputs a forecast was generated from (Plan.md §35)."""
    return hashlib.sha256(canonical_json(data_inputs).encode()).hexdigest()


def new_forecast_id() -> str:
    """Generate a new opaque forecast identifier."""
    return uuid4().hex


def _history_sha256(dates: list[str], closes: list[str]) -> str:
    return hashlib.sha256(canonical_json({"dates": dates, "closes": closes}).encode()).hexdigest()


class PriceHistorySnapshot(BaseModel):
    """The exact close series a quant baseline was computed from, content-addressed.

    Dates and closes are kept as the strings they were hashed from (closes are
    ``Decimal`` text), so the series round-trips exactly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    history_sha256: str
    symbol: str
    dates: list[str]
    closes: list[str]

    @classmethod
    def from_points(cls, symbol: str, points: list[Any]) -> PriceHistorySnapshot:
        """Build from price bars with a ``close`` and (optionally) a ``date`` attribute."""
        dates = [str(getattr(p, "date", None)) for p in points]
        closes = [str(p.close) for p in points]
        return cls(
            history_sha256=_history_sha256(dates, closes), symbol=symbol, dates=dates, closes=closes
        )

    @model_validator(mode="after")
    def _validate_hash(self) -> PriceHistorySnapshot:
        if len(self.dates) != len(self.closes):
            raise ValueError("dates and closes must have the same length")
        if _history_sha256(self.dates, self.closes) != self.history_sha256:
            raise ValueError("history_sha256 does not match the stored price series")
        return self

    def close_values(self) -> list[float]:
        """The closes as floats, exactly as the quant node converts them."""
        return [float(Decimal(c)) for c in self.closes]

    def fingerprint(self) -> dict[str, Any]:
        """Compact reference kept in graph state and hashed into data_snapshot_id."""

        def _date(value: Optional[str]) -> Optional[str]:
            return None if value in (None, "None") else value

        return {
            "symbol": self.symbol,
            "bars": len(self.closes),
            "first_date": _date(self.dates[0] if self.dates else None),
            "last_date": _date(self.dates[-1] if self.dates else None),
            "history_sha256": self.history_sha256,
        }


class AppliedCalibration(BaseModel):
    """The calibration parameters a forecast's quant baseline was computed with."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(ge=1)
    vol_multiplier: float = Field(gt=0)
    p50_bias_shift_pct: float


class DecisionAuditRecord(BaseModel):
    """One decision-engine result, with the inputs needed to reconstruct it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_engine: str = Field(description="Graph component that produced the decision")
    decision_model: str
    decision_model_version: str
    decision_type: DecisionType
    decision: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence: dict[str, Any] = Field(
        default_factory=dict, description="Structured inputs the engine evaluated"
    )


class DailyPrediction(BaseModel):
    """Forecast distribution for one trading day inside the horizon."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    day_index: int = Field(ge=1)
    target_date: date
    predicted_return_pct: float
    p10_price: float = Field(gt=0)
    p50_price: float = Field(gt=0)
    p90_price: float = Field(gt=0)
    prob_up: float = Field(ge=0.0, le=1.0)
    # Daily paths come from the quant simulation; LLM adjustments are endpoint-only.
    source: Literal["quant_baseline"] = "quant_baseline"

    @model_validator(mode="after")
    def _validate_price_ordering(self) -> DailyPrediction:
        if not self.p10_price <= self.p50_price <= self.p90_price:
            raise ValueError("daily prediction requires P10 <= P50 <= P90")
        return self


class ForecastSnapshot(BaseModel):
    """Immutable, auditable record of a completed forecast (Plan.md §11, §2.6)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_id: str = Field(min_length=1)
    root_forecast_id: str = Field(min_length=1, description="forecast_id of version 1")
    version: int = Field(default=1, ge=1)
    supersedes_forecast_id: Optional[str] = None
    correction_reason: Optional[str] = None
    corrected_at: Optional[AwareDatetime] = None

    ticker: str
    resolved_symbol: str
    company_name: str
    # When the original forecast was made; corrections keep it (see ``corrected``).
    made_at: AwareDatetime
    as_of_date: date
    target_date: date
    horizon_trading_days: int = Field(ge=1)
    last_close: float = Field(gt=0)

    market_regime: MarketRegime
    risk_category: RiskCategory

    quant_baseline: dict[str, Any]
    final_forecast: FinalForecast
    analyst_reports: dict[str, Optional[dict[str, Any]]]
    decisions: list[DecisionAuditRecord]
    critic_verdict: Optional[CriticResult] = None
    daily_predictions: list[DailyPrediction] = Field(default_factory=list)

    data_snapshot_id: str
    data_inputs: dict[str, Any] = Field(description="Compact inputs hashed into data_snapshot_id")
    data_quality: dict[str, Any] = Field(default_factory=dict)

    model_versions: dict[str, str]
    prompt_versions: dict[str, str]
    calibration_version: int = Field(default=UNCALIBRATED_VERSION, ge=0)
    calibration: Optional[AppliedCalibration] = None
    # Shadow forecast (Plan.md §23): the quant baseline before calibration
    uncalibrated_baseline: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate_lineage(self) -> ForecastSnapshot:
        if self.version == 1:
            if (
                self.supersedes_forecast_id is not None
                or self.correction_reason is not None
                or self.corrected_at is not None
                or self.root_forecast_id != self.forecast_id
            ):
                raise ValueError("version 1 must be its own root and carry no correction fields")
        elif not (self.supersedes_forecast_id and self.correction_reason and self.corrected_at):
            raise ValueError(
                "a corrected version needs supersedes_forecast_id, correction_reason "
                "and corrected_at"
            )
        return self

    @model_validator(mode="after")
    def _validate_calibration(self) -> ForecastSnapshot:
        applied = self.calibration.version if self.calibration else UNCALIBRATED_VERSION
        if applied != self.calibration_version:
            raise ValueError("calibration_version must match the applied calibration")
        if self.calibration is not None and self.uncalibrated_baseline is None:
            raise ValueError("a calibrated forecast must keep its uncalibrated baseline")
        return self

    @property
    def shadow_baseline(self) -> dict[str, Any]:
        """The uncalibrated quant baseline (the quant baseline itself when uncalibrated)."""
        return self.uncalibrated_baseline or self.quant_baseline

    @model_validator(mode="after")
    def _validate_data_snapshot_id(self) -> ForecastSnapshot:
        if compute_data_snapshot_id(self.data_inputs) != self.data_snapshot_id:
            raise ValueError("data_snapshot_id does not match the stored data_inputs")
        return self

    @model_validator(mode="after")
    def _validate_dates(self) -> ForecastSnapshot:
        if self.target_date <= self.as_of_date:
            raise ValueError("target_date must be after as_of_date")
        return self

    @model_validator(mode="after")
    def _validate_daily_path(self) -> ForecastSnapshot:
        if not self.daily_predictions:
            return self
        indices = [p.day_index for p in self.daily_predictions]
        dates = [p.target_date for p in self.daily_predictions]
        if indices != list(range(1, len(indices) + 1)) or len(indices) > self.horizon_trading_days:
            raise ValueError("daily predictions must be days 1..n within the horizon")
        if dates != sorted(set(dates)) or dates[0] <= self.as_of_date:
            raise ValueError("daily prediction dates must be increasing and after as_of_date")
        if len(dates) == self.horizon_trading_days and dates[-1] != self.target_date:
            raise ValueError("the last daily prediction must fall on target_date")
        return self

    def decision(self, decision_type: DecisionType) -> Optional[DecisionAuditRecord]:
        """Return the audit record for ``decision_type``, if one was made."""
        return next((d for d in self.decisions if d.decision_type == decision_type), None)

    def corrected(self, *, reason: str, corrected_at: datetime, **updates: Any) -> ForecastSnapshot:
        """Build the next version of this forecast; the original is left untouched.

        ``made_at`` is kept: a correction does not change when the forecast was
        made, so point-in-time evaluation still sees the original timing.
        """
        forbidden = {
            "forecast_id",
            "root_forecast_id",
            "version",
            "supersedes_forecast_id",
            "correction_reason",
            "corrected_at",
            "made_at",
        }
        if forbidden & updates.keys():
            raise ValueError(
                f"lineage fields cannot be overridden: {sorted(forbidden & updates.keys())}"
            )
        data = self.model_dump()
        data.update(updates)
        data.update(
            forecast_id=new_forecast_id(),
            root_forecast_id=self.root_forecast_id,
            version=self.version + 1,
            supersedes_forecast_id=self.forecast_id,
            correction_reason=reason,
            corrected_at=corrected_at,
        )
        return ForecastSnapshot.model_validate(data)
