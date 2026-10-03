"""Plan.md Phase 9 outcome schemas: what was predicted vs what actually happened."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Direction = Literal["up", "flat", "down"]


class OutcomeStatus(str, Enum):
    SCORED = "scored"
    UNRESOLVED = "unresolved"  # not stored: retried on the next review
    INVALID = "invalid"


class OutcomeDaily(BaseModel):
    """One trading day of the evaluated window (Plan.md §37.8)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    date: date
    close: float = Field(gt=0, description="Close as returned by the data source")
    close_on_forecast_basis: float = Field(gt=0)
    nifty_close: Optional[float] = None
    in_daily_band: Optional[bool] = None


class ForecastOutcome(BaseModel):
    """Deterministic evaluation of one forecast against actual prices (Plan.md §14-16).

    Price-level comparisons use ``actual_close_on_forecast_basis``: the actual
    return over the window applied to the forecast's own ``last_close``, so a
    dividend or split adjustment made to the price history after the forecast
    cannot masquerade as forecast error.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_id: str
    status: OutcomeStatus
    invalid_reason: Optional[str] = None
    evaluated_at: AwareDatetime
    scorer_version: str

    # Window and data-quality audit
    as_of_date: date
    target_date: date
    validity_checks: list[str] = Field(default_factory=list)
    corporate_actions: list[dict[str, Any]] = Field(default_factory=list)
    adjustment_factor: Optional[float] = None

    # Actual result
    actual_close: Optional[float] = None
    actual_close_on_forecast_basis: Optional[float] = None
    actual_return_pct: Optional[float] = None
    realized_direction: Optional[Direction] = None

    # Final forecast accuracy
    predicted_direction: Optional[Direction] = None
    direction_correct: Optional[bool] = None
    signed_error_pct: Optional[float] = None
    abs_error_pct: Optional[float] = None
    in_80pct_band: Optional[bool] = None
    daily_band_breaches: Optional[int] = None
    pit_percentile: Optional[float] = None
    brier: Optional[float] = None
    pinball: Optional[dict[str, float]] = None
    realized_vol_pct: Optional[float] = None
    vol_ratio: Optional[float] = None

    # Benchmark
    nifty_return_pct: Optional[float] = None
    excess_vs_nifty_pct: Optional[float] = None
    beta_adjusted_excess_pct: Optional[float] = None

    # Quant baseline vs final (Plan.md §16) on the same outcome
    loss_metric: str = "brier"
    baseline_signed_error_pct: Optional[float] = None
    baseline_brier: Optional[float] = None
    baseline_pinball: Optional[dict[str, float]] = None
    baseline_loss: Optional[float] = None
    final_loss: Optional[float] = None
    llm_value_added: Optional[float] = None
    llm_value_added_pinball: Optional[float] = None

    analyst_hits: Optional[dict[str, bool]] = None

    # Decision-engine attribution, copied from the forecast snapshot
    decision_engine: Optional[str] = None
    decision_engine_enabled: bool = False
    market_regime_decision: Optional[str] = None
    adjustment_gate_decision: Optional[str] = None
    adjustment_applied: bool = False
    fallback_to_quant: bool = False

    daily: list[OutcomeDaily] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_status(self) -> ForecastOutcome:
        if self.status == OutcomeStatus.SCORED:
            required = (self.actual_close, self.actual_return_pct, self.final_loss)
            if any(v is None for v in required):
                raise ValueError("a scored outcome needs actual close, return and losses")
            if self.invalid_reason:
                raise ValueError("a scored outcome cannot carry an invalid_reason")
        elif not self.invalid_reason:
            raise ValueError(f"a {self.status.value} outcome needs a reason")
        return self
