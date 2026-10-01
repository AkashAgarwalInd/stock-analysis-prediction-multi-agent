"""Phase 6 Forecast Pipeline Schemas.

Defines strongly typed Pydantic models for the forecasting decision pipeline:
- QuantBaseline (from Phase 4, re-exported)
- ForecastAdjustment - individual bounded adjustment with evidence
- PredictorResult - Gemini predictor output with adjustments
- CriticResult - Critic validation output
- FinalForecast - final consolidated forecast
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class ForecastAdjustmentType(str, Enum):
    """Types of forecast adjustments."""
    PROB_UP = "prob_up"
    PROB_FLAT = "prob_flat"
    PROB_DOWN = "prob_down"
    EXPECTED_RETURN = "expected_return_pct"
    P10_PRICE = "p10_price"
    P50_PRICE = "p50_price"
    P90_PRICE = "p90_price"
    WEEKLY_VOL = "weekly_vol_pct"


class ForecastAdjustment(BaseModel):
    """A single bounded adjustment to the quant baseline.

    Every adjustment must include:
    - what changed (adjustment_type)
    - previous value
    - new value
    - reason (human-readable)
    - evidence references (from analyst reports or quant data)
    """
    adjustment_type: ForecastAdjustmentType
    previous_value: float
    new_value: float
    reason: str = Field(min_length=1, max_length=500)
    evidence_refs: List[str] = Field(default_factory=list, max_items=5)

    @model_validator(mode="after")
    def validate_change_magnitude(self) -> "ForecastAdjustment":
        """Validate adjustment magnitude is reasonable."""
        if self.previous_value == 0:
            return self
        change_pct = abs(self.new_value - self.previous_value) / abs(self.previous_value)
        # Allow up to 50% change for most metrics, 100% for probabilities
        max_change = 1.0 if self.adjustment_type in (
            ForecastAdjustmentType.PROB_UP,
            ForecastAdjustmentType.PROB_FLAT,
            ForecastAdjustmentType.PROB_DOWN,
        ) else 0.5
        if change_pct > max_change:
            raise ValueError(
                f"Adjustment for {self.adjustment_type.value} exceeds maximum allowed change "
                f"({max_change*100:.0f}%): {change_pct*100:.1f}%"
            )
        return self


class PredictorResult(BaseModel):
    """Output from the Gemini Predictor with bounded adjustments.

    Contains the final forecast after applying adjustments to the quant baseline.
    """
    # Final forecast values (after adjustments)
    prob_up: float = Field(ge=0.0, le=1.0)
    prob_flat: float = Field(ge=0.0, le=1.0)
    prob_down: float = Field(ge=0.0, le=1.0)
    expected_return_pct: float
    p10_price: float = Field(gt=0)
    p50_price: float = Field(gt=0)
    p90_price: float = Field(gt=0)
    weekly_vol_pct: float = Field(ge=0)

    # Adjustment metadata
    adjustment_applied: bool
    adjustments: List[ForecastAdjustment] = Field(default_factory=list)
    reasoning: str = Field(min_length=1, max_length=2000)
    evidence: List[str] = Field(default_factory=list, max_items=10)
    risks: List[str] = Field(default_factory=list, max_items=5)

    # Quant baseline reference (for audit trail)
    quant_baseline_ref: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_probabilities_sum(self) -> "PredictorResult":
        """Ensure probabilities sum to 1.0 within tolerance."""
        total = self.prob_up + self.prob_flat + self.prob_down
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Probabilities must sum to 1.0, got {total:.4f} "
                f"(up={self.prob_up:.4f}, flat={self.prob_flat:.4f}, down={self.prob_down:.4f})"
            )
        return self

    @model_validator(mode="after")
    def validate_price_ordering(self) -> "PredictorResult":
        """Ensure P10 < P50 < P90."""
        if not (self.p10_price < self.p50_price < self.p90_price):
            raise ValueError(
                f"Price ordering violated: P10={self.p10_price} < "
                f"P50={self.p50_price} < P90={self.p90_price} required"
            )
        return self

    @model_validator(mode="after")
    def validate_quant_drift(self) -> "PredictorResult":
        """Warn if forecast drifts too far from quant baseline."""
        qb = self.quant_baseline_ref
        if qb:
            # Check probability shifts (MAX_PROB_SHIFT = 0.15)
            max_prob_shift = 0.15
            for field, pred_val in [
                ("prob_up", self.prob_up),
                ("prob_flat", self.prob_flat),
                ("prob_down", self.prob_down),
            ]:
                qb_val = qb.get(field)
                if qb_val is not None:
                    if abs(pred_val - qb_val) > max_prob_shift:
                        raise ValueError(
                            f"Probability shift for {field} exceeds MAX_PROB_SHIFT={max_prob_shift}: "
                            f"quant={qb_val:.4f}, pred={pred_val:.4f}, shift={abs(pred_val - qb_val):.4f}"
                        )
        return self


class CriticCheckType(str, Enum):
    """Types of critic validation checks."""
    PROBABILITY_VALIDITY = "probability_validity"
    PROBABILITY_SUM = "probability_sum"
    PRICE_ORDERING = "price_ordering"
    EXCESSIVE_ADJUSTMENT = "excessive_adjustment"
    UNSUPPORTED_CLAIMS = "unsupported_claims"
    EVIDENCE_LINKAGE = "evidence_linkage"
    OVERCONFIDENCE = "overconfidence"
    QUANT_DISAGREEMENT = "quant_disagreement"
    MISSING_RISKS = "missing_risks"
    DATA_QUALITY = "data_quality"
    REASONING_CONSISTENCY = "reasoning_consistency"


class CriticFinding(BaseModel):
    """A single finding from the critic."""
    check_type: CriticCheckType
    passed: bool
    message: str = Field(min_length=1, max_length=500)
    severity: str = Field(default="error", pattern="^(error|warning)$")
    details: dict = Field(default_factory=dict)


class CriticResult(BaseModel):
    """Structured output from the Critic validation."""
    passed: bool
    findings: List[CriticFinding] = Field(default_factory=list)
    revision_required: bool = False
    revision_guidance: Optional[str] = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def set_revision_flag(self) -> "CriticResult":
        """Set revision_required if any error-level findings exist."""
        self.revision_required = any(f.severity == "error" for f in self.findings)
        return self


class FinalForecast(BaseModel):
    """Final consolidated forecast after predictor + critic + revisions."""
    symbol: str
    forecast_date: str  # ISO format date
    horizon_trading_days: int

    # Final probabilities
    prob_up: float = Field(ge=0.0, le=1.0)
    prob_flat: float = Field(ge=0.0, le=1.0)
    prob_down: float = Field(ge=0.0, le=1.0)

    # Final price targets
    expected_return_pct: float
    p10_price: float = Field(gt=0)
    p50_price: float = Field(gt=0)
    p90_price: float = Field(gt=0)
    weekly_vol_pct: float = Field(ge=0)

    # Pipeline metadata
    adjustment_applied: bool
    adjustments: List[ForecastAdjustment] = Field(default_factory=list)
    quant_baseline: dict = Field(default_factory=dict)
    predictor_reasoning: str = ""
    critic_passed: bool = True
    revision_count: int = 0
    fallback_to_quant: bool = False

    # Evidence and risks
    evidence: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_final_forecast(self) -> "FinalForecast":
        """Validate final forecast constraints."""
        # Probability sum
        total = self.prob_up + self.prob_flat + self.prob_down
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Final probabilities sum to {total:.4f}, must be 1.0")

        # Price ordering
        if not (self.p10_price < self.p50_price < self.p90_price):
            raise ValueError("Final forecast price ordering violated: P10 < P50 < P90 required")

        return self