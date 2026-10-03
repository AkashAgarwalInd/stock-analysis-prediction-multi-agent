"""Plan.md Phases 11-12 schemas: postmortems, bounded calibration and decision outcomes.

Learning is controlled adaptation (Plan.md §2.7): a postmortem diagnoses one
scored forecast, lessons need repeated evidence before they are used, and the
only automatic change to the forecasting model is a bounded, versioned
calibration of the quant baseline.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from stock_analysis.schemas.memory import LessonCandidate

MAX_POSTMORTEM_LESSONS = 3


class CauseCategory(str, Enum):
    """Plan.md §18.1 cause enum."""

    WITHIN_EXPECTED_NOISE = "within_expected_noise"
    MARKET_WIDE_MOVE = "market_wide_move"
    EARNINGS_OR_CORPORATE_EVENT = "earnings_or_corporate_event"
    REGULATORY_OR_NEWS_SHOCK = "regulatory_or_news_shock"
    VOLATILITY_UNDERESTIMATED = "volatility_underestimated"
    VOLATILITY_OVERESTIMATED = "volatility_overestimated"
    TREND_MISREAD = "trend_misread"
    ANALYST_ERROR = "analyst_error"
    DATA_ISSUE = "data_issue"


# ---------------------------------------------------------------------------
# Postmortem input: every fact is labelled forecast-time or hindsight
# ---------------------------------------------------------------------------


class FactTiming(str, Enum):
    FORECAST_TIME = "forecast_time"  # recorded in the snapshot when the forecast was made
    HINDSIGHT = "hindsight"  # only known after the forecast was made


class Fact(BaseModel):
    """One numbered fact given to the postmortem (``F<n>`` forecast-time, ``H<n>`` hindsight)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    timing: FactTiming
    source: str
    text: str


class PostmortemFacts(BaseModel):
    """Everything the postmortem may refer to, already split by when it was knowable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_time: list[Fact]
    hindsight: list[Fact]

    def by_id(self) -> dict[str, Fact]:
        return {f.id: f for f in (*self.forecast_time, *self.hindsight)}


class HindsightNewsItem(BaseModel):
    """A news item published during the forecast window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    published_at: AwareDatetime
    headline: str = Field(min_length=1, max_length=300)
    source: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM output (validated, then checked deterministically before it is stored)
# ---------------------------------------------------------------------------


class CitedStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1, max_length=400)
    fact_ids: list[str] = Field(min_length=1, description="IDs of the supplied facts it rests on")


class ProposedLesson(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=20, max_length=500, description="Specific, testable, scoped")
    scope: Literal["ticker", "sector", "general"]
    category: CauseCategory
    evidence_refs: list[str] = Field(
        min_length=1, description="Forecast-time fact IDs (F...) the lesson is based on"
    )

    @field_validator("category")
    @classmethod
    def _not_noise(cls, value: CauseCategory) -> CauseCategory:
        if value == CauseCategory.WITHIN_EXPECTED_NOISE:
            raise ValueError("normal noise does not produce a lesson")
        return value


class PostmortemDiagnosis(BaseModel):
    """What the postmortem LLM returns. It has no metric fields: scores cannot be changed."""

    model_config = ConfigDict(extra="forbid")

    primary_cause: CauseCategory
    explanation: str = Field(min_length=1, max_length=1500)
    knowable_at_forecast_time: list[CitedStatement] = Field(default_factory=list)
    only_in_hindsight: list[CitedStatement] = Field(default_factory=list)
    lessons: list[ProposedLesson] = Field(default_factory=list, max_length=MAX_POSTMORTEM_LESSONS)
    confidence: Literal["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Stored postmortem (Plan.md §37.10)
# ---------------------------------------------------------------------------


class PostmortemMethod(str, Enum):
    DETERMINISTIC = "deterministic"  # expected noise: no LLM call is needed
    LLM = "llm"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"  # LLM failed or was rejected


class PostMortem(BaseModel):
    """Diagnosis of one scored forecast. Metrics stay in ``forecast_outcomes``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_id: str
    ticker: str
    created_at: AwareDatetime
    method: PostmortemMethod
    expected_noise: bool = Field(
        description="Deterministic: actual inside the 80% band, normal vol"
    )
    primary_cause: CauseCategory
    confidence: Literal["low", "medium", "high"]
    explanation: str
    knowable_at_forecast_time: list[CitedStatement] = Field(default_factory=list)
    only_in_hindsight: list[CitedStatement] = Field(default_factory=list)
    lessons: list[LessonCandidate] = Field(default_factory=list, max_length=MAX_POSTMORTEM_LESSONS)
    guard_notes: list[str] = Field(
        default_factory=list, description="What the deterministic checks rejected or moved, and why"
    )
    facts: PostmortemFacts
    prompt_version: Optional[str] = None
    llm_model: Optional[str] = None


# ---------------------------------------------------------------------------
# Calibration (Plan.md §21-22, §37.12)
# ---------------------------------------------------------------------------


class CalibrationParams(BaseModel):
    """One stored calibration version for a ticker; versions are append-only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str
    version: int = Field(ge=1)
    vol_multiplier: float = Field(gt=0)
    p50_bias_shift_pct: float
    previous_vol_multiplier: float = Field(gt=0)
    previous_p50_bias_shift_pct: float
    n_samples: int = Field(ge=0)
    reason: list[str]
    evidence: dict[str, Any] = Field(description="Statistics and the forecast IDs they came from")
    calibrator_version: str
    created_at: AwareDatetime


class CalibrationObservation(BaseModel):
    """One scored forecast, measured against its uncalibrated quant baseline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_id: str
    as_of_date: date
    target_date: date
    residual_log: float = Field(description="ln(actual / uncalibrated P50)")
    predicted_sigma_log: float = Field(gt=0, description="Uncalibrated horizon volatility (log)")


class CalibrationEstimate(BaseModel):
    """The calibrator's result before any decision to store it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vol_multiplier: float
    p50_bias_shift_pct: float
    n_samples: int
    reason: list[str]
    evidence: dict[str, Any]


class CalibrationUpdate(BaseModel):
    """Outcome of one calibration pass for a ticker (stored only when ``changed``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str
    changed: bool
    previous_version: int = Field(ge=0)
    previous_vol_multiplier: float
    previous_p50_bias_shift_pct: float
    estimate: CalibrationEstimate
    stored: Optional[CalibrationParams] = None


# ---------------------------------------------------------------------------
# Decision outcomes (recorded for later evaluation, never acted on automatically)
# ---------------------------------------------------------------------------


class DecisionOutcome(BaseModel):
    """One decision-engine decision of an evaluated forecast, with how the forecast did."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_id: str
    decision_type: str
    ticker: str
    as_of_date: date
    target_date: date
    decision_engine: str
    decision_model: str
    decision_model_version: str
    decision: str
    confidence: float
    adjustment_applied: bool
    fallback_to_quant: bool
    calibration_version: int = Field(ge=0)
    direction_correct: Optional[bool] = None
    in_80pct_band: Optional[bool] = None
    final_loss: Optional[float] = None
    baseline_loss: Optional[float] = None
    uncalibrated_loss: Optional[float] = None
    llm_value_added: Optional[float] = None
    llm_value_added_pinball: Optional[float] = None
    calibration_value_added: Optional[float] = None
    primary_cause: Optional[CauseCategory] = None
    recorded_at: AwareDatetime


class DecisionOutcomeSummary(BaseModel):
    """Aggregate of recorded decision outcomes for one decision type and value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_type: str
    decision: str
    n: int
    mean_llm_value_added: Optional[float]
    mean_final_loss: Optional[float]
    direction_hit_rate: Optional[float]


class BenchmarkComparison(BaseModel):
    """Plan.md §53 benchmarks 2-4 on the same scored forecasts (Brier loss, lower is better)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    n_calibrated: int = Field(description="Forecasts that had a calibration applied")
    mean_uncalibrated_loss: Optional[float]
    mean_calibrated_loss: Optional[float]
    mean_final_loss: Optional[float]
    finding: str
