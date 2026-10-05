"""Phase 7 long-term memory schemas: insights, lessons and the track record.

Memory is controlled, auditable context (Plan.md §2.7), not "LLM memory":
insights are deterministic summaries of stored forecast snapshots, lessons
follow an evidence-gated lifecycle, and the track record only reports what
has actually been measured.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from stock_analysis.schemas.scorecard import GroupScore, HitRate
from stock_analysis.schemas.snapshot import ForecastSnapshot, ForecastSource

# Plan.md §20: discrete event lessons (e.g. earnings) activate after fewer confirmations.
EVENT_LESSON_CATEGORIES = frozenset({"earnings", "earnings_or_corporate_event", "corporate_event"})

_MAX_INSIGHT_DRIVERS = 3
_MAX_INSIGHT_RISKS = 2


class ForecastInsight(BaseModel):
    """Distilled, prompt-sized summary of one stored forecast (Plan.md node 17)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_id: str
    ticker: str
    made_at: AwareDatetime
    as_of_date: date
    target_date: date
    market_regime: str
    risk_category: str
    last_close: float
    prob_up: float
    prob_flat: float
    prob_down: float
    p10_price: float
    p50_price: float
    p90_price: float
    adjustment_applied: bool
    fallback_to_quant: bool
    key_drivers: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)

    @classmethod
    def from_snapshot(cls, snapshot: ForecastSnapshot) -> ForecastInsight:
        """Copy (never recompute) the headline facts of a stored forecast."""
        final = snapshot.final_forecast
        return cls(
            forecast_id=snapshot.forecast_id,
            ticker=snapshot.ticker,
            made_at=snapshot.made_at,
            as_of_date=snapshot.as_of_date,
            target_date=snapshot.target_date,
            market_regime=snapshot.market_regime.value,
            risk_category=snapshot.risk_category.value,
            last_close=snapshot.last_close,
            prob_up=final.prob_up,
            prob_flat=final.prob_flat,
            prob_down=final.prob_down,
            p10_price=final.p10_price,
            p50_price=final.p50_price,
            p90_price=final.p90_price,
            adjustment_applied=final.adjustment_applied,
            fallback_to_quant=final.fallback_to_quant,
            key_drivers=final.evidence[:_MAX_INSIGHT_DRIVERS],
            key_risks=final.risks[:_MAX_INSIGHT_RISKS],
        )


class LessonCandidate(BaseModel):
    """A proposed lesson (Plan.md §19, §40 ``LessonCandidate``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=20, max_length=500, description="Specific, testable, scoped")
    scope: Literal["ticker", "sector", "general"]
    category: str = Field(min_length=1, max_length=60)
    ticker: Optional[str] = None
    sector: Optional[str] = None

    @model_validator(mode="after")
    def _validate_scope(self) -> LessonCandidate:
        if self.scope == "ticker" and not self.ticker:
            raise ValueError("a ticker-scoped lesson needs a ticker")
        if self.scope == "sector" and not self.sector:
            raise ValueError("a sector-scoped lesson needs a sector")
        return self


class LessonStatus(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    RETIRED = "retired"


class LessonEvidenceKind(str, Enum):
    CONFIRMED = "confirmed"
    CONTRADICTED = "contradicted"
    RETIRED = "retired"


class Lesson(BaseModel):
    """A lesson with its lifecycle state derived from the append-only evidence log."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lesson_id: str
    text: str
    scope: Literal["ticker", "sector", "general"]
    category: str
    ticker: Optional[str] = None
    sector: Optional[str] = None
    status: LessonStatus
    evidence_count: int = Field(ge=0)
    contradicted_count: int = Field(ge=0)
    source_forecast_ids: list[str]
    first_seen: AwareDatetime
    last_confirmed: Optional[AwareDatetime] = None


def lesson_status(
    *,
    category: str,
    evidence_count: int,
    contradicted_count: int,
    retired: bool,
    min_evidence: int,
    event_min_evidence: int,
) -> LessonStatus:
    """Plan.md §20 lifecycle: active only with enough uncontradicted confirmations."""
    if retired or (contradicted_count > 0 and contradicted_count >= evidence_count):
        return LessonStatus.RETIRED
    needed = event_min_evidence if category in EVENT_LESSON_CATEGORIES else min_evidence
    return LessonStatus.ACTIVE if evidence_count >= needed else LessonStatus.CANDIDATE


class VariantMetrics(BaseModel):
    """One benchmark's record over a track-record window (Plan.md §31/§53)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    variant: str = Field(description="naive_flat, quant_uncalibrated, quant_calibrated or final")
    label: str
    direction: HitRate
    coverage_80pct: Optional[float] = Field(default=None, description="None: no band (naive)")
    brier: Optional[float] = Field(default=None, description="None: no probabilities (naive)")
    mean_abs_error_pct: Optional[float] = None


class WindowMetrics(BaseModel):
    """The final forecast's record over one rolling track-record window (Plan.md §28),
    with every benchmark's record over the same forecasts in ``variants``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    n: int = Field(ge=0)
    direction: HitRate
    coverage_80pct: Optional[float] = None
    brier: Optional[float] = None
    mean_abs_error_pct: Optional[float] = None
    mean_signed_error_pct: Optional[float] = None
    n_adjusted: int = Field(default=0, ge=0, description="Forecasts the LLM adjusted")
    llm_value_added: Optional[float] = Field(
        default=None, description="Mean calibrated-quant Brier - final Brier"
    )
    finding: str
    variants: list[VariantMetrics] = Field(default_factory=list)


class OutcomeMetrics(BaseModel):
    """Rolling accuracy of the independent scored forecasts, copied from the track record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Optional[ForecastSource] = Field(
        default=None, description="None: stored before records were kept per source (pooled)"
    )
    n_forecasts: int = Field(ge=0, description="Independent scored forecasts")
    n_inferred_source: int = Field(default=0, ge=0, description="Of which the source was inferred")
    min_samples: int = Field(ge=1, description="Below this no variant is called better")
    windows: list[WindowMetrics] = Field(default_factory=list)

    def distinct_windows(self) -> list[tuple[str, WindowMetrics]]:
        """Non-empty windows, merging a longer window into a shorter one with the same
        forecasts (windows are nested, so an equal count means the same forecasts)."""
        merged: list[tuple[list[str], WindowMetrics]] = []
        for w in self.windows:
            if not w.n:
                continue
            if merged and merged[-1][1].n == w.n:
                merged[-1][0].append(w.label)
            else:
                merged.append(([w.label], w))
        return [(" = ".join(labels), w) for labels, w in merged]


class TrackRecord(BaseModel):
    """What is known about this ticker's past forecasts.

    Counts come from the stored records; ``outcome_metrics_by_source`` (Plan.md
    §28) is copied from the point-in-time track records, one per forecast source
    (live first), and is empty when it was not loaded (no outcome store) or
    nothing has been scored yet. ``outcome_metrics`` is the single pooled record
    of contexts stored before sources were kept apart; new runs leave it ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str
    forecasts_made: int = Field(ge=0)
    forecasts_scored: int = Field(default=0, ge=0)
    forecasts_awaiting_outcome: int = Field(
        ge=0, description="Target date passed, not yet evaluated"
    )
    outcome_metrics: Optional[OutcomeMetrics] = None
    outcome_metrics_by_source: list[OutcomeMetrics] = Field(default_factory=list)

    def metric_groups(self) -> list[OutcomeMetrics]:
        """The per-source records, or the pooled record of an older context."""
        if self.outcome_metrics_by_source:
            return list(self.outcome_metrics_by_source)
        return [self.outcome_metrics] if self.outcome_metrics is not None else []


class LessonNote(BaseModel):
    """A lesson tied to a reviewed forecast, with its status at load time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lesson_id: str
    text: str
    scope: Literal["ticker", "sector", "general"]
    status: LessonStatus
    evidence_count: int = Field(ge=0)


class LastReview(BaseModel):
    """Plan.md §5.2 ``last_review``: the latest evaluated forecast of the ticker vs actual.

    Every number is copied from the stored snapshot and outcome; the cause comes
    from the stored postmortem (``None`` when none existed yet).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_id: str
    as_of_date: date
    target_date: date
    evaluated_at: AwareDatetime
    status: Literal["scored", "invalid"]
    source: Optional[ForecastSource] = Field(
        default=None, description="None: stored before reviews recorded the source"
    )
    source_inferred: bool = False
    invalid_reason: Optional[str] = None
    last_close: float
    prob_up: float
    prob_flat: float
    prob_down: float
    p10_price: float
    p50_price: float
    p90_price: float
    adjustment_applied: bool
    calibration_version: int = Field(ge=0)
    # The quant baseline the forecast started from (calibrated when a version applied)
    baseline_prob_up: Optional[float] = None
    baseline_p10_price: Optional[float] = None
    baseline_p50_price: Optional[float] = None
    baseline_p90_price: Optional[float] = None
    baseline_direction: Optional[str] = None
    baseline_direction_correct: Optional[bool] = None
    baseline_signed_error_pct: Optional[float] = Field(
        default=None, description="Actual vs the quant baseline's P50"
    )
    baseline_in_80pct_band: Optional[bool] = None
    predicted_direction: Optional[str] = None
    realized_direction: Optional[str] = None
    direction_correct: Optional[bool] = None
    actual_close: Optional[float] = Field(default=None, description="On the forecast's price basis")
    actual_return_pct: Optional[float] = None
    signed_error_pct: Optional[float] = Field(default=None, description="Actual vs the final P50")
    in_80pct_band: Optional[bool] = None
    vol_ratio: Optional[float] = None
    nifty_return_pct: Optional[float] = None
    excess_vs_nifty_pct: Optional[float] = None
    baseline_loss: Optional[float] = None
    final_loss: Optional[float] = None
    llm_value_added: Optional[float] = None
    calibration_value_added: Optional[float] = None
    primary_cause: Optional[str] = None
    cause_explanation: Optional[str] = None
    expected_noise: Optional[bool] = None
    lessons: list[LessonNote] = Field(default_factory=list)


class Adaptation(BaseModel):
    """Plan.md §30 system adaptation: the calibration the quant baseline will use.

    ``version`` 0 means no calibration has been stored yet; the ``previous_*``
    values are what that version replaced.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(ge=0)
    vol_multiplier: float = 1.0
    p50_bias_shift_pct: float = 0.0
    previous_vol_multiplier: float = 1.0
    previous_p50_bias_shift_pct: float = 0.0
    created_at: Optional[AwareDatetime] = None
    n_samples: int = Field(default=0, ge=0, description="Forecasts the version was estimated on")
    reason: list[str] = Field(default_factory=list)
    min_samples: int = Field(ge=1, description="CALIBRATION_MIN_SAMPLES")
    applied: bool = Field(description="False when LEARNING_ENABLED is off")
    changed_since_last_forecast: Optional[bool] = Field(
        default=None,
        description="Stored after the previous forecast was made (None: no previous forecast)",
    )


class ScorecardSummary(BaseModel):
    """Plan.md §24-25 for the ticker: analyst hit rates and forecast quality per regime."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_forecasts: int = Field(ge=0)
    # Independent forecasts per source (live / backtest); the summary pools them
    sources: dict[str, int] = Field(default_factory=dict)
    min_samples: int = Field(ge=1, description="MIN_SAMPLES_ANALYST_WEIGHTS")
    analysts: dict[str, HitRate] = Field(default_factory=dict)
    regimes: list[GroupScore] = Field(default_factory=list)


class PreRunReview(BaseModel):
    """What the review run before this forecast did (Plan.md §5.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_at: AwareDatetime
    matured: int = Field(default=0, ge=0)
    scored: int = Field(default=0, ge=0)
    invalid: int = Field(default=0, ge=0)
    unresolved: int = Field(default=0, ge=0)
    postmortems: int = Field(default=0, ge=0)
    lessons_created: int = Field(default=0, ge=0)
    lessons_updated: int = Field(default=0, ge=0)
    calibration_version: Optional[int] = Field(
        default=None, description="New calibration version stored by this review"
    )
    errors: list[str] = Field(default_factory=list)


class MemoryContext(BaseModel):
    """Prior context loaded for a run; stored in the snapshot's data inputs.

    The learning fields (``last_review``, ``adaptation``, ``scorecards`` and
    the track record's ``outcome_metrics``) need the outcome and learning
    stores. ``learning_loaded`` names the parts that were loaded: a part that is
    missing from it was not configured or failed (with a note in ``warnings``),
    so ``None`` there does not mean "nothing to show".
    """

    model_config = ConfigDict(extra="forbid")

    loaded: bool
    prior_forecasts: list[ForecastInsight] = Field(default_factory=list)
    active_lessons: list[Lesson] = Field(default_factory=list)
    track_record: Optional[TrackRecord] = None
    last_review: Optional[LastReview] = None
    adaptation: Optional[Adaptation] = None
    scorecards: Optional[ScorecardSummary] = None
    learning_loaded: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None

    @property
    def has_content(self) -> bool:
        return bool(
            self.prior_forecasts
            or self.active_lessons
            or self.last_review
            or (self.adaptation and self.adaptation.version)
            or (self.scorecards and self.scorecards.n_forecasts)
        )
