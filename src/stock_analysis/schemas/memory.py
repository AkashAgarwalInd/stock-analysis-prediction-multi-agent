"""Phase 7 long-term memory schemas: insights, lessons and the track record.

Memory is controlled, auditable context (Plan.md §2.7), not "LLM memory":
insights are deterministic summaries of stored forecast snapshots, lessons
follow an evidence-gated lifecycle, and the track record only reports what
has actually been measured.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from stock_analysis.schemas.snapshot import ForecastSnapshot

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


class TrackRecord(BaseModel):
    """What is known about this ticker's past forecasts.

    ``outcome_metrics`` stays ``None`` until forecasts are scored against actual
    prices; nothing here is estimated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str
    forecasts_made: int = Field(ge=0)
    forecasts_awaiting_outcome: int = Field(ge=0, description="Target date passed, not scored")
    outcome_metrics: Optional[dict[str, Any]] = None


class MemoryContext(BaseModel):
    """Prior context loaded for a run; stored in the snapshot's data inputs."""

    model_config = ConfigDict(extra="forbid")

    loaded: bool
    prior_forecasts: list[ForecastInsight] = Field(default_factory=list)
    active_lessons: list[Lesson] = Field(default_factory=list)
    track_record: Optional[TrackRecord] = None
    error: Optional[str] = None

    @property
    def has_content(self) -> bool:
        return bool(self.prior_forecasts or self.active_lessons)
