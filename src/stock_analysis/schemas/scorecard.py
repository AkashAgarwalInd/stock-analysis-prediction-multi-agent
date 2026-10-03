"""Plan.md Phase 13 / §24-25 scorecards: how analysts, regimes and decisions have done.

Scorecards are aggregates over immutable outcome records, computed on demand
with a point-in-time cutoff, never stored as mutable counters. They are for
evaluation only: nothing in the forecasting pipeline reads them.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class ScoredForecast(BaseModel):
    """One scored original forecast with the attributes scorecards group by."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_id: str
    ticker: str
    sector: Optional[str] = None
    market_regime: str
    as_of_date: date
    target_date: date
    evaluated_at: AwareDatetime
    direction_correct: bool
    in_80pct_band: bool
    signed_error_pct: float
    abs_error_pct: float
    brier: float
    analyst_hits: dict[str, bool] = Field(default_factory=dict)


class HitRate(BaseModel):
    """Directional hits with a 95% Wilson interval (honest at small sample sizes)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int = Field(ge=0)
    hits: int = Field(ge=0)
    rate: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None


class AnalystScorecard(BaseModel):
    """Plan.md §24: one analyst's directional record, pooled and broken down."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analyst: str
    pooled: HitRate
    by_ticker: dict[str, HitRate]
    by_sector: dict[str, HitRate]
    by_regime: dict[str, HitRate]
    weighting_eligible: bool
    weighting_note: str


class GroupScore(BaseModel):
    """Plan.md §25 / §37.14: forecast quality for one regime, ticker or sector."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group: str
    n: int
    direction: HitRate
    mean_abs_error_pct: float
    mean_signed_error_pct: float
    brier: float
    coverage_80pct: float


class DecisionEvaluation(BaseModel):
    """Whether forecasts made under one decision-engine decision improved on average."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_type: str
    decision: str
    n: int
    n_adjusted: int = Field(ge=0, description="Forecasts where the LLM adjusted the baseline")
    mean_llm_value_added: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    share_improved: Optional[float] = None
    mean_llm_value_added_pinball: Optional[float] = None
    direction: HitRate
    coverage_80pct: Optional[float] = None
    finding: str


class Scorecards(BaseModel):
    """Everything known as of ``as_of`` (optionally for one ticker)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    as_of: AwareDatetime
    ticker: Optional[str] = None
    n_forecasts: int
    overlapping_excluded: int = Field(
        default=0, description="Forecasts left out because their window overlaps a later one"
    )
    min_samples: int
    analysts: list[AnalystScorecard]
    by_regime: list[GroupScore]
    by_ticker: list[GroupScore]
    by_sector: list[GroupScore]
    decisions: list[DecisionEvaluation]
