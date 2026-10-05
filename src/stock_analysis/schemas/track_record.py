"""Plan.md Phase 14 / §27-28, §52-55: probability calibration and the forecast track record.

Like the scorecards, these are aggregates over immutable snapshot and outcome
records, computed on demand with a point-in-time cutoff instead of being kept
in mutable tables. The forecast graph shows the final forecast's rolling
metrics to the predictor (Plan.md Phase 16); nothing changes automatically
because of them.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from stock_analysis.schemas.outcome import Direction
from stock_analysis.schemas.scorecard import HitRate
from stock_analysis.schemas.snapshot import ForecastSource

# Plan.md §53 benchmarks, simplest first
Variant = Literal["naive_flat", "quant_uncalibrated", "quant_calibrated", "final"]
VARIANTS: tuple[Variant, ...] = ("naive_flat", "quant_uncalibrated", "quant_calibrated", "final")
VARIANT_LABELS: dict[str, str] = {
    "naive_flat": "Naive flat",
    "quant_uncalibrated": "Quant uncalibrated",
    "quant_calibrated": "Quant calibrated",
    "final": "Final forecast",
}
# Variants that state up/flat/down probabilities (the naive benchmark is a point forecast)
PROBABILISTIC_VARIANTS: tuple[Variant, ...] = ("quant_uncalibrated", "quant_calibrated", "final")

# The binary events a probability is bucketed for; "all" pools the three classes
CalibrationEvent = Literal["up", "flat", "down", "all"]


class VariantScore(BaseModel):
    """One forecast variant scored against one actual outcome.

    The naive flat benchmark predicts no change (P50 = last close, direction
    flat) and states no probabilities or band, so its probability and interval
    metrics are ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prob_up: Optional[float] = None
    prob_flat: Optional[float] = None
    prob_down: Optional[float] = None
    direction_correct: bool
    signed_error_pct: float = Field(description="Actual relative to the P50, in percent")
    in_80pct_band: Optional[bool] = None
    brier: Optional[float] = None
    pinball: Optional[float] = Field(default=None, description="Mean P10/P50/P90 pinball loss")
    vol_ratio: Optional[float] = None
    sharpness_pct: Optional[float] = Field(
        default=None, description="(P90 - P10) / last close, in percent: narrower is sharper"
    )
    pit_percentile: Optional[float] = Field(
        default=None, description="Where the actual close fell in the forecast distribution (0-1)"
    )


class EvaluatedForecast(BaseModel):
    """One scored original forecast with every benchmark variant scored on the same outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_id: str
    ticker: str
    as_of_date: date
    target_date: date
    realized_direction: Direction
    actual_return_pct: float
    llm_value_added: Optional[float] = None
    calibration_value_added: Optional[float] = None
    adjusted: bool = Field(default=False, description="The LLM changed the quant baseline")
    calibrated: bool = Field(default=False, description="A calibration version was applied")
    source: ForecastSource = "live"
    source_inferred: bool = Field(
        default=False, description="Stored before sources were recorded; source inferred"
    )
    variants: dict[str, VariantScore]


class ProbabilityBucket(BaseModel):
    """Plan.md §27: forecasts whose predicted probability fell in ``[low, high)``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    low: float
    high: float
    n: int = Field(ge=1)
    mean_predicted: float
    observed: HitRate = Field(description="How often the event happened, with a 95% Wilson CI")

    @property
    def gap(self) -> float:
        """Observed frequency minus mean predicted probability (negative: overconfident)."""
        assert self.observed.rate is not None
        return self.observed.rate - self.mean_predicted


class ReliabilityTable(BaseModel):
    """Predicted probability vs observed frequency for one binary event.

    ``brier`` is the binary Brier score; the Murphy decomposition
    ``brier ≈ reliability - resolution + uncertainty`` is exact when every
    forecast in a bucket has the same probability. ``reliability`` and ``ece``
    (expected calibration error) are 0 for perfectly calibrated forecasts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event: CalibrationEvent
    n: int
    base_rate: Optional[float] = None
    brier: Optional[float] = None
    reliability: Optional[float] = None
    resolution: Optional[float] = None
    uncertainty: Optional[float] = None
    ece: Optional[float] = None
    buckets: list[ProbabilityBucket] = Field(default_factory=list)
    finding: str


class ProbabilityCalibration(BaseModel):
    """Plan.md §27 / §55 for one forecast variant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    variant: Variant
    n_forecasts: int
    brier: Optional[float] = Field(default=None, description="Mean up/flat/down Brier score")
    events: dict[str, ReliabilityTable]


class VariantSummary(BaseModel):
    """Plan.md §28 metrics of one variant over one window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    variant: Variant
    n: int
    direction: HitRate
    brier: Optional[float] = None
    pinball: Optional[float] = None
    coverage_80pct: Optional[float] = None
    mean_signed_error_pct: Optional[float] = None
    mean_abs_error_pct: Optional[float] = None
    mean_vol_ratio: Optional[float] = None
    mean_sharpness_pct: Optional[float] = None
    # Plan.md §52 PIT distribution: calibrated quantiles give about 10%, 50% and 10%
    pit_below_p10: Optional[float] = None
    pit_below_p50: Optional[float] = None
    pit_above_p90: Optional[float] = None


class MeanEffect(BaseModel):
    """A mean difference in loss with its 95% Student-t interval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    mean: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None


class TrackRecordWindow(BaseModel):
    """All variants over the forecasts whose target date falls in one rolling window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    weeks: Optional[int] = Field(default=None, description="None: all time")
    start: Optional[date] = Field(default=None, description="First target date included")
    n: int
    n_adjusted: int = Field(default=0, description="Forecasts the LLM adjusted")
    n_calibrated: int = Field(default=0, description="Forecasts with a calibration applied")
    variants: list[VariantSummary]
    llm_value_added: MeanEffect = Field(description="Calibrated-quant Brier - final Brier")
    calibration_value_added: MeanEffect = Field(
        description="Uncalibrated-quant Brier - calibrated-quant Brier"
    )
    finding: str


class ForecastTrackRecord(BaseModel):
    """Plan.md §28 / §31: rolling 8-week, 26-week and all-time records as of ``as_of``.

    The evaluation keeps one record per source: simulated backtest weeks are
    never pooled with live forecasts (an LLM may know how those past weeks went).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    as_of: AwareDatetime
    ticker: Optional[str] = None
    source: Optional[ForecastSource] = Field(
        default=None, description="None: built from forecasts of any source"
    )
    n_forecasts: int
    n_inferred_source: int = Field(
        default=0, description="Forecasts stored before sources were recorded (source inferred)"
    )
    overlapping_excluded: int = 0
    min_samples: int
    windows: list[TrackRecordWindow]


class EvaluationReport(BaseModel):
    """Track records plus probability calibration of every probabilistic variant.

    ``track_records`` holds one record per source with scored forecasts (live
    first), or a single empty record when nothing is scored. Probability
    calibration pools both sources; ``sources`` gives the counts behind it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    track_records: list[ForecastTrackRecord] = Field(min_length=1)
    probability_calibration: list[ProbabilityCalibration]
    n_buckets: int
    sources: dict[str, int] = Field(default_factory=dict)

    @property
    def n_forecasts(self) -> int:
        """Independent scored forecasts over all sources."""
        return sum(r.n_forecasts for r in self.track_records)

    def record_for(self, source: ForecastSource) -> Optional[ForecastTrackRecord]:
        """The record of ``source``, if any of its forecasts are scored."""
        return next((r for r in self.track_records if r.source == source), None)
