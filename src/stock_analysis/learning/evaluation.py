"""Plan.md Phase 14: probability calibration and the rolling forecast track record.

Every scored original forecast is re-scored for each Plan.md §53 benchmark on
the same outcome: naive flat (no change), the uncalibrated quant baseline, the
calibrated quant baseline and the final forecast. The variant metrics use the
same definitions as the outcome scorer, applied to the probabilities and prices
stored in the immutable snapshot.

Probability calibration (§27) buckets each predicted probability and compares
the mean prediction with how often the event happened; the track record (§28)
summarises 8-week, 26-week and all-time windows. Both are computed on demand
from records that existed at ``as_of`` and leave out forecasts whose windows
overlap a later one (as the scorecards and calibration do).
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any, Optional

from stock_analysis.config.settings import Settings, get_settings
from stock_analysis.database.forecast_store import ForecastSnapshotStore
from stock_analysis.database.outcome_store import OutcomeStore
from stock_analysis.learning.scorecards import hit_rate, independent_forecasts, mean_ci
from stock_analysis.logging import get_logger
from stock_analysis.review import metrics
from stock_analysis.schemas.outcome import ForecastOutcome, OutcomeStatus
from stock_analysis.schemas.scorecard import ScoredForecast
from stock_analysis.schemas.snapshot import FORECAST_SOURCES, ForecastSnapshot, ForecastSource
from stock_analysis.schemas.track_record import (
    PROBABILISTIC_VARIANTS,
    VARIANTS,
    CalibrationEvent,
    EvaluatedForecast,
    EvaluationReport,
    ForecastTrackRecord,
    MeanEffect,
    ProbabilityBucket,
    ProbabilityCalibration,
    ReliabilityTable,
    TrackRecordWindow,
    Variant,
    VariantScore,
    VariantSummary,
)
from stock_analysis.snapshots.builder import IST

logger = get_logger(__name__)

# Plan.md §28 rolling windows: (label, weeks); None = all time
TRACK_RECORD_WINDOWS: tuple[tuple[str, Optional[int]], ...] = (
    ("8 weeks", 8),
    ("26 weeks", 26),
    ("All time", None),
)
_EVENTS: tuple[CalibrationEvent, ...] = ("up", "flat", "down")


def _probabilistic_score(
    values: dict[str, Any], snapshot: ForecastSnapshot, outcome: ForecastOutcome
) -> VariantScore:
    """Score one probabilistic variant exactly as ``score_forecast`` scores the final forecast."""
    assert outcome.actual_close_on_forecast_basis is not None
    assert outcome.actual_return_pct is not None and outcome.realized_direction is not None
    basis, last = outcome.actual_close_on_forecast_basis, snapshot.last_close
    up, flat, down = values["prob_up"], values["prob_flat"], values["prob_down"]
    p10, p50, p90 = values["p10_price"], values["p50_price"], values["p90_price"]
    weekly_vol = values.get("weekly_vol_pct")
    return VariantScore(
        prob_up=up,
        prob_flat=flat,
        prob_down=down,
        direction_correct=metrics.predicted_direction(up, flat, down) == outcome.realized_direction,
        signed_error_pct=metrics.pct_return(p50, basis),
        in_80pct_band=p10 <= basis <= p90,
        brier=metrics.brier_score(up, flat, down, outcome.realized_direction),
        pinball=metrics.quantile_losses(last, p10, p50, p90, outcome.actual_return_pct)["mean"],
        vol_ratio=(
            metrics.vol_ratio(outcome.realized_vol_pct, float(weekly_vol))
            if isinstance(weekly_vol, (int, float))
            else None
        ),
        sharpness_pct=(p90 - p10) / last * 100.0,
        pit_percentile=_pit(basis, p10, p50, p90),
    )


def _pit(actual: float, p10: float, p50: float, p90: float) -> Optional[float]:
    try:
        return metrics.pit_percentile(actual, p10, p50, p90)
    except ValueError:  # degenerate quantiles (e.g. P10 == P50) have no PIT
        return None


def evaluate_forecast(snapshot: ForecastSnapshot, outcome: ForecastOutcome) -> EvaluatedForecast:
    """Score every Plan.md §53 benchmark variant of ``snapshot`` on its scored ``outcome``."""
    if outcome.status != OutcomeStatus.SCORED or outcome.forecast_id != snapshot.forecast_id:
        raise ValueError("evaluate_forecast needs the scored outcome of this forecast")
    assert outcome.actual_close_on_forecast_basis is not None
    assert outcome.actual_return_pct is not None and outcome.realized_direction is not None
    variants: dict[str, VariantScore] = {
        # Naive flat: the price stays at the last close, so the error is the actual return
        "naive_flat": VariantScore(
            direction_correct=outcome.realized_direction == "flat",
            signed_error_pct=metrics.pct_return(
                snapshot.last_close, outcome.actual_close_on_forecast_basis
            ),
        ),
        "quant_uncalibrated": _probabilistic_score(snapshot.shadow_baseline, snapshot, outcome),
        "quant_calibrated": _probabilistic_score(snapshot.quant_baseline, snapshot, outcome),
        "final": _probabilistic_score(
            snapshot.final_forecast.model_dump(mode="json"), snapshot, outcome
        ),
    }
    uncal, cal = variants["quant_uncalibrated"].brier, variants["quant_calibrated"].brier
    return EvaluatedForecast(
        forecast_id=snapshot.forecast_id,
        ticker=snapshot.ticker,
        as_of_date=snapshot.as_of_date,
        target_date=snapshot.target_date,
        realized_direction=outcome.realized_direction,
        actual_return_pct=outcome.actual_return_pct,
        llm_value_added=outcome.llm_value_added,
        calibration_value_added=(uncal - cal) if uncal is not None and cal is not None else None,
        adjusted=snapshot.final_forecast.adjustment_applied,
        calibrated=snapshot.calibration is not None,
        source=snapshot.effective_source,
        source_inferred=snapshot.source_inferred,
        variants=variants,
    )


def load_evaluated_forecasts(
    snapshots: ForecastSnapshotStore,
    outcomes: OutcomeStore,
    *,
    as_of: datetime,
    ticker: Optional[str] = None,
) -> tuple[list[EvaluatedForecast], list[str]]:
    """Independent scored original forecasts known at ``as_of`` (oldest first) and the
    IDs left out because their windows overlap a later forecast of the same ticker."""
    kept, excluded = independent_forecasts(outcomes.scored_forecasts(before=as_of, ticker=ticker))
    return _evaluate(snapshots, outcomes, kept), excluded


def _evaluate(
    snapshots: ForecastSnapshotStore, outcomes: OutcomeStore, kept: Sequence[ScoredForecast]
) -> list[EvaluatedForecast]:
    evaluated = []
    for f in kept:
        snapshot, outcome = snapshots.get(f.forecast_id), outcomes.get(f.forecast_id)
        if snapshot is None or outcome is None:
            logger.warning("evaluation_record_missing", forecast_id=f.forecast_id)
            continue
        evaluated.append(evaluate_forecast(snapshot, outcome))
    return evaluated


# --- Probability calibration (Plan.md §27) -------------------------------------------


def bucket_index(probability: float, n_buckets: int) -> int:
    """Bucket ``i`` covers ``[i/n, (i+1)/n)``; the last one also includes 1.0."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability {probability} is outside [0, 1]")
    # The epsilon keeps values like 0.7 (0.7 * 10 = 6.9999...) in their intended bucket
    return min(int(probability * n_buckets + 1e-9), n_buckets - 1)


def _calibration_finding(n: int, buckets: list[ProbabilityBucket], minimum: int) -> str:
    if n < minimum:
        return f"insufficient evidence: {n} forecasts, at least {minimum} needed"
    too_high = [
        b
        for b in buckets
        if b.observed.ci_high is not None and b.mean_predicted > b.observed.ci_high
    ]
    too_low = [
        b for b in buckets if b.observed.ci_low is not None and b.mean_predicted < b.observed.ci_low
    ]
    if not too_high and not too_low:
        return (
            "observed frequencies are consistent with the predicted probabilities (95% intervals)"
        )

    def names(items: list[ProbabilityBucket]) -> str:
        return ", ".join(f"{b.low:.0%}–{b.high:.0%}" for b in items)

    parts = []
    if too_high:
        parts.append(f"happened less often than predicted in {names(too_high)}")
    if too_low:
        parts.append(f"happened more often than predicted in {names(too_low)}")
    return "the event " + "; ".join(parts) + " (beyond the 95% interval)"


def reliability_table(
    pairs: Sequence[tuple[float, bool]],
    *,
    event: CalibrationEvent,
    n_buckets: int,
    min_samples: int,
) -> ReliabilityTable:
    """Bucket ``(predicted probability, event happened)`` pairs and summarise calibration.

    Returns the binary Brier score with its Murphy decomposition (reliability,
    resolution, uncertainty) and the expected calibration error (ECE).
    """
    if n_buckets < 1:
        raise ValueError("n_buckets must be at least 1")
    n = len(pairs)
    if n == 0:
        return ReliabilityTable(event=event, n=0, finding=_calibration_finding(0, [], min_samples))
    grouped: dict[int, list[tuple[float, bool]]] = {}
    for p, happened in pairs:
        grouped.setdefault(bucket_index(p, n_buckets), []).append((p, happened))
    base_rate = sum(happened for _, happened in pairs) / n
    buckets, reliability, resolution, ece = [], 0.0, 0.0, 0.0
    for index in sorted(grouped):
        items = grouped[index]
        mean_predicted = statistics.fmean(p for p, _ in items)
        hits = sum(happened for _, happened in items)
        observed = hit_rate(hits, len(items))
        assert observed.rate is not None
        weight = len(items) / n
        reliability += weight * (mean_predicted - observed.rate) ** 2
        resolution += weight * (observed.rate - base_rate) ** 2
        ece += weight * abs(mean_predicted - observed.rate)
        buckets.append(
            ProbabilityBucket(
                low=index / n_buckets,
                high=(index + 1) / n_buckets,
                n=len(items),
                mean_predicted=mean_predicted,
                observed=observed,
            )
        )
    return ReliabilityTable(
        event=event,
        n=n,
        base_rate=base_rate,
        brier=statistics.fmean((p - float(happened)) ** 2 for p, happened in pairs),
        reliability=reliability,
        resolution=resolution,
        uncertainty=base_rate * (1.0 - base_rate),
        ece=ece,
        buckets=buckets,
        finding=_calibration_finding(n, buckets, min_samples),
    )


def probability_calibration(
    forecasts: Sequence[EvaluatedForecast],
    variant: Variant,
    *,
    n_buckets: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> ProbabilityCalibration:
    """Plan.md §27 for one probabilistic variant: P(up), P(flat), P(down) and all pooled.

    The pooled table counts each forecast three times (once per class), so its
    Brier score is a third of the mean up/flat/down Brier score.
    """
    if variant not in PROBABILISTIC_VARIANTS:
        raise ValueError(f"{variant} states no probabilities")
    settings = settings or get_settings()
    n_buckets = n_buckets or settings.probability_calibration_buckets
    minimum = settings.min_samples_probability_calibration
    pairs: dict[str, list[tuple[float, bool]]] = {event: [] for event in _EVENTS}
    briers = []
    for f in forecasts:
        score = f.variants[variant]
        probs = {"up": score.prob_up, "flat": score.prob_flat, "down": score.prob_down}
        for event in _EVENTS:
            p = probs[event]
            assert p is not None
            pairs[event].append((p, f.realized_direction == event))
        assert score.brier is not None
        briers.append(score.brier)
    events: dict[str, ReliabilityTable] = {
        event: reliability_table(
            pairs[event], event=event, n_buckets=n_buckets, min_samples=minimum
        )
        for event in _EVENTS
    }
    pooled = [pair for event in _EVENTS for pair in pairs[event]]
    # Each forecast adds three pairs; the evidence threshold is about forecasts
    events["all"] = reliability_table(
        pooled, event="all", n_buckets=n_buckets, min_samples=minimum * len(_EVENTS)
    )
    return ProbabilityCalibration(
        variant=variant,
        n_forecasts=len(forecasts),
        brier=statistics.fmean(briers) if briers else None,
        events=events,
    )


# --- Track record (Plan.md §28) --------------------------------------------------------


def _mean(values: list[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


def summarize_variant(forecasts: Sequence[EvaluatedForecast], variant: Variant) -> VariantSummary:
    """Plan.md §28 metrics of ``variant`` over ``forecasts``."""
    scores = [f.variants[variant] for f in forecasts]
    bands = [s.in_80pct_band for s in scores if s.in_80pct_band is not None]
    pits = [s.pit_percentile for s in scores if s.pit_percentile is not None]

    def pit_share(test: Callable[[float], bool]) -> Optional[float]:
        return sum(map(test, pits)) / len(pits) if pits else None

    return VariantSummary(
        variant=variant,
        n=len(scores),
        direction=hit_rate(sum(s.direction_correct for s in scores), len(scores)),
        brier=_mean([s.brier for s in scores]),
        pinball=_mean([s.pinball for s in scores]),
        coverage_80pct=statistics.fmean(map(float, bands)) if bands else None,
        mean_signed_error_pct=_mean([s.signed_error_pct for s in scores]),
        mean_abs_error_pct=_mean([abs(s.signed_error_pct) for s in scores]),
        mean_vol_ratio=_mean([s.vol_ratio for s in scores]),
        mean_sharpness_pct=_mean([s.sharpness_pct for s in scores]),
        pit_below_p10=pit_share(lambda p: p < 0.1),
        pit_below_p50=pit_share(lambda p: p < 0.5),
        pit_above_p90=pit_share(lambda p: p > 0.9),
    )


def _effect(values: list[Optional[float]]) -> MeanEffect:
    present = [v for v in values if v is not None]
    mean, low, high = mean_ci(present)
    return MeanEffect(n=len(present), mean=mean, ci_low=low, ci_high=high)


def _window_finding(n: int, n_adjusted: int, llm: MeanEffect, minimum: int) -> str:
    if n and not n_adjusted:
        return "no forecast was adjusted by the LLM: the final forecast was the quant baseline"
    if n < minimum or llm.ci_low is None or llm.ci_high is None:
        return (
            f"insufficient evidence: {n} forecasts, at least {minimum} needed before any "
            "variant can be called better"
        )
    if llm.ci_low > 0:
        return "the final forecast beat the calibrated quant baseline (Brier, 95% CI above zero)"
    if llm.ci_high < 0:
        return (
            "the final forecast was worse than the calibrated quant baseline "
            "(Brier, 95% CI below zero)"
        )
    return "no measurable difference between the final forecast and the calibrated quant baseline"


def track_record_window(
    forecasts: Sequence[EvaluatedForecast],
    label: str,
    *,
    weeks: Optional[int],
    as_of: datetime,
    min_samples: int,
) -> TrackRecordWindow:
    """Forecasts whose target date falls in the last ``weeks`` weeks up to ``as_of`` (IST)."""
    start = None
    selected = list(forecasts)
    if weeks is not None:
        start = as_of.astimezone(IST).date() - timedelta(weeks=weeks) + timedelta(days=1)
        selected = [f for f in forecasts if f.target_date >= start]
    llm = _effect([f.llm_value_added for f in selected])
    n_adjusted = sum(f.adjusted for f in selected)
    return TrackRecordWindow(
        label=label,
        weeks=weeks,
        start=start,
        n=len(selected),
        n_adjusted=n_adjusted,
        n_calibrated=sum(f.calibrated for f in selected),
        variants=[summarize_variant(selected, v) for v in VARIANTS],
        llm_value_added=llm,
        calibration_value_added=_effect([f.calibration_value_added for f in selected]),
        finding=_window_finding(len(selected), n_adjusted, llm, min_samples),
    )


def build_track_record(
    forecasts: Sequence[EvaluatedForecast],
    *,
    as_of: datetime,
    ticker: Optional[str] = None,
    source: Optional[ForecastSource] = None,
    overlapping_excluded: int = 0,
    settings: Optional[Settings] = None,
) -> ForecastTrackRecord:
    """Plan.md §28 rolling 8-week, 26-week and all-time windows.

    With ``source`` only that source's forecasts are included.
    """
    settings = settings or get_settings()
    minimum = settings.min_samples_decision_evaluation
    if source is not None:
        forecasts = [f for f in forecasts if f.source == source]
    return ForecastTrackRecord(
        as_of=as_of,
        ticker=ticker,
        source=source,
        n_forecasts=len(forecasts),
        n_inferred_source=sum(f.source_inferred for f in forecasts),
        overlapping_excluded=overlapping_excluded,
        min_samples=minimum,
        windows=[
            track_record_window(forecasts, label, weeks=weeks, as_of=as_of, min_samples=minimum)
            for label, weeks in TRACK_RECORD_WINDOWS
        ],
    )


def build_evaluation(
    snapshots: ForecastSnapshotStore,
    outcomes: OutcomeStore,
    *,
    as_of: datetime,
    ticker: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> EvaluationReport:
    """Track records (one per source) and probability calibration from the records that
    existed at ``as_of``.

    Probability calibration pools the sources, so overlaps are dropped across them;
    each track record drops overlaps within its own source.
    """
    settings = settings or get_settings()
    scored = outcomes.scored_forecasts(before=as_of, ticker=ticker)
    kept, excluded = independent_forecasts(scored)
    # Each source's record drops overlaps within that source only: a live forecast
    # is not left out because a later backtest week overlaps it
    by_source = {
        s: independent_forecasts([f for f in scored if f.source == s]) for s in FORECAST_SOURCES
    }
    needed = {f.forecast_id: f for f in kept}
    for own, _ in by_source.values():
        needed.update((f.forecast_id, f) for f in own)
    evaluated = {e.forecast_id: e for e in _evaluate(snapshots, outcomes, list(needed.values()))}

    def known(items: Sequence[ScoredForecast]) -> list[EvaluatedForecast]:
        return [evaluated[f.forecast_id] for f in items if f.forecast_id in evaluated]

    forecasts = known(kept)
    records = [
        build_track_record(
            known(own),
            as_of=as_of,
            ticker=ticker,
            source=source,
            overlapping_excluded=len(own_excluded),
            settings=settings,
        )
        for source, (own, own_excluded) in by_source.items()
        if own
    ] or [
        build_track_record(
            [], as_of=as_of, ticker=ticker, overlapping_excluded=len(excluded), settings=settings
        )
    ]
    sources = {s: sum(f.source == s for f in forecasts) for s in FORECAST_SOURCES}
    return EvaluationReport(
        track_records=records,
        probability_calibration=[
            probability_calibration(forecasts, v, settings=settings) for v in PROBABILISTIC_VARIANTS
        ],
        n_buckets=settings.probability_calibration_buckets,
        sources={s: n for s, n in sources.items() if n},
    )
