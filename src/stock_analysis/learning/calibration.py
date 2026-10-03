"""Plan.md §21-23: deterministic, bounded calibration of the quant baseline.

Two parameters per ticker, both measured against the *uncalibrated* (shadow)
baseline so successive calibrations never compound:

* ``vol_multiplier``: ``median(|residual| / predicted sigma) / 0.6745`` (the
  median absolute value of a standard normal), shrunk toward 1.0 with weight
  ``n / (n + k)`` and clipped to the configured bounds.
* ``p50_bias_shift_pct``: the median signed error, shrunk the same way and
  clipped, applied only when the bias is persistent (most errors share its
  sign) and statistically meaningful (|t| above the threshold).

Nothing changes until ``CALIBRATION_MIN_SAMPLES`` forecasts have been scored,
and forecasts whose windows overlap count once (the newest is kept): daily
forecasts during one bad fortnight see the same market move, so they are not
independent evidence. One bad week cannot move the model.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from datetime import datetime
from typing import Optional

from stock_analysis.config.settings import Settings, get_settings
from stock_analysis.database.forecast_store import ForecastSnapshotStore
from stock_analysis.database.learning_store import LearningStore
from stock_analysis.database.outcome_store import OutcomeStore
from stock_analysis.logging import get_logger
from stock_analysis.schemas.learning import (
    BenchmarkComparison,
    CalibrationEstimate,
    CalibrationObservation,
    CalibrationParams,
    CalibrationUpdate,
)
from stock_analysis.schemas.outcome import ForecastOutcome, OutcomeStatus
from stock_analysis.schemas.snapshot import ForecastSnapshot
from stock_analysis.versions import CALIBRATOR_VERSION

logger = get_logger(__name__)

NORMAL_MEDIAN_ABS = 0.6745  # median |Z| for Z ~ N(0, 1)
_Z_80PCT = 1.2816  # |Z| beyond this falls outside a central 80% band
# A new calibration version is stored only for a change at least this large
MIN_VOL_MULTIPLIER_CHANGE = 0.01
MIN_BIAS_SHIFT_CHANGE_PCT = 0.05
_DECIMALS = 4
_OVERLAP_LOOKBACK = 5  # candidates fetched per observation needed (daily forecasts, 5-day windows)


def observation_from(
    snapshot: ForecastSnapshot, outcome: ForecastOutcome
) -> Optional[CalibrationObservation]:
    """Measure one scored forecast against its uncalibrated baseline (None if unusable)."""
    basis = outcome.actual_close_on_forecast_basis
    if outcome.status != OutcomeStatus.SCORED or basis is None or basis <= 0:
        return None
    shadow = snapshot.shadow_baseline
    weekly_vol = float(shadow.get("weekly_vol_pct") or 0.0)
    # weekly_vol_pct is a 5-day figure; rescale if the horizon differs
    sigma = weekly_vol / 100.0 * math.sqrt(snapshot.horizon_trading_days / 5)
    if sigma <= 0 or shadow.get("p50_price", 0) <= 0:
        return None
    return CalibrationObservation(
        forecast_id=snapshot.forecast_id,
        as_of_date=snapshot.as_of_date,
        target_date=snapshot.target_date,
        residual_log=math.log(basis / float(shadow["p50_price"])),
        predicted_sigma_log=sigma,
    )


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def estimate_calibration(
    observations: list[CalibrationObservation],
    settings: Optional[Settings] = None,
    *,
    excluded_overlapping: Sequence[str] = (),
) -> CalibrationEstimate:
    """Estimate the calibration from independent scored forecasts (pure; no I/O).

    ``excluded_overlapping`` only documents forecasts left out by the caller.
    """
    settings = settings or get_settings()
    n = len(observations)
    ids = [o.forecast_id for o in observations]
    excluded = list(excluded_overlapping)
    # A spread needs at least two observations, whatever the configured minimum
    if n < max(settings.calibration_min_samples, 2):
        return CalibrationEstimate(
            vol_multiplier=1.0,
            p50_bias_shift_pct=0.0,
            n_samples=n,
            reason=[
                f"Only {n} scored forecasts; calibration needs at least "
                f"{settings.calibration_min_samples}, so the model is left unchanged"
            ],
            evidence={
                "forecast_ids": ids,
                "min_samples": settings.calibration_min_samples,
                "overlapping_forecasts_excluded": excluded,
            },
        )

    weight = n / (n + settings.calibration_shrinkage_k)
    reasons: list[str] = []

    # P50 bias (Plan.md §22)
    errors_pct = [(math.exp(o.residual_log) - 1.0) * 100.0 for o in observations]
    mean_err = statistics.fmean(errors_pct)
    median_err = statistics.median(errors_pct)
    sd = statistics.stdev(errors_pct)
    t_stat = mean_err / (sd / math.sqrt(n)) if sd > 0 else (math.inf if mean_err else 0.0)
    direction = _sign(median_err)
    agreement = sum(_sign(e) == direction for e in errors_pct) / n if direction else 0.0
    bias_eligible = (
        direction != 0
        and _sign(mean_err) == direction
        and abs(t_stat) >= settings.calibration_bias_min_t_stat
        and agreement >= settings.calibration_bias_min_sign_agreement
    )
    bias_shift = 0.0
    if bias_eligible:
        cap = settings.calibration_bias_max_shift_pct
        bias_shift = max(-cap, min(cap, weight * median_err))
        reasons.append(
            f"Actual closes were persistently {'above' if direction > 0 else 'below'} the "
            f"quant P50 (median error {median_err:+.2f}%, t = {t_stat:.2f}, "
            f"{agreement:.0%} of errors with that sign); P50 shift {bias_shift:+.2f}% "
            f"after shrinkage (weight {weight:.2f}) and the ±{cap:.2f}% bound"
        )
    else:
        reasons.append(
            f"No P50 bias applied: median error {median_err:+.2f}%, t = {t_stat:.2f}, "
            f"{agreement:.0%} sign agreement is not persistent and significant enough"
        )

    # Volatility multiplier (Plan.md §21), measured around the (possibly shifted) centre
    centre = math.log1p(bias_shift / 100.0)
    abs_z = [abs(o.residual_log - centre) / o.predicted_sigma_log for o in observations]
    raw_multiplier = statistics.median(abs_z) / NORMAL_MEDIAN_ABS
    shrunk = 1.0 + weight * (raw_multiplier - 1.0)
    lo, hi = settings.calibration_vol_multiplier_min, settings.calibration_vol_multiplier_max
    vol_multiplier = max(lo, min(hi, shrunk))
    exceeded = sum(z > 1.0 for z in abs_z)
    outside = sum(abs(o.residual_log) / o.predicted_sigma_log > _Z_80PCT for o in observations)
    reasons.append(
        f"{exceeded} of the last {n} forecasts missed by more than the predicted volatility "
        f"and {outside} fell outside the uncalibrated 80% band; raw multiplier "
        f"{raw_multiplier:.2f}, shrunk toward 1.0 (weight {weight:.2f}) to {shrunk:.2f}"
        + (f", bounded to {vol_multiplier:.2f}" if vol_multiplier != shrunk else "")
    )

    return CalibrationEstimate(
        vol_multiplier=round(vol_multiplier, _DECIMALS),
        p50_bias_shift_pct=round(bias_shift, _DECIMALS),
        n_samples=n,
        reason=reasons,
        evidence={
            "forecast_ids": ids,
            "shrinkage_weight": round(weight, _DECIMALS),
            "raw_vol_multiplier": round(raw_multiplier, _DECIMALS),
            "forecasts_exceeding_predicted_vol": exceeded,
            "forecasts_outside_uncalibrated_80pct_band": outside,
            "mean_signed_error_pct": round(mean_err, _DECIMALS),
            "median_signed_error_pct": round(median_err, _DECIMALS),
            "bias_t_stat": round(t_stat, _DECIMALS) if math.isfinite(t_stat) else None,
            "bias_sign_agreement": round(agreement, _DECIMALS),
            "bias_applied": bias_eligible,
            "overlapping_forecasts_excluded": excluded,
        },
    )


def load_observations(
    ticker: str,
    *,
    before: datetime,
    snapshot_store: ForecastSnapshotStore,
    outcome_store: OutcomeStore,
    learning_store: LearningStore,
    window: int,
) -> tuple[list[CalibrationObservation], list[str]]:
    """Up to ``window`` independent observations scored by ``before`` (oldest first),
    plus the IDs of forecasts skipped because their window overlaps a newer one kept.
    """
    observations: list[CalibrationObservation] = []
    excluded: list[str] = []
    # Newest first; overlapping forecasts can be several per week, so look further back
    candidates = learning_store.scored_forecast_ids(
        ticker, before=before, limit=window * _OVERLAP_LOOKBACK
    )
    for forecast_id in candidates:
        if len(observations) >= window:
            break
        snapshot = snapshot_store.get(forecast_id)
        outcome = outcome_store.get(forecast_id)
        if snapshot is None or outcome is None:
            continue
        obs = observation_from(snapshot, outcome)
        if obs is None:
            continue
        if any(_overlaps(obs, kept) for kept in observations):
            excluded.append(obs.forecast_id)
            continue
        observations.append(obs)
    observations.sort(key=lambda o: (o.target_date, o.forecast_id))
    return observations, excluded


def _overlaps(a: CalibrationObservation, b: CalibrationObservation) -> bool:
    """Windows sharing more than an endpoint cover the same price moves."""
    return a.as_of_date < b.target_date and b.as_of_date < a.target_date


def update_calibration(
    ticker: str,
    *,
    now: datetime,
    snapshot_store: ForecastSnapshotStore,
    outcome_store: OutcomeStore,
    learning_store: LearningStore,
    settings: Optional[Settings] = None,
) -> CalibrationUpdate:
    """Re-estimate ``ticker``'s calibration; store a new version only if it changed materially."""
    settings = settings or get_settings()
    observations, excluded = load_observations(
        ticker,
        before=now,
        snapshot_store=snapshot_store,
        outcome_store=outcome_store,
        learning_store=learning_store,
        window=settings.calibration_window,
    )
    estimate = estimate_calibration(observations, settings, excluded_overlapping=excluded)
    previous = learning_store.latest_calibration(ticker, as_of=now)
    prev_version = previous.version if previous else 0
    prev_vol = previous.vol_multiplier if previous else 1.0
    prev_bias = previous.p50_bias_shift_pct if previous else 0.0
    changed = (
        abs(estimate.vol_multiplier - prev_vol) >= MIN_VOL_MULTIPLIER_CHANGE
        or abs(estimate.p50_bias_shift_pct - prev_bias) >= MIN_BIAS_SHIFT_CHANGE_PCT
    )
    stored = None
    if changed:
        stored = CalibrationParams(
            ticker=ticker,
            version=prev_version + 1,
            vol_multiplier=estimate.vol_multiplier,
            p50_bias_shift_pct=estimate.p50_bias_shift_pct,
            previous_vol_multiplier=prev_vol,
            previous_p50_bias_shift_pct=prev_bias,
            n_samples=estimate.n_samples,
            reason=estimate.reason,
            evidence=estimate.evidence,
            calibrator_version=CALIBRATOR_VERSION,
            created_at=now,
        )
        learning_store.save_calibration(stored)
    logger.info(
        "calibration_reviewed",
        ticker=ticker,
        n_samples=estimate.n_samples,
        changed=changed,
        vol_multiplier=estimate.vol_multiplier,
        p50_bias_shift_pct=estimate.p50_bias_shift_pct,
    )
    return CalibrationUpdate(
        ticker=ticker,
        changed=changed,
        previous_version=prev_version,
        previous_vol_multiplier=prev_vol,
        previous_p50_bias_shift_pct=prev_bias,
        estimate=estimate,
        stored=stored,
    )


def compare_benchmarks(outcomes: list[ForecastOutcome]) -> BenchmarkComparison:
    """Uncalibrated quant vs calibrated quant vs final forecast on the same scored outcomes."""
    scored = [o for o in outcomes if o.status == OutcomeStatus.SCORED]

    def mean(values: list[Optional[float]]) -> Optional[float]:
        present = [v for v in values if v is not None]
        return statistics.fmean(present) if present else None

    uncal = mean(
        [
            o.uncalibrated_loss if o.uncalibrated_loss is not None else o.baseline_loss
            for o in scored
        ]
    )
    cal = mean([o.baseline_loss for o in scored])
    final = mean([o.final_loss for o in scored])
    calibrated = [o for o in scored if o.calibration_version > 0]

    if not scored:
        finding = "No scored forecasts yet."
    elif not calibrated:
        finding = (
            "No scored forecast used a calibration yet, so the calibrated and uncalibrated "
            "quant baselines are identical."
        )
    else:
        before = mean([o.uncalibrated_loss for o in calibrated])
        after = mean([o.baseline_loss for o in calibrated])
        if before is None or after is None:
            finding = "Calibrated forecasts lack shadow scores; calibration cannot be assessed."
        elif after < before:
            finding = (
                f"On the {len(calibrated)} forecasts with calibration applied, calibration "
                f"lowered the mean Brier loss from {before:.4f} to {after:.4f}."
            )
        else:
            finding = (
                f"On the {len(calibrated)} forecasts with calibration applied, calibration did "
                f"not lower the mean Brier loss ({before:.4f} uncalibrated vs {after:.4f} "
                "calibrated); consider LEARNING_ENABLED=false."
            )
    return BenchmarkComparison(
        n=len(scored),
        n_calibrated=len(calibrated),
        mean_uncalibrated_loss=uncal,
        mean_calibrated_loss=cal,
        mean_final_loss=final,
        finding=finding,
    )
