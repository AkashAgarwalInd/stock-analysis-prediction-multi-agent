"""Plan.md Phase 16: what a forecast run learns from earlier reviews.

``add_learning_context`` extends the memory context with the latest forecast
review (actual vs forecast, error, postmortem cause, lessons), the calibration
the quant baseline will use and how it changed, the ticker's scorecards
(analyst hit rates, quality per regime) and its rolling track record. Every
read is point-in-time (``as_of``), every number is copied from stored records,
and a part that fails to load is skipped with a warning so the forecast is
never blocked.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime
from typing import Optional

from stock_analysis.config.settings import get_settings
from stock_analysis.database.forecast_store import ForecastSnapshotError, ForecastSnapshotStore
from stock_analysis.database.learning_store import LearningStore, LearningStoreError
from stock_analysis.database.memory_store import MemoryStore, MemoryStoreError
from stock_analysis.database.outcome_store import OutcomeStore, OutcomeStoreError
from stock_analysis.learning.evaluation import build_evaluation
from stock_analysis.learning.scorecards import build_scorecards
from stock_analysis.logging import get_logger
from stock_analysis.review.metrics import predicted_direction
from stock_analysis.schemas.memory import (
    Adaptation,
    LastReview,
    LessonNote,
    MemoryContext,
    OutcomeMetrics,
    ScorecardSummary,
    VariantMetrics,
    WindowMetrics,
)
from stock_analysis.schemas.outcome import OutcomeStatus
from stock_analysis.schemas.scorecard import HitRate, Scorecards
from stock_analysis.schemas.track_record import VARIANT_LABELS, ForecastTrackRecord

logger = get_logger(__name__)

# ValueError includes pydantic's ValidationError
_LOAD_ERRORS = (
    ForecastSnapshotError,
    OutcomeStoreError,
    LearningStoreError,
    MemoryStoreError,
    sqlite3.Error,
    ValueError,
)


def load_last_review(
    ticker: str,
    *,
    as_of: datetime,
    snapshot_store: ForecastSnapshotStore,
    outcome_store: OutcomeStore,
    learning_store: Optional[LearningStore] = None,
    memory_store: Optional[MemoryStore] = None,
) -> Optional[LastReview]:
    """The newest forecast of ``ticker`` evaluated before ``as_of``, or None."""
    outcome = outcome_store.latest_evaluated(ticker, before=as_of)
    if outcome is None:
        return None
    snapshot = snapshot_store.get(outcome.forecast_id)
    if snapshot is None:
        raise ForecastSnapshotError(
            f"Forecast {outcome.forecast_id} has an outcome but no snapshot"
        )

    postmortem = learning_store.get_postmortem(outcome.forecast_id) if learning_store else None
    if postmortem is not None and postmortem.created_at > as_of:
        postmortem = None  # written after this run's clock
    lessons = (
        memory_store.lessons_for_forecast(outcome.forecast_id, as_of=as_of) if memory_store else []
    )
    final = snapshot.final_forecast
    base = snapshot.quant_baseline
    base_direction = None
    if all(isinstance(base.get(k), (int, float)) for k in ("prob_up", "prob_flat", "prob_down")):
        base_direction = predicted_direction(base["prob_up"], base["prob_flat"], base["prob_down"])
    return LastReview(
        forecast_id=outcome.forecast_id,
        as_of_date=snapshot.as_of_date,
        target_date=snapshot.target_date,
        evaluated_at=outcome.evaluated_at,
        status="scored" if outcome.status == OutcomeStatus.SCORED else "invalid",
        invalid_reason=outcome.invalid_reason,
        last_close=snapshot.last_close,
        prob_up=final.prob_up,
        prob_flat=final.prob_flat,
        prob_down=final.prob_down,
        p10_price=final.p10_price,
        p50_price=final.p50_price,
        p90_price=final.p90_price,
        adjustment_applied=final.adjustment_applied,
        calibration_version=snapshot.calibration_version,
        baseline_prob_up=base.get("prob_up"),
        baseline_p10_price=base.get("p10_price"),
        baseline_p50_price=base.get("p50_price"),
        baseline_p90_price=base.get("p90_price"),
        baseline_direction=base_direction,
        baseline_direction_correct=(
            base_direction == outcome.realized_direction
            if base_direction and outcome.realized_direction
            else None
        ),
        baseline_signed_error_pct=outcome.baseline_signed_error_pct,
        baseline_in_80pct_band=outcome.baseline_in_80pct_band,
        predicted_direction=outcome.predicted_direction,
        realized_direction=outcome.realized_direction,
        direction_correct=outcome.direction_correct,
        actual_close=outcome.actual_close_on_forecast_basis,
        actual_return_pct=outcome.actual_return_pct,
        signed_error_pct=outcome.signed_error_pct,
        in_80pct_band=outcome.in_80pct_band,
        vol_ratio=outcome.vol_ratio,
        nifty_return_pct=outcome.nifty_return_pct,
        excess_vs_nifty_pct=outcome.excess_vs_nifty_pct,
        baseline_loss=outcome.baseline_loss,
        final_loss=outcome.final_loss,
        llm_value_added=outcome.llm_value_added,
        calibration_value_added=outcome.calibration_value_added,
        primary_cause=postmortem.primary_cause.value if postmortem else None,
        cause_explanation=postmortem.explanation[:500] if postmortem else None,
        expected_noise=postmortem.expected_noise if postmortem else None,
        lessons=[
            LessonNote(
                lesson_id=lesson.lesson_id,
                text=lesson.text,
                scope=lesson.scope,
                status=lesson.status,
                evidence_count=lesson.evidence_count,
            )
            for lesson in lessons
        ],
    )


def load_adaptation(
    ticker: str,
    *,
    as_of: datetime,
    learning_store: LearningStore,
    previous_forecast_at: Optional[datetime] = None,
) -> Adaptation:
    """The calibration in force for ``ticker`` at ``as_of`` (version 0: none yet)."""
    settings = get_settings()
    params = learning_store.latest_calibration(ticker, as_of=as_of)
    if params is None:
        return Adaptation(
            version=0,
            min_samples=settings.calibration_min_samples,
            applied=False,
            changed_since_last_forecast=False if previous_forecast_at else None,
        )
    return Adaptation(
        version=params.version,
        vol_multiplier=params.vol_multiplier,
        p50_bias_shift_pct=params.p50_bias_shift_pct,
        previous_vol_multiplier=params.previous_vol_multiplier,
        previous_p50_bias_shift_pct=params.previous_p50_bias_shift_pct,
        created_at=params.created_at,
        n_samples=params.n_samples,
        reason=params.reason,
        min_samples=settings.calibration_min_samples,
        applied=settings.learning_enabled,
        changed_since_last_forecast=(
            params.created_at > previous_forecast_at if previous_forecast_at else None
        ),
    )


def summarize_scorecards(cards: Scorecards) -> ScorecardSummary:
    """The prompt-sized part of the scorecards: pooled analyst hit rates and regimes."""
    return ScorecardSummary(
        n_forecasts=cards.n_forecasts,
        min_samples=cards.min_samples,
        analysts={a.analyst: a.pooled for a in cards.analysts if a.pooled.n},
        regimes=cards.by_regime,
    )


def summarize_track_record(record: ForecastTrackRecord) -> OutcomeMetrics:
    """The final forecast's metrics per rolling window, with every benchmark's."""
    windows = []
    for w in record.windows:
        final = next((v for v in w.variants if v.variant == "final"), None)
        windows.append(
            WindowMetrics(
                label=w.label,
                n=w.n,
                direction=final.direction if final else HitRate(n=0, hits=0),
                coverage_80pct=final.coverage_80pct if final else None,
                brier=final.brier if final else None,
                mean_abs_error_pct=final.mean_abs_error_pct if final else None,
                mean_signed_error_pct=final.mean_signed_error_pct if final else None,
                n_adjusted=w.n_adjusted,
                llm_value_added=w.llm_value_added.mean if w.n_adjusted else None,
                finding=w.finding,
                variants=[
                    VariantMetrics(
                        variant=v.variant,
                        label=VARIANT_LABELS[v.variant],
                        direction=v.direction,
                        coverage_80pct=v.coverage_80pct,
                        brier=v.brier,
                        mean_abs_error_pct=v.mean_abs_error_pct,
                    )
                    for v in w.variants
                ],
            )
        )
    return OutcomeMetrics(
        n_forecasts=record.n_forecasts, min_samples=record.min_samples, windows=windows
    )


def add_learning_context(
    context: MemoryContext,
    ticker: str,
    *,
    as_of: datetime,
    snapshot_store: Optional[ForecastSnapshotStore] = None,
    outcome_store: Optional[OutcomeStore] = None,
    learning_store: Optional[LearningStore] = None,
    memory_store: Optional[MemoryStore] = None,
) -> MemoryContext:
    """Return ``context`` with the review, adaptation, scorecards and track record added.

    Loaded parts are listed in ``learning_loaded``. A part whose stores are not
    given stays ``None``; a part that fails is skipped and noted in
    ``warnings``: learning context never blocks a forecast.
    """
    update: dict[str, object] = {}
    loaded = list(context.learning_loaded)
    warnings = list(context.warnings)

    def attempt(part: str, load: Callable[[], object]) -> None:
        try:
            update[part] = load()
        except _LOAD_ERRORS as err:
            logger.warning("learning_context_failed", part=part, ticker=ticker, error=str(err))
            warnings.append(f"{part} could not be loaded: {err}"[:300])
        except Exception as err:  # an unexpected failure must not block the forecast either
            logger.exception("learning_context_error", part=part, ticker=ticker)
            warnings.append(f"{part} could not be loaded: {type(err).__name__}: {err}"[:300])
        else:
            loaded.append(part)

    if snapshot_store is not None and outcome_store is not None:
        attempt(
            "last_review",
            lambda: load_last_review(
                ticker,
                as_of=as_of,
                snapshot_store=snapshot_store,
                outcome_store=outcome_store,
                learning_store=learning_store,
                memory_store=memory_store,
            ),
        )
        if context.track_record is not None:
            record = context.track_record

            def metrics() -> object:
                evaluation = build_evaluation(
                    snapshot_store, outcome_store, as_of=as_of, ticker=ticker
                )
                if not evaluation.track_record.n_forecasts:
                    return record
                return record.model_copy(
                    update={"outcome_metrics": summarize_track_record(evaluation.track_record)}
                )

            attempt("track_record", metrics)
    if learning_store is not None:
        previous = context.prior_forecasts[0].made_at if context.prior_forecasts else None
        attempt(
            "adaptation",
            lambda: load_adaptation(
                ticker, as_of=as_of, learning_store=learning_store, previous_forecast_at=previous
            ),
        )
        if outcome_store is not None:
            attempt(
                "scorecards",
                lambda: summarize_scorecards(
                    build_scorecards(outcome_store, learning_store, as_of=as_of, ticker=ticker)
                ),
            )
    return context.model_copy(update={**update, "learning_loaded": loaded, "warnings": warnings})
