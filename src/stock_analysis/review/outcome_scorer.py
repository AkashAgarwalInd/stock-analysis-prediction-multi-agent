"""Plan.md §14 outcome scorer: pure Python, no LLM, never modifies the forecast.

``score_forecast`` compares one stored forecast snapshot with actual prices.
Only bars dated between the forecast's ``as_of_date`` and ``target_date`` are
used, and everything about the forecast side comes from the immutable
snapshot, so the evaluation cannot leak information the forecast did not have.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Optional

from stock_analysis.config.settings import get_settings
from stock_analysis.market.calendar import TradingCalendar, get_trading_calendar
from stock_analysis.quant import QuantForecaster
from stock_analysis.review import metrics
from stock_analysis.review.prices import PriceBar, PriceSeries
from stock_analysis.schemas.analyst_reports import AnalystStance, DecisionType
from stock_analysis.schemas.outcome import Direction, ForecastOutcome, OutcomeDaily, OutcomeStatus
from stock_analysis.schemas.snapshot import ForecastSnapshot
from stock_analysis.snapshots.builder import IST, NSE_CLOSE_IST

# Bump when any metric definition or validity rule changes.
# 2: also scores the uncalibrated shadow baseline (Plan.md §23).
SCORER_VERSION = "2"

# Price history re-adjusted by more than this since the forecast is noted in the audit trail
_ADJUSTMENT_NOTE_TOLERANCE = 0.005

_STANCE_DIRECTION: dict[str, Direction] = {
    AnalystStance.STRONG_BULLISH.value: "up",
    AnalystStance.BULLISH.value: "up",
    AnalystStance.NEUTRAL.value: "flat",
    AnalystStance.BEARISH.value: "down",
    AnalystStance.STRONG_BEARISH.value: "down",
}


def last_completed_trading_date(
    now: datetime, calendar: Optional[TradingCalendar] = None, delay_minutes: Optional[int] = None
) -> date:
    """The latest NSE session that has closed and had time to publish end-of-day data."""
    calendar = calendar or get_trading_calendar()
    delay = get_settings().outcome_data_delay_minutes if delay_minutes is None else delay_minutes
    local = now.astimezone(IST)
    published = datetime.combine(local.date(), NSE_CLOSE_IST, tzinfo=IST) + timedelta(minutes=delay)
    if calendar.is_trading_day(local.date()) and local >= published:
        return local.date()
    previous = calendar.previous_trading_day(local.date())
    if previous is None:
        raise ValueError(f"No trading day found before {local.date()}")
    return previous


def is_matured(
    target_date: date, now: datetime, calendar: Optional[TradingCalendar] = None
) -> bool:
    """A forecast is matured once its target date's session has completed (Plan.md §5.2)."""
    return target_date <= last_completed_trading_date(now, calendar)


def _base(snapshot: ForecastSnapshot, now: datetime) -> dict[str, Any]:
    """Identity, window and decision-engine attribution, copied from the snapshot."""
    gate = snapshot.decision(DecisionType.FORECAST_ADJUSTMENT_GATE)
    regime = snapshot.decision(DecisionType.MARKET_REGIME)
    final = snapshot.final_forecast
    return {
        "forecast_id": snapshot.forecast_id,
        "evaluated_at": now,
        "scorer_version": SCORER_VERSION,
        "as_of_date": snapshot.as_of_date,
        "target_date": snapshot.target_date,
        "decision_engine": (
            f"{gate.decision_model}@{gate.decision_model_version}" if gate else None
        ),
        "decision_engine_enabled": gate is not None and gate.decision_model != "guardrails",
        "market_regime_decision": regime.decision if regime else None,
        "adjustment_gate_decision": gate.decision if gate else None,
        "adjustment_applied": final.adjustment_applied,
        "fallback_to_quant": final.fallback_to_quant,
        "calibration_version": snapshot.calibration_version,
    }


def _not_scored(
    snapshot: ForecastSnapshot,
    now: datetime,
    reason: str,
    *,
    permanent: bool,
    checks: Optional[list[str]] = None,
    corporate_actions: Optional[list[dict[str, Any]]] = None,
) -> ForecastOutcome:
    return ForecastOutcome(
        **_base(snapshot, now),
        status=OutcomeStatus.INVALID if permanent else OutcomeStatus.UNRESOLVED,
        invalid_reason=reason,
        validity_checks=checks or [],
        corporate_actions=corporate_actions or [],
    )


def _corporate_actions(bars: list[PriceBar], as_of: date) -> list[dict[str, Any]]:
    actions = []
    for bar in bars:
        if bar.date <= as_of:
            continue
        if bar.split:
            actions.append({"date": bar.date.isoformat(), "type": "split", "ratio": bar.split})
        if bar.dividend:
            actions.append(
                {"date": bar.date.isoformat(), "type": "dividend", "amount": bar.dividend}
            )
    return actions


def _analyst_hits(snapshot: ForecastSnapshot, realized: Direction) -> Optional[dict[str, bool]]:
    hits = {}
    for name, report in snapshot.analyst_reports.items():
        if not report or not report.get("confidence"):
            continue  # missing or degraded (confidence 0) analysts made no call
        direction = _STANCE_DIRECTION.get(str(report.get("stance")))
        if direction is not None:
            hits[name] = direction == realized
    return hits or None


def score_forecast(
    snapshot: ForecastSnapshot,
    stock: PriceSeries,
    nifty: Optional[PriceSeries],
    *,
    now: datetime,
    calendar: Optional[TradingCalendar] = None,
    flat_threshold_pct: float = QuantForecaster.FLAT_THRESHOLD_PCT,
) -> ForecastOutcome:
    """Evaluate ``snapshot`` against actual prices (Plan.md §14-16).

    Returns an UNRESOLVED outcome when the forecast has not matured or data is
    not yet available (retry later), INVALID when the data cannot support a fair
    evaluation, and SCORED otherwise.
    """
    calendar = calendar or get_trading_calendar()
    settings = get_settings()
    as_of, target = snapshot.as_of_date, snapshot.target_date

    completed = last_completed_trading_date(now, calendar)
    if target > completed:
        return _not_scored(
            snapshot,
            now,
            f"Target session {target} has not completed (last completed: {completed})",
            permanent=False,
        )
    waited = len(calendar.get_trading_days(target + timedelta(days=1), completed))
    gave_up = waited > settings.outcome_max_wait_trading_days

    # Point-in-time: nothing dated after the target session is ever looked at
    bars = stock.window(as_of, target)
    if stock.error or not bars:
        reason = f"No price data for {stock.symbol}: {stock.error or 'empty response'}"
        return _not_scored(snapshot, now, reason, permanent=gave_up)

    # Trading dates and horizon
    expected = calendar.get_trading_days(as_of + timedelta(days=1), target)
    session_dates = [b.date for b in bars if b.date > as_of]
    missing = sorted(set(expected) - set(session_dates))
    if bars[0].date != as_of:
        missing.insert(0, as_of)
    if missing:
        reason = f"Missing prices for {', '.join(d.isoformat() for d in missing)}"
        return _not_scored(snapshot, now, reason, permanent=gave_up)
    extra = sorted(set(session_dates) - set(expected))
    if extra:
        reason = (
            "Data source has sessions the forecast calendar did not expect: "
            f"{', '.join(d.isoformat() for d in extra)}"
        )
        return _not_scored(snapshot, now, reason, permanent=True)
    if len(session_dates) != snapshot.horizon_trading_days:
        reason = (
            f"Window has {len(session_dates)} sessions but the forecast horizon is "
            f"{snapshot.horizon_trading_days}"
        )
        return _not_scored(snapshot, now, reason, permanent=True)
    checks = [f"{len(session_dates)} trading sessions from {as_of} to {target} match the horizon"]

    if any(b.close <= 0 for b in bars):
        return _not_scored(snapshot, now, "Non-positive close in window", permanent=True)

    # Corporate actions and split/dividend anomalies
    actions = _corporate_actions(bars, as_of)
    for prev, bar in zip(bars, bars[1:], strict=False):
        move = metrics.pct_return(prev.close, bar.close)
        if abs(move) > settings.outcome_max_daily_move_pct:
            what = (
                "split not reflected in adjusted closes"
                if bar.split
                else ("possible unadjusted corporate action")
            )
            reason = f"{move:+.1f}% move on {bar.date}: {what}"
            return _not_scored(
                snapshot, now, reason, permanent=True, checks=checks, corporate_actions=actions
            )
    if actions:
        checks.append("Corporate actions in window are reflected in the adjusted closes")
    adjustment_factor = bars[0].close / snapshot.last_close
    if abs(adjustment_factor - 1.0) > _ADJUSTMENT_NOTE_TOLERANCE:
        checks.append(
            f"Price history was re-adjusted after the forecast (factor {adjustment_factor:.4f}); "
            "returns are measured on the adjusted series"
        )

    # Actual result, expressed on the forecast's own price basis
    as_of_close, target_close = bars[0].close, bars[-1].close
    actual_return = metrics.pct_return(as_of_close, target_close)
    basis = snapshot.last_close * (1.0 + actual_return / 100.0)
    realized = metrics.realized_direction(actual_return, flat_threshold_pct)

    final = snapshot.final_forecast
    quant = snapshot.quant_baseline
    shadow = snapshot.shadow_baseline
    final_brier = metrics.brier_score(final.prob_up, final.prob_flat, final.prob_down, realized)
    base_brier = metrics.brier_score(
        quant["prob_up"], quant["prob_flat"], quant["prob_down"], realized
    )
    final_pinball = metrics.quantile_losses(
        snapshot.last_close, final.p10_price, final.p50_price, final.p90_price, actual_return
    )
    base_pinball = metrics.quantile_losses(
        snapshot.last_close,
        quant["p10_price"],
        quant["p50_price"],
        quant["p90_price"],
        actual_return,
    )
    shadow_brier = metrics.brier_score(
        shadow["prob_up"], shadow["prob_flat"], shadow["prob_down"], realized
    )
    shadow_pinball = metrics.quantile_losses(
        snapshot.last_close,
        shadow["p10_price"],
        shadow["p50_price"],
        shadow["p90_price"],
        actual_return,
    )
    predicted = metrics.predicted_direction(final.prob_up, final.prob_flat, final.prob_down)
    signed_error = metrics.pct_return(final.p50_price, basis)

    # Daily path (quant daily predictions) and benchmark
    daily_pred = {p.target_date: p for p in snapshot.daily_predictions}
    nifty_bars = {b.date: b.close for b in nifty.window(as_of, target)} if nifty else {}
    daily: list[OutcomeDaily] = []
    for bar in bars[1:]:
        close_basis = snapshot.last_close * bar.close / as_of_close
        pred = daily_pred.get(bar.date)
        daily.append(
            OutcomeDaily(
                date=bar.date,
                close=bar.close,
                close_on_forecast_basis=close_basis,
                nifty_close=nifty_bars.get(bar.date),
                in_daily_band=(pred.p10_price <= close_basis <= pred.p90_price if pred else None),
            )
        )
    band_flags = [d.in_daily_band for d in daily if d.in_daily_band is not None]

    nifty_return = excess = beta_excess = None
    if as_of in nifty_bars and target in nifty_bars:
        nifty_return = metrics.pct_return(nifty_bars[as_of], nifty_bars[target])
        excess = actual_return - nifty_return
        beta = (snapshot.data_inputs.get("market_context") or {}).get("beta")
        if isinstance(beta, (int, float)) and not isinstance(beta, bool):
            beta_excess = actual_return - beta * nifty_return
    else:
        checks.append("Nifty 50 data unavailable for the window; excess return not computed")

    realized_vol = metrics.realized_weekly_vol_pct([b.close for b in bars])

    return ForecastOutcome(
        **_base(snapshot, now),
        status=OutcomeStatus.SCORED,
        validity_checks=checks,
        corporate_actions=actions,
        adjustment_factor=adjustment_factor,
        actual_close=target_close,
        actual_close_on_forecast_basis=basis,
        actual_return_pct=actual_return,
        realized_direction=realized,
        predicted_direction=predicted,
        direction_correct=predicted == realized,
        signed_error_pct=signed_error,
        abs_error_pct=abs(signed_error),
        in_80pct_band=final.p10_price <= basis <= final.p90_price,
        daily_band_breaches=band_flags.count(False) if band_flags else None,
        pit_percentile=metrics.pit_percentile(
            basis, final.p10_price, final.p50_price, final.p90_price
        ),
        brier=final_brier,
        pinball=final_pinball,
        realized_vol_pct=realized_vol,
        vol_ratio=metrics.vol_ratio(realized_vol, final.weekly_vol_pct),
        nifty_return_pct=nifty_return,
        excess_vs_nifty_pct=excess,
        beta_adjusted_excess_pct=beta_excess,
        baseline_signed_error_pct=metrics.pct_return(quant["p50_price"], basis),
        baseline_brier=base_brier,
        baseline_pinball=base_pinball,
        baseline_loss=base_brier,
        final_loss=final_brier,
        llm_value_added=base_brier - final_brier,
        llm_value_added_pinball=base_pinball["mean"] - final_pinball["mean"],
        baseline_in_80pct_band=quant["p10_price"] <= basis <= quant["p90_price"],
        uncalibrated_signed_error_pct=metrics.pct_return(shadow["p50_price"], basis),
        uncalibrated_in_80pct_band=shadow["p10_price"] <= basis <= shadow["p90_price"],
        uncalibrated_brier=shadow_brier,
        uncalibrated_pinball=shadow_pinball,
        uncalibrated_loss=shadow_brier,
        calibration_value_added=shadow_brier - base_brier,
        calibration_value_added_pinball=shadow_pinball["mean"] - base_pinball["mean"],
        analyst_hits=_analyst_hits(snapshot, realized),
        daily=daily,
    )
