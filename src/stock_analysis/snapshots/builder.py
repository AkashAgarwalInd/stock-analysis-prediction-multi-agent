"""Assemble an immutable ``ForecastSnapshot`` from a completed graph state.

The builder only copies and organises values already present in state; it does
not recompute or adjust any forecast number.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time
from typing import Any, Optional
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from stock_analysis.config.settings import get_settings
from stock_analysis.guardrails import get_critical_gaps
from stock_analysis.market.calendar import TradingCalendar, get_trading_calendar
from stock_analysis.schemas.analyst_reports import (
    AnalystReport,
    DecisionResult,
    MarketRegime,
    RiskCategory,
)
from stock_analysis.schemas.forecast_pipeline import CriticResult, FinalForecast
from stock_analysis.schemas.graph_state import GraphState
from stock_analysis.schemas.snapshot import (
    UNCALIBRATED_VERSION,
    AppliedCalibration,
    DailyPrediction,
    DecisionAuditRecord,
    ForecastSnapshot,
    canonical_json,
    compute_data_snapshot_id,
    new_forecast_id,
)
from stock_analysis.versions import get_model_versions, get_prompt_versions

IST = ZoneInfo("Asia/Kolkata")
NSE_CLOSE_IST = time(15, 30)

_ANALYST_FIELDS = {
    "technical": "technical_report",
    "fundamental": "fundamental_report",
    "sentiment": "sentiment_report",
    "context": "context_report",
}


class SnapshotNotReadyError(ValueError):
    """The graph state does not contain a completed, valid forecast."""


def _plain(value: Any) -> Any:
    """Round-trip through JSON so enums/dates become plain, hashable JSON values."""
    return json.loads(canonical_json(value))


def collect_data_inputs(state: GraphState) -> dict[str, Any]:
    """The compact inputs a forecast was generated from (hashed into data_snapshot_id)."""
    inputs: dict[str, Any] = _plain(
        {
            "symbol": state.symbol,
            "resolved_symbol": state.resolved_symbol,
            "price": state.price_snapshot,
            "technical_indicators": state.technical_indicators_summary,
            "fundamentals": state.fundamentals_summary,
            "news": state.news_summary,
            "market_context": state.market_context_summary,
            # Prior context the run saw (Plan.md §35: "what did the model actually see?")
            "memory": state.memory_context,
        }
    )
    if state.disabled_analysts:
        # Only recorded when used, so live snapshots keep their existing input hash
        inputs["disabled_analysts"] = _plain(state.disabled_analysts)
    if state.review_summary is not None:
        # The review run just before this forecast (Plan.md §5.2); only when one ran
        inputs["review"] = _plain(state.review_summary)
    return inputs


def _analyst_evidence(reports: dict[str, Optional[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Per-analyst facts the rules engine keys on: stance, confidence, critical gaps."""
    out = []
    for name, rpt in reports.items():
        entry: dict[str, Any] = {"analyst": name, "present": rpt is not None}
        if rpt is not None:
            try:
                parsed = AnalystReport(**rpt)
                entry.update(valid=True, stance=parsed.stance.value, confidence=parsed.confidence)
            except ValidationError:
                entry["valid"] = False
            entry["critical_gaps"] = [g["description"] for g in get_critical_gaps([rpt])]
        out.append(entry)
    return out


def _decision_record(
    decision: DecisionResult, engine: str, evidence: dict[str, Any]
) -> DecisionAuditRecord:
    return DecisionAuditRecord(
        decision_engine=engine,
        decision_model=decision.model,
        decision_model_version=decision.model_version,
        decision_type=decision.decision_type,
        decision=decision.result,
        confidence=decision.confidence,
        rationale=decision.rationale,
        evidence=_plain(evidence),
    )


def _trading_dates(as_of: date, n: int, calendar: TradingCalendar) -> list[date]:
    dates: list[date] = []
    current = as_of
    for _ in range(n):
        nxt = calendar.next_trading_day(current)
        if nxt is None:
            raise SnapshotNotReadyError(f"No trading day found after {current}")
        dates.append(nxt)
        current = nxt
    return dates


def _data_quality_warnings(
    as_of_date: date, trading_dates: list[date], made_at: datetime, calendar: TradingCalendar
) -> list[str]:
    """Risks to the forecast dates/prices that the inputs themselves cannot reveal."""
    warnings = []
    uncovered = sorted({d.year for d in trading_dates if not calendar.covers(d)})
    if uncovered:
        years = ", ".join(str(y) for y in uncovered)
        warnings.append(
            f"NSE holiday calendar does not cover {years}; horizon dates may include "
            "unlisted holidays"
        )
    made_ist = made_at.astimezone(IST)
    if as_of_date == made_ist.date() and made_ist.time() < NSE_CLOSE_IST:
        warnings.append(
            "Forecast was made before the NSE close on its as-of date; the last price bar "
            "may be an unclosed intraday bar"
        )
    return warnings


def build_forecast_snapshot(
    state: GraphState,
    *,
    made_at: Optional[datetime] = None,
    calendar: Optional[TradingCalendar] = None,
) -> ForecastSnapshot:
    """Build the immutable snapshot for the forecast in ``state``.

    The calibration applied by the quant node (if any) and the uncalibrated
    quant baseline (the shadow forecast) are always recorded.

    Raises:
        SnapshotNotReadyError: no valid final forecast / quant baseline in state.
    """
    raw_final = state.final_forecast
    if not raw_final or "error" in raw_final:
        reason = (raw_final or {}).get("error", "missing final forecast")
        raise SnapshotNotReadyError(f"No completed forecast to snapshot: {reason}")
    quant = state.quant_baseline
    if not quant or "error" in quant:
        raise SnapshotNotReadyError("No valid quant baseline to snapshot")

    final = FinalForecast(**raw_final)
    calibration = (
        AppliedCalibration.model_validate(state.calibration) if state.calibration else None
    )
    if calibration is not None and not state.quant_baseline_uncalibrated:
        raise SnapshotNotReadyError("Calibrated forecast is missing its uncalibrated baseline")
    made_at = made_at or state.run_at or datetime.now(UTC)
    calendar = calendar or get_trading_calendar()

    price = state.price_snapshot or {}
    as_of_date = (
        date.fromisoformat(price["last_date"])
        if price.get("last_date")
        else made_at.astimezone(IST).date()
    )
    horizon = int(quant["horizon_trading_days"])
    trading_dates = _trading_dates(as_of_date, horizon, calendar)

    reports = {name: getattr(state, field) for name, field in _ANALYST_FIELDS.items()}
    analysts = _analyst_evidence(reports)
    violations = [v.get("message") for v in state.guardrail_violations]
    settings = get_settings()

    decisions: list[DecisionAuditRecord] = []
    regime = MarketRegime.MIXED
    if state.decision is not None:
        decisions.append(
            _decision_record(
                state.decision,
                "guardrails" if state.decision.model == "guardrails" else "decision_engine",
                {"analysts": analysts, "guardrail_violations": violations},
            )
        )
        regime = MarketRegime(state.decision.result)
    risk = RiskCategory.HIGH
    if state.risk_decision is not None:
        decisions.append(
            _decision_record(
                state.risk_decision,
                "decision_engine",
                {"analysts": analysts, "guardrail_violations": violations},
            )
        )
        risk = RiskCategory(state.risk_decision.result)
    if state.adjustment_gate_decision is not None:
        decisions.append(
            _decision_record(
                state.adjustment_gate_decision,
                "adjustment_gate",
                {
                    "analysts": analysts,
                    "guardrail_violations": violations,
                    "quant_baseline_available": True,
                    "predictor_max_prob_shift": settings.predictor_max_prob_shift,
                    "max_revisions": settings.max_revisions,
                },
            )
        )

    daily = [
        DailyPrediction(
            day_index=d["day_index"],
            target_date=trading_dates[d["day_index"] - 1],
            predicted_return_pct=d["predicted_return_pct"],
            p10_price=d["p10_price"],
            p50_price=d["p50_price"],
            p90_price=d["p90_price"],
            prob_up=d["prob_up"],
        )
        for d in (state.quant_daily_path or [])
        if 1 <= d["day_index"] <= horizon
    ]

    data_inputs = collect_data_inputs(state)
    forecast_id = new_forecast_id()
    return ForecastSnapshot(
        forecast_id=forecast_id,
        root_forecast_id=forecast_id,
        ticker=state.symbol,
        resolved_symbol=state.resolved_symbol,
        company_name=state.company_name or state.symbol,
        made_at=made_at,
        as_of_date=as_of_date,
        target_date=trading_dates[-1],
        horizon_trading_days=horizon,
        last_close=quant["last_close"],
        market_regime=regime,
        risk_category=risk,
        quant_baseline=_plain(quant),
        final_forecast=final,
        analyst_reports=_plain(reports),
        decisions=decisions,
        critic_verdict=CriticResult(**state.critic_result) if state.critic_result else None,
        daily_predictions=daily,
        data_snapshot_id=compute_data_snapshot_id(data_inputs),
        data_inputs=data_inputs,
        data_quality=_plain(
            {
                "guardrail_violations": state.guardrail_violations,
                "guardrail_warnings": state.guardrail_warnings,
                "data_gaps": [
                    {"analyst": name, **gap}
                    for name, rpt in reports.items()
                    for gap in _report_gaps(rpt)
                ],
                "warnings": _data_quality_warnings(as_of_date, trading_dates, made_at, calendar),
            }
        ),
        model_versions=get_model_versions(str(quant.get("method", "unknown"))),
        prompt_versions=get_prompt_versions(),
        calibration_version=calibration.version if calibration else UNCALIBRATED_VERSION,
        calibration=calibration,
        uncalibrated_baseline=_plain(state.quant_baseline_uncalibrated or quant),
    )


def _report_gaps(report: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalised data gaps of one analyst report (strings/dicts -> description + severity)."""
    if report is None:
        return []
    try:
        gaps = AnalystReport(**report).data_gaps
    except ValidationError:
        return [{"description": str(g), "severity": "unknown"} for g in report.get("data_gaps", [])]
    return [{"description": g.description, "severity": g.severity.value} for g in gaps]
