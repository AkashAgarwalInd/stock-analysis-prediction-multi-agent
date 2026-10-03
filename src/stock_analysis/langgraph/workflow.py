from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from stock_analysis.config.settings import get_settings
from stock_analysis.database.forecast_store import ForecastSnapshotError, ForecastSnapshotStore
from stock_analysis.database.learning_store import LearningStore, LearningStoreError
from stock_analysis.database.memory_store import MemoryStore, MemoryStoreError
from stock_analysis.guardrails import run_preflight_guardrails
from stock_analysis.llm.factory import LLMModel, get_llm_factory
from stock_analysis.logging import get_logger
from stock_analysis.quant import QuantForecaster
from stock_analysis.schemas.analyst_reports import (
    AdjustmentGateDecision,
    AnalystReport,
    AnalystStance,
    AnalystType,
    DecisionResult,
    DecisionType,
    MarketRegime,
    RiskCategory,
    RulesDecisionEngine,
)
from stock_analysis.schemas.forecast_pipeline import (
    CriticCheckType,
    CriticFinding,
    CriticResult,
    FinalForecast,
    PredictorResult,
)
from stock_analysis.schemas.graph_state import GraphState
from stock_analysis.schemas.memory import ForecastInsight, MemoryContext
from stock_analysis.schemas.snapshot import PriceHistorySnapshot
from stock_analysis.snapshots import (
    SnapshotNotReadyError,
    build_forecast_snapshot,
    render_forecast_report,
)
from stock_analysis.snapshots.builder import IST
from stock_analysis.versions import (
    ANALYST_PROMPT_FILES,
    PROMPTS_DIR,
    QUANT_HISTORY_PERIOD,
    QUANT_N_PATHS,
    QUANT_SEED,
)

logger = get_logger(__name__)

# Tolerances used by the deterministic critic
_PROB_SUM_TOLERANCE = 0.001
_PROB_SHIFT_EPSILON = 1e-9
_OVERCONFIDENCE_THRESHOLD = 0.9

_ANALYST_REPORT_FIELDS = (
    "technical_report",
    "fundamental_report",
    "sentiment_report",
    "context_report",
)
_PROB_FIELDS = ("prob_up", "prob_flat", "prob_down")


def _max_prob_shift() -> float:
    """Maximum allowed probability shift from the quant baseline (Plan.md §9)."""
    return get_settings().predictor_max_prob_shift


def _max_revisions() -> int:
    """Maximum number of predictor revisions (Plan.md §10)."""
    return get_settings().max_revisions


def _collect_reports(state: GraphState) -> list[dict]:
    """Return the analyst reports that are present in state."""
    reports = []
    for field in _ANALYST_REPORT_FIELDS:
        rpt = getattr(state, field)
        if rpt:
            reports.append(rpt)
    return reports


def _dedupe(items: list[str]) -> list[str]:
    """De-duplicate while preserving order."""
    return list(dict.fromkeys(items))


# ---------------------------------------------------------------------------
# Guardrail node
# ---------------------------------------------------------------------------


def make_guardrail_node() -> Callable[[GraphState], dict]:
    """Pre-flight guardrail node that validates state before decision engine."""

    def guardrail_node(state: GraphState) -> dict:
        result = run_preflight_guardrails(state)

        if not result.passed:
            error_messages = [v.message for v in result.violations]

            # Return degraded decision indicating guardrail failure
            return {
                "decision": DecisionResult(
                    decision_type=DecisionType.MARKET_REGIME,
                    result=MarketRegime.MIXED.value,
                    confidence=0.0,
                    rationale=f"Guardrail violations: {'; '.join(error_messages)}",
                    model="guardrails",
                    model_version="0.1.0",
                ).model_dump(mode="json"),
                "guardrail_violations": [v.__dict__ for v in result.violations],
                "guardrail_warnings": [w.__dict__ for w in result.warnings],
            }

        return {
            "guardrail_violations": [],
            "guardrail_warnings": [w.__dict__ for w in result.warnings],
        }

    return guardrail_node


# ---------------------------------------------------------------------------
# Analyst node factory
# ---------------------------------------------------------------------------


def make_analyst_node(
    name: AnalystType,
    prompt_path: str,
    llm_factory,
) -> Callable[[GraphState], dict]:
    """Factory that creates an analyst node function for the LangGraph.

    The node:
    1. Reads structured collector summaries from GraphState
    2. Formats the prompt with available data
    3. Calls LLM for structured output via the factory
    4. Returns a single ``<analyst>_report`` dict (Pydantic-validated)

    Raises ``FileNotFoundError`` at build time if the prompt file is missing.
    """
    instructions = Path(prompt_path).read_text()

    def analyst_node(state: GraphState) -> dict:
        # Compact summaries only; never large datasets
        if name == AnalystType.TECHNICAL:
            prompt_data = {
                "symbol": state.symbol,
                "indicators": state.technical_indicators_summary or {},
            }
        elif name == AnalystType.FUNDAMENTAL:
            prompt_data = {
                "symbol": state.symbol,
                "fundamentals": state.fundamentals_summary or {},
            }
        elif name == AnalystType.SENTIMENT:
            prompt_data = {"symbol": state.symbol, "news": state.news_summary or {}}
        else:
            prompt_data = {
                "symbol": state.symbol,
                "market_context": state.market_context_summary or {},
            }

        prompt = f"{instructions}\n\nINPUT DATA:\n{json.dumps(prompt_data, default=str)}"

        try:
            factory = get_llm_factory() if llm_factory is None else llm_factory
            response = factory.generate_structured(
                role=LLMModel.PRIMARY,
                prompt=prompt,
                response_schema=AnalystReport,
            )
            report_dict = response.model_dump(mode="json")
        except Exception as err:
            # Graceful degradation – return a clearly marked degraded result
            logger.warning("analyst_failed", analyst=name.value, error=str(err))
            report_dict = {
                "analyst": name.value,
                "stance": AnalystStance.NEUTRAL.value,
                "confidence": 0.0,
                "key_points": [f"Analysis failed: {err}"[:200]],
                "evidence": [],
                "risks": ["LLM call failed"],
                "data_gaps": ["LLM unavailable"],
            }

        return {f"{name.value}_report": report_dict}

    return analyst_node


# ---------------------------------------------------------------------------
# Decision Engine node
# ---------------------------------------------------------------------------


def make_decision_engine_node() -> Callable[[GraphState], dict]:
    """Decision Engine node that classifies the market regime from analyst reports.

    Delegates to the deterministic ``RulesDecisionEngine``. If the guardrails
    already produced a degraded decision, it is preserved.
    """

    engine = RulesDecisionEngine()

    def decide_node(state: GraphState) -> dict:
        if state.guardrail_violations and state.decision is not None:
            risk = DecisionResult(
                decision_type=DecisionType.RISK_CATEGORY,
                result=RiskCategory.HIGH.value,
                confidence=1.0,
                rationale="Guardrail violations: analyst evidence is unusable",
                model="guardrails",
            )
            return {
                "decision": state.decision.model_dump(mode="json"),
                "risk_decision": risk.model_dump(mode="json"),
            }

        context = {"analyst_reports": _collect_reports(state)}
        result = engine.decide(DecisionType.MARKET_REGIME, context)
        risk = engine.decide(DecisionType.RISK_CATEGORY, context)
        return {
            "decision": result.model_dump(mode="json"),
            "risk_decision": risk.model_dump(mode="json"),
        }

    return decide_node


# ---------------------------------------------------------------------------
# Quant baseline node
# ---------------------------------------------------------------------------


def _load_calibration(
    calibration_store: Optional[LearningStore], symbol: str
) -> Optional[dict]:
    """The ticker's latest calibration as applied params, or None (never blocks a forecast)."""
    if calibration_store is None or not get_settings().learning_enabled:
        return None
    try:
        params = calibration_store.latest_calibration(symbol, as_of=datetime.now(UTC))
    except (LearningStoreError, sqlite3.Error, ValidationError) as err:
        logger.warning("calibration_load_failed", symbol=symbol, error=str(err))
        return None
    if params is None:
        return None
    return {
        "version": params.version,
        "vol_multiplier": params.vol_multiplier,
        "p50_bias_shift_pct": params.p50_bias_shift_pct,
    }


def make_quant_baseline_node(
    price_store: Optional[ForecastSnapshotStore] = None,
    calibration_store: Optional[LearningStore] = None,
) -> Callable[[GraphState], dict]:
    """Quant baseline node; with ``price_store`` it also persists the exact close series.

    The stored series (content-addressed by ``history_sha256``) is what lets a
    snapshot's baseline be reproduced later (``snapshots.reproduce``). With a
    ``calibration_store`` the ticker's latest calibration is applied (unless
    ``LEARNING_ENABLED`` is false); the uncalibrated baseline is always kept as
    the shadow forecast.
    """

    def quant_baseline_node(state: GraphState) -> dict:
        """Generate the Phase 4 quant baseline from real price history."""
        from stock_analysis.market.collector import get_price_collector

        def _failed(reason: str) -> dict:
            return {"quant_baseline": {"error": reason, "source": "quant_baseline_node_failed"}}

        try:
            price_history = get_price_collector().fetch_history(
                state.resolved_symbol, period=QUANT_HISTORY_PERIOD, interval="1d"
            )
        except Exception as err:
            logger.warning(
                "quant_price_fetch_failed", symbol=state.resolved_symbol, error=str(err)
            )
            return _failed(f"Price fetch failed for {state.resolved_symbol}: {err}")

        points = [d for d in (price_history.data or []) if d.close is not None]
        close_prices = np.array([float(d.close) for d in points])
        if len(close_prices) < 20:
            return _failed(f"Insufficient valid close prices for {state.resolved_symbol}")

        forecaster = QuantForecaster(
            seed=QUANT_SEED, n_paths=QUANT_N_PATHS, horizon=get_settings().forecast_horizon_days
        )
        calibration = _load_calibration(calibration_store, state.symbol)
        try:
            uncalibrated, daily_path = forecaster.forecast_with_daily_path(close_prices)
            baseline = uncalibrated
            if calibration is not None:
                # Same seed, same draws: the two baselines differ only by the calibration
                baseline, daily_path = forecaster.forecast_with_daily_path(
                    close_prices, calibration
                )
        except ValueError as err:
            return _failed(str(err))

        history = PriceHistorySnapshot.from_points(state.resolved_symbol, points)
        if price_store is not None:
            price_store.save_price_history(history)
        return {
            "quant_baseline": asdict(baseline),
            "quant_baseline_uncalibrated": asdict(uncalibrated),
            "calibration": calibration,
            "quant_daily_path": [asdict(d) for d in daily_path],
            "price_snapshot": history.fingerprint(),
        }

    return quant_baseline_node


quant_baseline_node = make_quant_baseline_node()


# ---------------------------------------------------------------------------
# Adjustment gate node
# ---------------------------------------------------------------------------


def adjustment_gate_node(state: GraphState) -> dict:
    """Decide whether the predictor may adjust the quant baseline."""
    if state.guardrail_violations:
        result = DecisionResult(
            decision_type=DecisionType.FORECAST_ADJUSTMENT_GATE,
            result=AdjustmentGateDecision.NO_ADJUSTMENT.value,
            confidence=1.0,
            rationale="Guardrail violations block adjustment",
            model="guardrails",
        )
    else:
        result = RulesDecisionEngine().decide(
            DecisionType.FORECAST_ADJUSTMENT_GATE,
            {
                "analyst_reports": _collect_reports(state),
                "quant_baseline": state.quant_baseline,
            },
        )
    return {"adjustment_gate_decision": result.model_dump(mode="json")}


# ---------------------------------------------------------------------------
# Predictor node
# ---------------------------------------------------------------------------


def build_predictor_prompt(
    quant_baseline: dict,
    analyst_reports: list[dict],
    evidence: list[str],
    key_points: list[str],
    risks: list[str],
    revision_guidance: Optional[str] = None,
    prior_context: Optional[dict] = None,
) -> str:
    """Build the constrained predictor prompt.

    Numbers come from the quant baseline; the LLM may only propose bounded,
    evidence-backed adjustments to them. ``prior_context`` (from the memory
    loader) adds earlier forecasts, active lessons and the track record.
    """
    max_shift = _max_prob_shift()
    lines = [
        "You are a forecast predictor for an Indian NSE stock (educational use only).",
        "Start from the QUANT BASELINE and apply ONLY small, evidence-backed adjustments.",
        "Do not invent facts, override deterministic data, or narrow uncertainty dramatically.",
        "Describe evidence and probabilities only; never phrase output as a buy, sell or hold "
        "recommendation, price target or trading instruction.",
        "",
        "QUANT BASELINE:",
        json.dumps(quant_baseline, indent=2, default=str),
        "",
    ]

    for rpt in analyst_reports:
        lines.append(
            f"{str(rpt.get('analyst', 'unknown')).upper()} ANALYST "
            f"(stance: {str(rpt.get('stance', 'unknown')).upper()}, "
            f"confidence: {rpt.get('confidence')})"
        )
        for label, key in (("Key points", "key_points"), ("Evidence", "evidence"), ("Risks", "risks")):
            for item in rpt.get(key, []):
                lines.append(f"  {label}: {item}")
        for gap in rpt.get("data_gaps", []):
            desc = gap.get("description") if isinstance(gap, dict) else gap
            lines.append(f"  Data gap: {desc}")
        lines.append("")

    if evidence:
        lines.append("EVIDENCE INDEX (cite these in evidence_refs):")
        lines.extend(f"  - {e}" for e in evidence)
        lines.append("")
    if key_points:
        lines.append("KEY POINTS:")
        lines.extend(f"  - {k}" for k in key_points)
        lines.append("")
    if risks:
        lines.append("RISKS:")
        lines.extend(f"  - {r}" for r in risks)
        lines.append("")

    lines += format_prior_context(prior_context)

    lines += [
        "CONSTRAINTS:",
        f"- MAX_PROB_SHIFT = {max_shift} (per-probability change vs the baseline)",
        "- prob_up + prob_flat + prob_down MUST equal 1.0",
        "- P10 < P50 < P90",
        "- Every adjustment needs a reason and evidence_refs",
        "- If no adjustment is justified, return the baseline with adjustment_applied=false",
        "",
        "Return JSON matching the PredictorResult schema.",
    ]

    if revision_guidance:
        lines += ["", "REVISION REQUIRED - fix these critic findings:", revision_guidance]

    return "\n".join(lines)


def format_prior_context(prior_context: Optional[dict]) -> list[str]:
    """Prompt lines for memory context; empty when there is nothing to show."""
    if not prior_context:
        return []
    ctx = MemoryContext.model_validate(prior_context)
    track = ctx.track_record
    if not ctx.has_content and not (track and track.forecasts_made):
        return []

    lines = [
        "PRIOR CONTEXT (from earlier runs; background only, not evidence about this week):"
    ]
    for f in ctx.prior_forecasts:
        lines.append(
            f"  Previous forecast as of {f.as_of_date} for {f.target_date}: regime {f.market_regime}, "
            f"P(up) {f.prob_up:.3f}, P(down) {f.prob_down:.3f}, "
            f"P10/P50/P90 {f.p10_price:.2f}/{f.p50_price:.2f}/{f.p90_price:.2f}, "
            f"adjustment applied: {'yes' if f.adjustment_applied else 'no'}"
        )
    if ctx.active_lessons:
        lines.append("  Active lessons (cite as evidence_refs 'lesson:<id>' only if relevant):")
        lines.extend(
            f"    - lesson:{lesson.lesson_id} [{lesson.scope}, {lesson.evidence_count} confirmations] "
            f"{lesson.text}"
            for lesson in ctx.active_lessons
        )
    if track:
        lines.append(
            f"  Track record: earlier forecasts {track.forecasts_made}; "
            f"scored against actual prices {track.forecasts_scored}; "
            f"awaiting evaluation {track.forecasts_awaiting_outcome}."
        )
    lines += [
        "  Do not treat a previous forecast as evidence; it only shows what was predicted before.",
        "",
    ]
    return lines


def _quant_as_predictor_result(quant: dict, reasoning: str) -> dict:
    """Express the quant baseline as an unadjusted ``PredictorResult`` dict."""
    return PredictorResult(
        prob_up=quant["prob_up"],
        prob_flat=quant["prob_flat"],
        prob_down=quant["prob_down"],
        expected_return_pct=quant["expected_return_pct"],
        p10_price=quant["p10_price"],
        p50_price=quant["p50_price"],
        p90_price=quant["p90_price"],
        weekly_vol_pct=quant["weekly_vol_pct"],
        adjustment_applied=False,
        adjustments=[],
        reasoning=reasoning[:2000],
        evidence=[],
        risks=[],
        quant_baseline_ref=quant,
    ).model_dump(mode="json")


def make_predictor_node(llm_factory=None) -> Callable[[GraphState], dict]:
    """Predictor node that applies bounded adjustments to the quant baseline."""

    def predictor_node(state: GraphState) -> dict:
        quant = state.quant_baseline
        if not quant or quant.get("error"):
            return {"predictor_result": {"error": "No valid quant baseline"}}

        gate = state.adjustment_gate_decision
        if gate is None or gate.result != AdjustmentGateDecision.ALLOW_ADJUSTMENT.value:
            reason = gate.rationale if gate else "gate not evaluated"
            return {
                "predictor_result": _quant_as_predictor_result(
                    quant, f"No adjustment allowed by gate: {reason}"
                )
            }

        reports = _collect_reports(state)
        critic = state.critic_result
        guidance = critic.get("revision_guidance") if critic and not critic.get("passed") else None
        prompt = build_predictor_prompt(
            quant_baseline=quant,
            analyst_reports=reports,
            evidence=_dedupe([e for r in reports for e in r.get("evidence", [])]),
            key_points=_dedupe([k for r in reports for k in r.get("key_points", [])]),
            risks=_dedupe([x for r in reports for x in r.get("risks", [])]),
            revision_guidance=guidance,
            prior_context=state.memory_context,
        )

        try:
            factory = get_llm_factory() if llm_factory is None else llm_factory
            response = factory.generate_structured(
                role=LLMModel.PRIMARY,
                prompt=prompt,
                response_schema=PredictorResult,
            )
            # Re-validate against the real baseline so the drift bound cannot be
            # bypassed by an LLM that omits or alters quant_baseline_ref.
            predictor_result = PredictorResult(
                **{**response.model_dump(), "quant_baseline_ref": quant}
            )
        except ValidationError as err:
            # Output violated schema bounds: surface it so the critic can drive a revision
            logger.warning("predictor_invalid_output", error=str(err))
            return {"predictor_result": {"error": f"Invalid predictor output: {err}"}}
        except Exception as err:
            # LLM unavailable: revising will not help, fall back to the baseline
            logger.warning("predictor_llm_failed", error=str(err))
            return {
                "predictor_result": _quant_as_predictor_result(
                    quant, f"LLM adjustment failed, falling back to quant baseline: {err}"
                )
            }

        return {"predictor_result": predictor_result.model_dump(mode="json")}

    return predictor_node


# ---------------------------------------------------------------------------
# Critic / revision / final forecast
# ---------------------------------------------------------------------------


def critic_node(state: GraphState) -> dict:
    """Deterministic critic over the raw predictor output.

    Works on the raw dict (not ``PredictorResult``) so that it can judge output
    that would fail strict schema validation.
    """
    pred = state.predictor_result or {}
    quant = state.quant_baseline or {}
    findings: list[CriticFinding] = []

    def add(check: CriticCheckType, message: str, severity: str = "error") -> None:
        findings.append(
            CriticFinding(
                check_type=check, passed=False, message=message[:500], severity=severity
            )
        )

    if not pred or "error" in pred:
        add(
            CriticCheckType.DATA_QUALITY,
            f"Predictor produced no valid forecast: {pred.get('error', 'missing')}",
        )
    else:
        probs = {f: pred.get(f) for f in _PROB_FIELDS}

        valid_probs = True
        for name, val in probs.items():
            if not isinstance(val, (int, float)) or not 0.0 <= val <= 1.0:
                valid_probs = False
                add(CriticCheckType.PROBABILITY_VALIDITY, f"{name}={val} is outside [0, 1]")

        if valid_probs:
            total = sum(probs.values())
            if abs(total - 1.0) > _PROB_SUM_TOLERANCE:
                add(CriticCheckType.PROBABILITY_SUM, f"Probabilities sum to {total:.4f}, expected 1.0")

        p10, p50, p90 = (pred.get(k) for k in ("p10_price", "p50_price", "p90_price"))
        if not all(isinstance(p, (int, float)) for p in (p10, p50, p90)) or not p10 < p50 < p90:
            add(CriticCheckType.PRICE_ORDERING, f"Require P10 < P50 < P90, got {p10}, {p50}, {p90}")

        max_shift = _max_prob_shift()
        for name in _PROB_FIELDS:
            base, new = quant.get(name), probs[name]
            if isinstance(base, (int, float)) and isinstance(new, (int, float)):
                shift = abs(new - base)
                if shift > max_shift + _PROB_SHIFT_EPSILON:
                    add(
                        CriticCheckType.EXCESSIVE_ADJUSTMENT,
                        f"{name} shifted {shift:.3f} from baseline, exceeds MAX_PROB_SHIFT={max_shift}",
                    )

        adjustments = pred.get("adjustments") or []
        if pred.get("adjustment_applied") and not adjustments:
            add(
                CriticCheckType.UNSUPPORTED_CLAIMS,
                "adjustment_applied=true but no adjustments were listed",
                severity="warning",
            )
        for adj in adjustments:
            if not adj.get("evidence_refs"):
                add(
                    CriticCheckType.EVIDENCE_LINKAGE,
                    f"Adjustment to {adj.get('adjustment_type')} has no evidence references",
                    severity="warning",
                )

        if valid_probs and max(probs.values()) >= _OVERCONFIDENCE_THRESHOLD:
            add(
                CriticCheckType.OVERCONFIDENCE,
                f"Extreme probability (>= {_OVERCONFIDENCE_THRESHOLD}) for a 5-day forecast",
                severity="warning",
            )

    errors = [f.message for f in findings if f.severity == "error"]
    result = CriticResult(
        passed=not errors,
        findings=findings,
        revision_guidance="; ".join(errors)[:1000] if errors else None,
    )
    return {"critic_result": result.model_dump(mode="json")}


def revision_node(state: GraphState) -> dict:
    """Count revisions; once the budget is spent, fall back to the quant baseline."""
    count = state.forecast_revision_count
    critic = state.critic_result or {}
    if critic.get("passed", False):
        return {"forecast_revision_count": count}

    if count < _max_revisions():
        return {"forecast_revision_count": count + 1}

    update: dict = {"forecast_revision_count": count + 1}
    quant = state.quant_baseline
    if quant and not quant.get("error"):
        update["predictor_result"] = _quant_as_predictor_result(
            quant, f"Max revisions ({_max_revisions()}) reached; using quant baseline"
        )
    return update


def route_after_revision(state: GraphState) -> Literal["predictor", "final_forecast"]:
    """Loop back to the predictor until the critic passes or revisions are exhausted."""
    critic = state.critic_result or {}
    if critic.get("passed", False) or state.forecast_revision_count > _max_revisions():
        return "final_forecast"
    return "predictor"


def final_forecast_node(state: GraphState) -> dict:
    """Consolidate the predictor output (or quant fallback) into the final forecast."""
    quant = state.quant_baseline or {}
    critic = state.critic_result
    critic_passed = bool(critic.get("passed", False)) if critic else False

    predictor: Optional[PredictorResult] = None
    pred_raw = state.predictor_result
    if pred_raw and "error" not in pred_raw:
        try:
            predictor = PredictorResult(**pred_raw)
        except ValidationError as err:
            logger.warning("final_forecast_invalid_predictor", error=str(err))

    revisions_exhausted = (
        not critic_passed and state.forecast_revision_count > _max_revisions()
    )
    fallback = predictor is None or revisions_exhausted

    if predictor is None:
        if not quant or quant.get("error"):
            return {"final_forecast": {"error": "No valid forecast: predictor and quant baseline unavailable"}}
        predictor = PredictorResult(
            **_quant_as_predictor_result(quant, "Predictor output unavailable; using quant baseline")
        )

    reports = _collect_reports(state)
    final = FinalForecast(
        symbol=state.symbol,
        forecast_date=date.today().isoformat(),
        horizon_trading_days=quant.get("horizon_trading_days", get_settings().forecast_horizon_days),
        prob_up=predictor.prob_up,
        prob_flat=predictor.prob_flat,
        prob_down=predictor.prob_down,
        expected_return_pct=predictor.expected_return_pct,
        p10_price=predictor.p10_price,
        p50_price=predictor.p50_price,
        p90_price=predictor.p90_price,
        weekly_vol_pct=predictor.weekly_vol_pct,
        adjustment_applied=predictor.adjustment_applied,
        adjustments=predictor.adjustments,
        quant_baseline=quant,
        predictor_reasoning=predictor.reasoning,
        critic_passed=critic_passed,
        revision_count=state.forecast_revision_count,
        fallback_to_quant=fallback,
        evidence=_dedupe(predictor.evidence + [e for r in reports for e in r.get("evidence", [])]),
        risks=_dedupe(predictor.risks + [x for r in reports for x in r.get("risks", [])]),
    )
    return {"final_forecast": final.model_dump(mode="json")}


# ---------------------------------------------------------------------------
# Final join/fusion node – exported for testing
# ---------------------------------------------------------------------------


def join_node(state: GraphState) -> dict:
    """Compile the final report from all assembled pieces."""
    reports = {
        "technical": state.technical_report,
        "fundamental": state.fundamental_report,
        "sentiment": state.sentiment_report,
        "context": state.context_report,
    }

    stance_counts: dict[str, int] = {}
    data_gaps: list = []
    for rpt_dict in reports.values():
        if rpt_dict is None:
            continue
        data_gaps.extend(rpt_dict.get("data_gaps", []))
        try:
            rpt = AnalystReport(**rpt_dict)
        except ValidationError:
            continue
        stance_counts[rpt.stance.value] = stance_counts.get(rpt.stance.value, 0) + 1

    final_report = {
        "symbol": state.symbol,
        "analyst_stances": stance_counts,
        "decision": state.decision.model_dump(mode="json") if state.decision else None,
        "quant_baseline": state.quant_baseline,
        "final_forecast": state.final_forecast,
        "data_gaps": data_gaps,
    }

    return {"final_report": final_report}


# ---------------------------------------------------------------------------
# Forecast snapshot node (Phase 7)
# ---------------------------------------------------------------------------


def make_snapshot_node(
    snapshot_store: Optional[ForecastSnapshotStore] = None,
) -> Callable[[GraphState], dict]:
    """Snapshot the completed forecast, persist it if a store is given, render the report.

    Without a store the snapshot is still built and validated (and the report
    rendered), but ``snapshot_persisted`` stays False.
    """

    def snapshot_node(state: GraphState) -> dict:
        try:
            snapshot = build_forecast_snapshot(state)
        except SnapshotNotReadyError as err:
            logger.warning("forecast_snapshot_skipped", symbol=state.symbol, reason=str(err))
            return {}

        report = render_forecast_report(snapshot)
        if snapshot_store is not None:
            snapshot_store.save(snapshot)
        return {
            "forecast_id": snapshot.forecast_id,
            "data_snapshot_id": snapshot.data_snapshot_id,
            "snapshot_persisted": snapshot_store is not None,
            "forecast_report": report,
        }

    return snapshot_node


# ---------------------------------------------------------------------------
# Memory nodes (Plan.md Phase 7)
# ---------------------------------------------------------------------------


def make_memory_loader_node(
    memory_store: Optional[MemoryStore] = None,
) -> Callable[[GraphState], dict]:
    """Load prior insights, active lessons and the track record for the ticker.

    Without a store memory is not configured and ``memory_context`` stays None.
    A failing store never blocks the forecast: the run continues with an
    empty context that records the error.
    """

    def memory_loader_node(state: GraphState) -> dict:
        if memory_store is None:
            return {}
        settings = get_settings()
        now = datetime.now(UTC)
        sector = (state.fundamentals_summary or {}).get("sector")
        try:
            context = MemoryContext(
                loaded=True,
                prior_forecasts=memory_store.recent_insights(
                    state.symbol, before=now, limit=settings.memory_max_prior_forecasts
                ),
                active_lessons=memory_store.active_lessons(
                    state.symbol,
                    sector if isinstance(sector, str) else None,
                    as_of=now,
                    limit=settings.max_active_lessons_in_prompt,
                ),
                track_record=memory_store.track_record(
                    state.symbol, before=now, as_of_date=now.astimezone(IST).date()
                ),
            )
        except (MemoryStoreError, sqlite3.Error, ValidationError) as err:
            logger.warning("memory_load_failed", symbol=state.symbol, error=str(err))
            context = MemoryContext(loaded=False, error=str(err)[:500])
        return {"memory_context": context.model_dump(mode="json")}

    return memory_loader_node


def make_memory_writer_node(
    snapshot_store: Optional[ForecastSnapshotStore] = None,
    memory_store: Optional[MemoryStore] = None,
) -> Callable[[GraphState], dict]:
    """Store a distilled insight for the persisted forecast so later runs can see it.

    Memory is auxiliary: a failed write is logged and reported in
    ``memory_written`` but never undoes the already-stored forecast.
    """

    def memory_writer_node(state: GraphState) -> dict:
        if (
            memory_store is None
            or snapshot_store is None
            or not state.snapshot_persisted
            or state.forecast_id is None
        ):
            return {}
        try:
            snapshot = snapshot_store.get(state.forecast_id)
            if snapshot is None:
                raise MemoryStoreError(f"Forecast {state.forecast_id} is not stored")
            memory_store.save_insight(ForecastInsight.from_snapshot(snapshot))
        except (MemoryStoreError, ForecastSnapshotError, sqlite3.Error) as err:
            logger.warning("memory_write_failed", forecast_id=state.forecast_id, error=str(err))
            return {"memory_written": False}
        return {"memory_written": True}

    return memory_writer_node


# ---------------------------------------------------------------------------
# LangGraph workflow construction
# ---------------------------------------------------------------------------


def build_workflow(
    llm_factory=None,
    snapshot_store: Optional[ForecastSnapshotStore] = None,
    memory_store: Optional[MemoryStore] = None,
    learning_store: Optional[LearningStore] = None,
) -> StateGraph:
    """Build the LangGraph workflow.

    Topology:
        memory_loader (prior context when ``memory_store`` is given)
          → mark_collectors
          → parallel analysts (technical, fundamental, sentiment, context)
          → guardrails
          → decision_engine
          → quant_baseline (calibrated when ``learning_store`` holds a calibration;
                            the uncalibrated shadow baseline is always kept)
          → adjustment_gate
          → predictor → critic → revision ─┬→ predictor (revise, bounded)
                                           └→ final_forecast
          → join
          → forecast_snapshot (persisted when ``snapshot_store`` is given)
          → memory_writer (insight stored when both stores are given) → END
    """
    factory = get_llm_factory() if llm_factory is None else llm_factory

    workflow = StateGraph(GraphState)

    workflow.add_node("memory_loader", make_memory_loader_node(memory_store))
    workflow.set_entry_point("memory_loader")

    # Collectors are pre-loaded into state; this node only marks them complete
    workflow.add_node("mark_collectors", lambda state: {"collectors_complete": True})
    workflow.add_edge("memory_loader", "mark_collectors")

    analysts = {
        "technical_analyst": AnalystType.TECHNICAL,
        "fundamental_analyst": AnalystType.FUNDAMENTAL,
        "sentiment_analyst": AnalystType.SENTIMENT,
        "context_analyst": AnalystType.CONTEXT,
    }
    for node_name, analyst_type in analysts.items():
        prompt_path = PROMPTS_DIR / ANALYST_PROMPT_FILES[node_name]
        workflow.add_node(
            node_name,
            make_analyst_node(analyst_type, str(prompt_path), factory),
        )
        workflow.add_edge("mark_collectors", node_name)
        workflow.add_edge(node_name, "guardrails")

    workflow.add_node("guardrails", make_guardrail_node())
    workflow.add_node("decision_engine", make_decision_engine_node())
    workflow.add_node("quant_baseline", make_quant_baseline_node(snapshot_store, learning_store))
    workflow.add_node("adjustment_gate", adjustment_gate_node)
    workflow.add_node("predictor", make_predictor_node(factory))
    workflow.add_node("critic", critic_node)
    workflow.add_node("revision", revision_node)
    workflow.add_node("final_forecast", final_forecast_node)
    workflow.add_node("join", join_node)
    workflow.add_node("forecast_snapshot", make_snapshot_node(snapshot_store))
    workflow.add_node("memory_writer", make_memory_writer_node(snapshot_store, memory_store))

    workflow.add_edge("guardrails", "decision_engine")
    workflow.add_edge("decision_engine", "quant_baseline")
    workflow.add_edge("quant_baseline", "adjustment_gate")
    workflow.add_edge("adjustment_gate", "predictor")
    workflow.add_edge("predictor", "critic")
    workflow.add_edge("critic", "revision")
    workflow.add_conditional_edges(
        "revision",
        route_after_revision,
        {"predictor": "predictor", "final_forecast": "final_forecast"},
    )
    workflow.add_edge("final_forecast", "join")
    workflow.add_edge("join", "forecast_snapshot")
    workflow.add_edge("forecast_snapshot", "memory_writer")
    workflow.add_edge("memory_writer", END)

    return workflow


# ---------------------------------------------------------------------------
# Entry point for graph compilation
# ---------------------------------------------------------------------------


def compile_graph(
    llm_factory=None,
    snapshot_store: Optional[ForecastSnapshotStore] = None,
    memory_store: Optional[MemoryStore] = None,
    learning_store: Optional[LearningStore] = None,
):
    """Compile the LangGraph workflow.

    Snapshots, memory and calibration are used only when their stores are
    given; production callers should use ``langgraph.runner.run_forecast``,
    which always wires all three.
    """
    return build_workflow(llm_factory, snapshot_store, memory_store, learning_store).compile()
