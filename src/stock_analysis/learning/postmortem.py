"""Plan.md §18: postmortem of one scored forecast, with a hindsight guard.

The LLM runs only after the deterministic outcome metrics exist, and only when
the error is not ordinary noise. Every fact it sees is numbered and labelled
forecast-time (``F<n>``) or hindsight (``H<n>``). Its answer is then checked in
code:

* statements claimed as knowable at forecast time that cite hindsight facts are
  moved to hindsight; statements citing unknown facts are dropped;
* the primary cause must be consistent with the scored metrics, otherwise a
  deterministic classification replaces it;
* lessons must rest on forecast-time facts only and must not quote numbers that
  only hindsight facts contain; normal noise never yields a lesson.

The metrics themselves are never part of the LLM output, so they cannot change.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Optional

from stock_analysis.config.settings import get_settings
from stock_analysis.llm.models import LLMModel
from stock_analysis.logging import get_logger
from stock_analysis.schemas.learning import (
    CauseCategory,
    CitedStatement,
    Fact,
    FactTiming,
    HindsightNewsItem,
    PostMortem,
    PostmortemDiagnosis,
    PostmortemFacts,
    PostmortemMethod,
)
from stock_analysis.schemas.memory import LessonCandidate
from stock_analysis.schemas.outcome import ForecastOutcome, OutcomeStatus
from stock_analysis.schemas.snapshot import ForecastSnapshot
from stock_analysis.snapshots.builder import IST, NSE_CLOSE_IST
from stock_analysis.untrusted import sanitize_untrusted_text
from stock_analysis.versions import POSTMORTEM_PROMPT_FILE, PROMPTS_DIR, get_postmortem_version

logger = get_logger(__name__)

# Deterministic thresholds used to check (or replace) the LLM's cause
NOISE_VOL_RATIO_RANGE = (0.5, 2.0)
VOL_UNDERESTIMATED_RATIO = 1.2
VOL_OVERESTIMATED_RATIO = 0.8
MARKET_MOVE_MIN_PCT = 1.0
MAX_NEWS_FACT_CHARS = 300

_EVENT_SOURCES = frozenset({"corporate_action", "news"})
_DECIMAL_RE = re.compile(r"\d+\.\d+")


class PostmortemError(ValueError):
    """A postmortem was requested for a forecast that has not been scored."""


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


def _probs(d: dict[str, Any]) -> str:
    return (
        f"P(up) {d['prob_up']:.3f}, P(flat) {d['prob_flat']:.3f}, P(down) {d['prob_down']:.3f}; "
        f"P10/P50/P90 {d['p10_price']:.2f}/{d['p50_price']:.2f}/{d['p90_price']:.2f}; "
        f"expected return {d['expected_return_pct']:+.2f}%; "
        f"weekly volatility {d['weekly_vol_pct']:.2f}%"
    )


def _forecast_time_texts(snapshot: ForecastSnapshot) -> list[tuple[str, str]]:
    """(source, text) pairs recorded in the snapshot when the forecast was made."""
    final = snapshot.final_forecast
    out = [
        ("price", f"Last close on {snapshot.as_of_date}: {snapshot.last_close:.2f}"),
        (
            "final_forecast",
            f"Final forecast for {snapshot.target_date}: " + _probs(final.model_dump(mode="json")),
        ),
        ("quant_baseline", "Quant baseline used: " + _probs(snapshot.quant_baseline)),
    ]
    if snapshot.calibration is not None:
        cal = snapshot.calibration
        out.append(
            (
                "calibration",
                f"Calibration v{cal.version} applied: volatility multiplier "
                f"{cal.vol_multiplier:.2f}, P50 shift {cal.p50_bias_shift_pct:+.2f}%; "
                "uncalibrated baseline: " + _probs(snapshot.shadow_baseline),
            )
        )
    adjustments = "; ".join(
        f"{a.adjustment_type.value} {a.previous_value:.3f} -> {a.new_value:.3f} ({a.reason})"
        for a in final.adjustments
    )
    out.append(
        (
            "predictor",
            f"Adjustment applied: {'yes' if final.adjustment_applied else 'no'}; fell back to "
            f"quant: {'yes' if final.fallback_to_quant else 'no'}"
            + (f"; adjustments: {adjustments}" if adjustments else ""),
        )
    )
    for d in snapshot.decisions:
        out.append(
            (
                "decision_engine",
                f"{d.decision_type.value}: {d.decision} (confidence {d.confidence:.2f}) — "
                f"{d.rationale}",
            )
        )
    for name, report in snapshot.analyst_reports.items():
        if not report:
            out.append(("analyst_report", f"{name} analyst: no report"))
            continue
        points = "; ".join(report.get("key_points", [])[:3])
        risks = "; ".join(report.get("risks", [])[:2])
        out.append(
            (
                "analyst_report",
                f"{name} analyst: {report.get('stance')} (confidence "
                f"{float(report.get('confidence', 0.0)):.2f})"
                + (f". Key points: {points}" if points else "")
                + (f". Risks: {risks}" if risks else ""),
            )
        )
    quality = snapshot.data_quality
    for gap in quality.get("data_gaps", []):
        out.append(
            (
                "data_quality",
                f"Data gap ({gap.get('analyst')}, {gap.get('severity')}): {gap.get('description')}",
            )
        )
    out += [("data_quality", f"Data warning: {w}") for w in quality.get("warnings", [])]
    out += [
        ("data_quality", f"Guardrail violation: {v.get('message')}")
        for v in quality.get("guardrail_violations", [])
    ]
    memory = snapshot.data_inputs.get("memory") or {}
    out += [
        ("memory", f"Active lesson used: {lesson.get('text')}")
        for lesson in memory.get("active_lessons", [])
    ]
    return out


def _hindsight_texts(
    snapshot: ForecastSnapshot, outcome: ForecastOutcome, news: list[HindsightNewsItem]
) -> list[tuple[str, str]]:
    """(source, text) pairs that only became known after the forecast was made."""
    out = [
        (
            "outcome",
            f"Actual close on {outcome.target_date}: {outcome.actual_close_on_forecast_basis:.2f} "
            f"on the forecast's price basis; return {outcome.actual_return_pct:+.2f}%; "
            f"realized direction {outcome.realized_direction}",
        ),
        (
            "outcome",
            f"Predicted direction {outcome.predicted_direction} was "
            f"{'correct' if outcome.direction_correct else 'wrong'}; actual was "
            f"{'inside' if outcome.in_80pct_band else 'outside'} the final 80% band; "
            f"PIT {outcome.pit_percentile:.2f}",
        ),
        (
            "outcome",
            f"Brier loss: final {outcome.final_loss:.4f}, quant baseline "
            f"{outcome.baseline_loss:.4f}; LLM value added {outcome.llm_value_added:+.4f}",
        ),
    ]
    if outcome.vol_ratio is not None and outcome.realized_vol_pct is not None:
        out.append(
            (
                "outcome",
                f"Realized weekly volatility {outcome.realized_vol_pct:.2f}% vs predicted "
                f"{snapshot.final_forecast.weekly_vol_pct:.2f}% (ratio {outcome.vol_ratio:.2f})",
            )
        )
    if outcome.nifty_return_pct is not None:
        out.append(
            (
                "market",
                f"Nifty 50 return over the window {outcome.nifty_return_pct:+.2f}%; stock "
                f"excess vs Nifty {outcome.excess_vs_nifty_pct:+.2f}%",
            )
        )
    if outcome.daily:
        path = ", ".join(f"{d.date} {d.close_on_forecast_basis:.2f}" for d in outcome.daily)
        out.append(("outcome", f"Daily closes on the forecast's price basis: {path}"))
    if outcome.analyst_hits:
        calls = ", ".join(
            f"{name} {'correct' if hit else 'wrong'}" for name, hit in outcome.analyst_hits.items()
        )
        out.append(("analyst_hits", f"Analyst direction calls: {calls}"))
    for action in outcome.corporate_actions:
        detail = ", ".join(f"{k} {v}" for k, v in action.items() if k not in ("date", "type"))
        out.append(("corporate_action", f"{action.get('type')} on {action.get('date')}: {detail}"))
    out += [("data_quality", f"Evaluation check: {c}") for c in outcome.validity_checks]
    for item in news:
        # Untrusted web text (Plan.md §49): cleaned before it can reach the prompt
        headline, _ = sanitize_untrusted_text(item.headline, MAX_NEWS_FACT_CHARS)
        source, _ = sanitize_untrusted_text(item.source, 100)
        out.append(
            (
                "news",
                f"{item.published_at.astimezone(IST):%Y-%m-%d %H:%M} IST"
                + (f" ({source})" if source else "")
                + f": {headline}",
            )
        )
    return out


def window_news(
    snapshot: ForecastSnapshot, news: Iterable[HindsightNewsItem]
) -> list[HindsightNewsItem]:
    """News published after the forecast was made and by the target session's close.

    Earlier items are dropped: the snapshot is the record of what the forecast
    saw, so unrecorded earlier news cannot be presented as forecast-time evidence.
    """
    close = datetime.combine(snapshot.target_date, NSE_CLOSE_IST, tzinfo=IST)
    return sorted(
        (n for n in news if snapshot.made_at < n.published_at <= close),
        key=lambda n: n.published_at,
    )


def build_postmortem_facts(
    snapshot: ForecastSnapshot, outcome: ForecastOutcome, news: Iterable[HindsightNewsItem] = ()
) -> PostmortemFacts:
    """Number every fact and label it forecast-time (F) or hindsight (H)."""
    return PostmortemFacts(
        forecast_time=[
            Fact(id=f"F{i}", timing=FactTiming.FORECAST_TIME, source=src, text=text)
            for i, (src, text) in enumerate(_forecast_time_texts(snapshot), start=1)
        ],
        hindsight=[
            Fact(id=f"H{i}", timing=FactTiming.HINDSIGHT, source=src, text=text)
            for i, (src, text) in enumerate(
                _hindsight_texts(snapshot, outcome, window_news(snapshot, news)), start=1
            )
        ],
    )


def build_postmortem_prompt(
    snapshot: ForecastSnapshot, outcome: ForecastOutcome, facts: PostmortemFacts
) -> str:
    """The postmortem prompt: rules, then forecast-time and hindsight facts in separate sections."""
    sector = _sector(snapshot)
    lines = [
        (PROMPTS_DIR / POSTMORTEM_PROMPT_FILE).read_text().strip(),
        "",
        f"FORECAST: {snapshot.ticker} ({snapshot.company_name}), made {snapshot.made_at:%Y-%m-%d}, "
        f"as of {snapshot.as_of_date}, target {snapshot.target_date}; "
        f"sector: {sector or 'unknown'}; market regime {snapshot.market_regime.value}, "
        f"risk {snapshot.risk_category.value}",
        "",
        "FORECAST-TIME FACTS (knowable when the forecast was made):",
        *(f"  {f.id} [{f.source}] {f.text}" for f in facts.forecast_time),
        "",
        "HINDSIGHT FACTS (only known after the forecast was made; never lesson evidence):",
        *(f"  {f.id} [{f.source}] {f.text}" for f in facts.hindsight),
        "",
        "DETERMINISTIC CHECK: the outcome is NOT within expected noise (outside the 80% band "
        "or volatility far from the forecast)."
        if not is_expected_noise(outcome)
        else "DETERMINISTIC CHECK: the outcome is within expected noise.",
        "",
        "Return JSON matching the PostmortemDiagnosis schema.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------


def is_expected_noise(outcome: ForecastOutcome) -> bool:
    """Actual inside the 80% band with realised volatility in the normal range (Plan.md §19)."""
    if not outcome.in_80pct_band:
        return False
    lo, hi = NOISE_VOL_RATIO_RANGE
    return outcome.vol_ratio is None or lo <= outcome.vol_ratio <= hi


def _sign(value: Optional[float]) -> int:
    return 0 if value is None else (value > 0) - (value < 0)


def cause_rejection(
    cause: CauseCategory, outcome: ForecastOutcome, facts: PostmortemFacts, cited_ids: set[str]
) -> Optional[str]:
    """Why ``cause`` is inconsistent with the scored metrics and cited facts (None if supported)."""
    by_id = facts.by_id()
    cited_sources = {by_id[i].source for i in cited_ids if i in by_id}
    vol_ratio = outcome.vol_ratio
    if cause == CauseCategory.WITHIN_EXPECTED_NOISE:
        return None if outcome.in_80pct_band else "the actual close was outside the 80% band"
    if cause == CauseCategory.MARKET_WIDE_MOVE:
        nifty = outcome.nifty_return_pct
        if nifty is None:
            return "no Nifty 50 data for the window"
        if abs(nifty) < MARKET_MOVE_MIN_PCT:
            return f"Nifty 50 moved only {nifty:+.2f}%"
        if _sign(nifty) != _sign(outcome.signed_error_pct):
            return "the market moved against the direction of the forecast error"
        return None
    if cause == CauseCategory.EARNINGS_OR_CORPORATE_EVENT:
        return None if cited_sources & _EVENT_SOURCES else "no corporate-action or news fact cited"
    if cause == CauseCategory.REGULATORY_OR_NEWS_SHOCK:
        return None if "news" in cited_sources else "no news fact cited"
    if cause == CauseCategory.VOLATILITY_UNDERESTIMATED:
        if vol_ratio is None:
            return None if not outcome.in_80pct_band else "no volatility evidence"
        return (
            None
            if vol_ratio >= VOL_UNDERESTIMATED_RATIO
            else f"realised/predicted volatility was only {vol_ratio:.2f}x"
        )
    if cause == CauseCategory.VOLATILITY_OVERESTIMATED:
        if vol_ratio is None or vol_ratio > VOL_OVERESTIMATED_RATIO:
            return "realised volatility was not well below the forecast"
        return None
    if cause == CauseCategory.TREND_MISREAD:
        if outcome.direction_correct and outcome.in_80pct_band:
            return "direction was correct and the close was inside the 80% band"
        return None
    if cause == CauseCategory.ANALYST_ERROR:
        if not outcome.analyst_hits or all(outcome.analyst_hits.values()):
            return "no analyst made a wrong directional call"
        return None
    if cause == CauseCategory.DATA_ISSUE:
        return None if "data_quality" in cited_sources else "no data-quality fact cited"
    return f"unknown cause {cause}"


def classify_deterministically(
    outcome: ForecastOutcome, facts: PostmortemFacts
) -> tuple[CauseCategory, str]:
    """Rule-based cause, used for expected noise and when the LLM fails or is rejected."""
    all_ids = set(facts.by_id())
    if is_expected_noise(outcome):
        return CauseCategory.WITHIN_EXPECTED_NOISE, (
            "the actual close was inside the 80% band with normal volatility"
        )
    if cause_rejection(CauseCategory.MARKET_WIDE_MOVE, outcome, facts, all_ids) is None:
        return CauseCategory.MARKET_WIDE_MOVE, (
            f"Nifty 50 moved {outcome.nifty_return_pct:+.2f}% in the direction of the error"
        )
    if outcome.corporate_actions:
        return CauseCategory.EARNINGS_OR_CORPORATE_EVENT, "a corporate action fell in the window"
    vol_ratio = outcome.vol_ratio
    if (vol_ratio is not None and vol_ratio >= VOL_UNDERESTIMATED_RATIO) or (
        vol_ratio is None and not outcome.in_80pct_band
    ):
        return CauseCategory.VOLATILITY_UNDERESTIMATED, "realised volatility exceeded the forecast"
    if vol_ratio is not None and vol_ratio <= VOL_OVERESTIMATED_RATIO:
        return CauseCategory.VOLATILITY_OVERESTIMATED, "realised volatility was below the forecast"
    return CauseCategory.TREND_MISREAD, "the direction or size of the move was misjudged"


def _sector(snapshot: ForecastSnapshot) -> Optional[str]:
    sector = (snapshot.data_inputs.get("fundamentals") or {}).get("sector")
    return sector if isinstance(sector, str) and sector else None


def _hindsight_only_numbers(facts: PostmortemFacts) -> set[str]:
    forecast_time = {t for f in facts.forecast_time for t in _DECIMAL_RE.findall(f.text)}
    return {t for f in facts.hindsight for t in _DECIMAL_RE.findall(f.text)} - forecast_time


def _guard_statements(
    diagnosis: PostmortemDiagnosis, facts: PostmortemFacts, notes: list[str]
) -> tuple[list[CitedStatement], list[CitedStatement]]:
    """Apply the hindsight guard to the knowable / hindsight statement lists."""
    known_ids = set(facts.by_id())
    hindsight_ids = {f.id for f in facts.hindsight}
    leaked_numbers = _hindsight_only_numbers(facts)
    knowable: list[CitedStatement] = []
    hindsight: list[CitedStatement] = []
    for stmt in diagnosis.knowable_at_forecast_time:
        unknown = sorted(set(stmt.fact_ids) - known_ids)
        leaked = sorted(set(_DECIMAL_RE.findall(stmt.statement)) & leaked_numbers)
        if unknown:
            notes.append(f"Dropped statement citing unknown facts {unknown}: {stmt.statement}")
        elif set(stmt.fact_ids) & hindsight_ids:
            cited = sorted(set(stmt.fact_ids) & hindsight_ids)
            notes.append(f"Moved to hindsight (cites hindsight facts {cited}): {stmt.statement}")
            hindsight.append(stmt)
        elif leaked:
            notes.append(
                f"Moved to hindsight (quotes numbers known only in hindsight {leaked}): "
                f"{stmt.statement}"
            )
            hindsight.append(stmt)
        else:
            knowable.append(stmt)
    for stmt in diagnosis.only_in_hindsight:
        unknown = sorted(set(stmt.fact_ids) - known_ids)
        if unknown:
            notes.append(f"Dropped statement citing unknown facts {unknown}: {stmt.statement}")
        else:
            hindsight.append(stmt)
    return knowable, hindsight


def _guard_lessons(
    diagnosis: PostmortemDiagnosis,
    snapshot: ForecastSnapshot,
    facts: PostmortemFacts,
    notes: list[str],
) -> list[LessonCandidate]:
    """Keep only scoped lessons resting purely on forecast-time facts."""
    forecast_ids = {f.id for f in facts.forecast_time}
    hindsight_ids = {f.id for f in facts.hindsight}
    leaked_numbers = _hindsight_only_numbers(facts)
    sector = _sector(snapshot)
    accepted: list[LessonCandidate] = []
    seen: set[tuple[str, str]] = set()
    for lesson in diagnosis.lessons:
        refs = set(lesson.evidence_refs)
        reason = None
        if refs & hindsight_ids:
            reason = f"cites hindsight facts {sorted(refs & hindsight_ids)}"
        elif refs - forecast_ids:
            reason = f"cites unknown facts {sorted(refs - forecast_ids)}"
        elif leaked := sorted(set(_DECIMAL_RE.findall(lesson.text)) & leaked_numbers):
            reason = f"quotes numbers known only in hindsight {leaked}"
        elif lesson.scope == "sector" and sector is None:
            reason = "sector-scoped but the forecast's sector is unknown"
        elif (lesson.scope, lesson.category.value) in seen:
            reason = "duplicates another lesson of the same scope and category"
        if reason:
            notes.append(f"Rejected lesson ({reason}): {lesson.text}")
            continue
        seen.add((lesson.scope, lesson.category.value))
        accepted.append(
            LessonCandidate(
                text=lesson.text,
                scope=lesson.scope,
                category=lesson.category.value,
                ticker=snapshot.ticker if lesson.scope == "ticker" else None,
                sector=sector if lesson.scope == "sector" else None,
            )
        )
    return accepted


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _deterministic(
    snapshot: ForecastSnapshot,
    outcome: ForecastOutcome,
    facts: PostmortemFacts,
    *,
    now: datetime,
    method: PostmortemMethod,
    notes: list[str],
) -> PostMortem:
    cause, why = classify_deterministically(outcome, facts)
    noise = is_expected_noise(outcome)
    outcome_facts = [f.id for f in facts.hindsight if f.source == "outcome"]
    explanation = f"Deterministic classification: {why}."
    if noise:
        explanation += " This is normal statistical variation, so no lesson is drawn."
    return PostMortem(
        forecast_id=snapshot.forecast_id,
        ticker=snapshot.ticker,
        created_at=now,
        method=method,
        expected_noise=noise,
        primary_cause=cause,
        confidence="high" if noise else "low",
        explanation=explanation,
        only_in_hindsight=[CitedStatement(statement=why, fact_ids=outcome_facts or ["H1"])],
        guard_notes=notes,
        facts=facts,
        prompt_version=None,
        llm_model=None,
    )


def run_postmortem(
    snapshot: ForecastSnapshot,
    outcome: ForecastOutcome,
    *,
    llm_factory: Any = None,
    now: datetime,
    news: Iterable[HindsightNewsItem] = (),
) -> PostMortem:
    """Diagnose a scored forecast; never raises for LLM problems (falls back to rules).

    Raises:
        PostmortemError: the outcome is not a scored evaluation of ``snapshot``.
    """
    if outcome.status != OutcomeStatus.SCORED or outcome.forecast_id != snapshot.forecast_id:
        raise PostmortemError(f"Forecast {snapshot.forecast_id} has no scored outcome")
    facts = build_postmortem_facts(snapshot, outcome, news)
    if is_expected_noise(outcome):
        return _deterministic(
            snapshot, outcome, facts, now=now, method=PostmortemMethod.DETERMINISTIC, notes=[]
        )

    fallback = PostmortemMethod.DETERMINISTIC_FALLBACK
    if llm_factory is None:
        return _deterministic(
            snapshot, outcome, facts, now=now, method=fallback, notes=["No LLM configured"]
        )
    settings = get_settings()
    try:
        diagnosis = llm_factory.generate_structured(
            role=LLMModel.PRIMARY,
            prompt=build_postmortem_prompt(snapshot, outcome, facts),
            response_schema=PostmortemDiagnosis,
            temperature=settings.postmortem_temperature,
        )
    except Exception as err:  # any LLM/API/schema failure falls back to the rules
        logger.warning("postmortem_llm_failed", forecast_id=snapshot.forecast_id, error=str(err))
        note = f"LLM postmortem failed ({type(err).__name__}): {str(err)[:200]}"
        return _deterministic(snapshot, outcome, facts, now=now, method=fallback, notes=[note])

    notes: list[str] = []
    knowable, hindsight = _guard_statements(diagnosis, facts, notes)
    cited = {i for s in (*knowable, *hindsight) for i in s.fact_ids}
    rejection = cause_rejection(diagnosis.primary_cause, outcome, facts, cited)
    if rejection:
        notes.append(f"LLM cause {diagnosis.primary_cause.value} rejected: {rejection}")
        return _deterministic(snapshot, outcome, facts, now=now, method=fallback, notes=notes)

    lessons: list[LessonCandidate] = []
    if diagnosis.primary_cause == CauseCategory.WITHIN_EXPECTED_NOISE:
        if diagnosis.lessons:
            notes.append("Lessons dropped: the cause is within expected noise")
    else:
        lessons = _guard_lessons(diagnosis, snapshot, facts, notes)

    return PostMortem(
        forecast_id=snapshot.forecast_id,
        ticker=snapshot.ticker,
        created_at=now,
        method=PostmortemMethod.LLM,
        expected_noise=False,
        primary_cause=diagnosis.primary_cause,
        confidence=diagnosis.confidence,
        explanation=diagnosis.explanation,
        knowable_at_forecast_time=knowable,
        only_in_hindsight=hindsight,
        lessons=lessons,
        guard_notes=notes,
        facts=facts,
        prompt_version=get_postmortem_version(),
        llm_model=settings.gemini_model_primary,
    )
