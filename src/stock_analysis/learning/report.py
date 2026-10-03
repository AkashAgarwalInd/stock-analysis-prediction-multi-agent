"""Deterministic report sections for the learning loop (Plan.md §29-30)."""

from __future__ import annotations

from typing import Optional

from stock_analysis.learning.lessons import LessonAction
from stock_analysis.schemas.learning import BenchmarkComparison, CalibrationUpdate, PostMortem


def render_postmortem(postmortem: PostMortem) -> str:
    """The "Why the forecast was wrong" section for one postmortem."""
    lines = [
        "## Postmortem",
        "",
        f"- Primary cause: **{postmortem.primary_cause.value}** "
        f"(confidence {postmortem.confidence}; {postmortem.method.value})",
        f"- {postmortem.explanation}",
        "",
        "Knowable at forecast time:",
    ]
    lines += [
        f"- {s.statement} ({', '.join(s.fact_ids)})" for s in postmortem.knowable_at_forecast_time
    ] or ["- (none identified)"]
    lines += ["", "Only known in hindsight:"]
    lines += [
        f"- {s.statement} ({', '.join(s.fact_ids)})" for s in postmortem.only_in_hindsight
    ] or ["- (none identified)"]
    lines += ["", "Candidate lessons:"]
    lines += [f"- [{c.scope}, {c.category}] {c.text}" for c in postmortem.lessons] or [
        "- None. A single forecast is not enough evidence to change the system."
        if not postmortem.expected_noise
        else "- None: the outcome was within expected noise."
    ]
    if postmortem.guard_notes:
        lines += ["", "Rejected or corrected by the deterministic checks:"]
        lines += [f"- {n}" for n in postmortem.guard_notes]
    lines.append("")
    return "\n".join(lines)


def render_lesson_actions(actions: list[LessonAction]) -> str:
    """One line per lesson lifecycle change."""
    if not actions:
        return ""
    lines = ["## Lessons", ""]
    for a in actions:
        note = f" — {a.note}" if a.note else ""
        lines.append(f"- `{a.lesson_id}` {a.action} → {a.status_after.value}{note}")
    lines.append("")
    return "\n".join(lines)


def render_adaptation(
    update: CalibrationUpdate, benchmarks: Optional[BenchmarkComparison] = None
) -> str:
    """Plan.md §30 "System adaptation" section for one ticker."""
    est = update.estimate
    lines = [
        f"## System adaptation: {update.ticker}",
        "",
        f"- Previous volatility multiplier: {update.previous_vol_multiplier:.2f} "
        f"(calibration v{update.previous_version})",
        f"- New volatility multiplier:      {est.vol_multiplier:.2f}",
        f"- Previous P50 bias shift: {update.previous_p50_bias_shift_pct:+.2f}%",
        f"- New P50 bias shift:      {est.p50_bias_shift_pct:+.2f}%",
        f"- Scored forecasts used: {est.n_samples}",
        (
            f"- Stored as calibration v{update.stored.version}; applies to forecasts made from now on"
            if update.stored
            else "- No material change; the current calibration stays in place"
        ),
        "",
        "Reason:",
        *(f"- {r}" for r in est.reason),
    ]
    if benchmarks is not None:
        lines += ["", "Does adaptation help?", f"- {benchmarks.finding}"]
        if benchmarks.n:
            lines.append(
                f"- Mean Brier loss over {benchmarks.n} scored forecasts: uncalibrated quant "
                f"{_loss(benchmarks.mean_uncalibrated_loss)}, calibrated quant "
                f"{_loss(benchmarks.mean_calibrated_loss)}, final forecast "
                f"{_loss(benchmarks.mean_final_loss)}"
            )
    lines.append("")
    return "\n".join(lines)


def _loss(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"
