"""Deterministic report sections for the learning loop (Plan.md §29-30)."""

from __future__ import annotations

from typing import Optional

from stock_analysis.learning.lessons import LessonAction
from stock_analysis.schemas.learning import BenchmarkComparison, CalibrationUpdate, PostMortem
from stock_analysis.schemas.scorecard import GroupScore, HitRate, Scorecards


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


def _rate(h: HitRate) -> str:
    if h.rate is None:
        return "no calls"
    return f"{h.hits}/{h.n} ({h.rate:.0%}, 95% CI {h.ci_low:.0%}–{h.ci_high:.0%})"


def _breakdown(label: str, groups: dict[str, HitRate]) -> str:
    parts = "; ".join(f"{name} {h.hits}/{h.n}" for name, h in groups.items())
    return f"  - {label}: {parts or 'none'}"


def _group_table(title: str, scores: list[GroupScore]) -> list[str]:
    if not scores:
        return []
    lines = [
        f"### By {title}",
        "",
        "| Group | n | Direction | MAE | Mean signed error | Brier | 80% coverage |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {s.group} | {s.n} | {_rate(s.direction)} | {s.mean_abs_error_pct:.2f}% "
        f"| {s.mean_signed_error_pct:+.2f}% | {s.brier:.4f} | {s.coverage_80pct:.0%} |"
        for s in scores
    ]
    return [*lines, ""]


def render_scorecards(cards: Scorecards) -> str:
    """Plan.md §24-25 scorecards plus the decision evaluation, as markdown."""
    scope = f" for {cards.ticker}" if cards.ticker else ""
    lines = [
        f"## Scorecards{scope} (as of {cards.as_of:%Y-%m-%d %H:%M %Z})",
        "",
        f"- Scored forecasts: {cards.n_forecasts}"
        + (
            f" ({cards.overlapping_excluded} more left out: their windows overlap a later "
            "forecast, so they are not independent evidence)"
            if cards.overlapping_excluded
            else ""
        ),
    ]
    if not cards.n_forecasts:
        return "\n".join([*lines, "- No scored forecasts yet.", ""])
    lines += [
        f"- Small samples are noisy: findings need at least {cards.min_samples} forecasts, and "
        "intervals show how uncertain each rate is.",
        "",
        "### Analysts (directional calls)",
        "",
    ]
    shared_note = len({a.weighting_note for a in cards.analysts}) == 1
    for a in cards.analysts:
        lines.append(f"- **{a.analyst}**: {_rate(a.pooled)}")
        if a.pooled.n:
            lines += [
                _breakdown("by ticker", a.by_ticker),
                _breakdown("by sector", a.by_sector),
                _breakdown("by regime", a.by_regime),
            ]
        if not shared_note:
            lines.append(f"  - weighting: {a.weighting_note}")
    if shared_note and cards.analysts:
        lines.append(f"- Weighting: {cards.analysts[0].weighting_note}")
    lines.append("")
    lines += _group_table("market regime", cards.by_regime)
    lines += _group_table("ticker", cards.by_ticker)
    lines += _group_table("sector", cards.by_sector)
    lines += [
        "### Decision-engine decisions",
        "",
        "LLM value added = quant-baseline Brier loss − final loss (positive: the final forecast "
        "was better). An association only; no decision rule changes automatically.",
        "",
        "| Decision | n | Adjusted | Mean value added (95% CI) | Improved | Direction "
        "| 80% coverage | Finding |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in cards.decisions:
        if d.mean_llm_value_added is None:
            added = "—"
        elif d.ci_low is None or d.ci_high is None:
            added = f"{d.mean_llm_value_added:+.4f}"
        else:
            added = f"{d.mean_llm_value_added:+.4f} ({d.ci_low:+.4f} to {d.ci_high:+.4f})"
        improved = "—" if d.share_improved is None else f"{d.share_improved:.0%}"
        coverage = "—" if d.coverage_80pct is None else f"{d.coverage_80pct:.0%}"
        lines.append(
            f"| {d.decision_type}: `{d.decision}` | {d.n} | {d.n_adjusted} | {added} | {improved} "
            f"| {_rate(d.direction)} | {coverage} | {d.finding} |"
        )
    lines.append("")
    return "\n".join(lines)
