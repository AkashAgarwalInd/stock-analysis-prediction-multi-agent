"""Deterministic report sections for the learning loop and its evaluation (Plan.md §27-31)."""

from __future__ import annotations

from typing import Optional

from stock_analysis.learning.lessons import LessonAction
from stock_analysis.schemas.learning import (
    BenchmarkComparison,
    CalibrationParams,
    CalibrationUpdate,
    PostMortem,
)
from stock_analysis.schemas.scorecard import GroupScore, HitRate, Scorecards
from stock_analysis.schemas.track_record import (
    VARIANT_LABELS,
    ForecastTrackRecord,
    MeanEffect,
    ProbabilityCalibration,
    ReliabilityTable,
    TrackRecordWindow,
)


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


# --- Plan.md Phase 14: track record and probability calibration -----------------------


def _opt(value: Optional[float], fmt: str) -> str:
    return "—" if value is None else format(value, fmt)


def _pct(value: Optional[float], fmt: str) -> str:
    return "—" if value is None else f"{format(value, fmt)}%"


def _effect(label: str, effect: MeanEffect) -> str:
    if effect.mean is None:
        return f"- {label}: no data"
    interval = (
        f" (95% CI {effect.ci_low:+.4f} to {effect.ci_high:+.4f})"
        if effect.ci_low is not None and effect.ci_high is not None
        else ""
    )
    return f"- {label}: {effect.mean:+.4f}{interval} over {effect.n} forecasts"


def _window(window: TrackRecordWindow) -> list[str]:
    since = f", targets from {window.start}" if window.start else ""
    lines = [f"### {window.label}{since}: {window.n} forecasts", ""]
    if not window.n:
        return [*lines, "- No evaluated forecasts in this window.", ""]
    lines += [
        "| Variant | Direction | Brier | Pinball | 80% coverage | PIT <P10 / <P50 / >P90 | MAE "
        "| Mean signed error | Vol ratio | Sharpness |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for v in window.variants:
        coverage = _opt(v.coverage_80pct, ".0%")
        pit = (
            "—"
            if v.pit_below_p10 is None
            else f"{v.pit_below_p10:.0%} / {_opt(v.pit_below_p50, '.0%')} / "
            f"{_opt(v.pit_above_p90, '.0%')}"
        )
        lines.append(
            f"| {VARIANT_LABELS[v.variant]} | {_rate(v.direction)} | {_opt(v.brier, '.4f')} "
            f"| {_opt(v.pinball, '.3f')} | {coverage} | {pit} | {_pct(v.mean_abs_error_pct, '.2f')} "
            f"| {_pct(v.mean_signed_error_pct, '+.2f')} | {_opt(v.mean_vol_ratio, '.2f')} "
            f"| {_pct(v.mean_sharpness_pct, '.2f')} |"
        )
    llm_label = "LLM value added (calibrated-quant Brier − final Brier)"
    cal_label = "Calibration value added (uncalibrated − calibrated quant Brier)"
    lines += [
        "",
        (
            f"{_effect(llm_label, window.llm_value_added)}; "
            f"{window.n_adjusted} of {window.n} adjusted by the LLM"
            if window.n_adjusted
            else f"- {llm_label}: none of the {window.n} forecasts was adjusted by the LLM"
        ),
        (
            f"{_effect(cal_label, window.calibration_value_added)}; "
            f"{window.n_calibrated} of {window.n} calibrated"
            if window.n_calibrated
            else f"- {cal_label}: no calibration was applied to any of the {window.n} forecasts"
        ),
        f"- Finding: {window.finding}",
        "",
    ]
    return lines


def render_track_record(record: ForecastTrackRecord) -> str:
    """Plan.md §28 / §31 forecast track record with the §53 benchmarks, as markdown."""
    scope = f" for {record.ticker}" if record.ticker else ""
    lines = [
        f"## Forecast track record{scope} (as of {record.as_of:%Y-%m-%d %H:%M %Z})",
        "",
        f"- Evaluated forecasts: {record.n_forecasts}"
        + (
            f" ({record.overlapping_excluded} more left out: their windows overlap a later "
            "forecast, so they are not independent evidence)"
            if record.overlapping_excluded
            else ""
        ),
    ]
    if not record.n_forecasts:
        return "\n".join([*lines, "- No scored forecasts yet.", ""])
    lines += [
        "- Naive flat predicts no change (P50 = last close, direction flat); it states no "
        "probabilities or range. Brier and pinball: lower is better. Sharpness is the "
        "P10–P90 width as a share of the price: narrower is sharper, but only worth it with "
        "coverage near 80%. PIT: the share of actual closes below P10, below P50 and above P90 "
        "(about 10%, 50% and 10% when the range is calibrated).",
        f"- No variant is called better on fewer than {record.min_samples} forecasts.",
        "",
    ]
    for window in record.windows:
        lines += _window(window)
    return "\n".join(lines)


def _decomposition(table: ReliabilityTable) -> str:
    if table.brier is None:
        return "no forecasts"
    return (
        f"Brier {table.brier:.4f} ≈ reliability {_opt(table.reliability, '.4f')} − resolution "
        f"{_opt(table.resolution, '.4f')} + uncertainty {_opt(table.uncertainty, '.4f')}; "
        f"ECE {_opt(table.ece, '.1%')}; base rate {_opt(table.base_rate, '.0%')}"
    )


_EVENT_LABELS = {"up": "P(up)", "flat": "P(flat)", "down": "P(down)", "all": "All classes"}


def _reliability_rows(table: ReliabilityTable) -> list[str]:
    lines = [
        f"#### {_EVENT_LABELS[table.event]}",
        "",
        "| Predicted | n | Mean predicted | Observed (95% CI) | Gap |",
        "|---|---|---|---|---|",
    ]
    for b in table.buckets:
        o = b.observed
        lines.append(
            f"| {b.low:.0%}–{b.high:.0%} | {b.n} | {b.mean_predicted:.1%} "
            f"| {o.rate:.1%} ({o.ci_low:.0%}–{o.ci_high:.0%}) | {b.gap:+.1%} |"
        )
    return [*lines, "", f"- {_decomposition(table)}", f"- Finding: {table.finding}", ""]


def render_probability_calibration(
    calibrations: list[ProbabilityCalibration], *, n_buckets: int, ticker: Optional[str] = None
) -> str:
    """Plan.md §27 / §55: predicted probability vs observed frequency.

    The final forecast gets full bucket tables; the quant variants a summary
    line per event so they can be compared with it.
    """
    scope = f" for {ticker}" if ticker else ""
    n = calibrations[0].n_forecasts if calibrations else 0
    lines = [f"## Probability calibration{scope}", "", f"- Evaluated forecasts: {n}"]
    if not n:
        return "\n".join([*lines, "- No scored forecasts yet.", ""])
    lines += [
        f"- Each predicted probability is put in one of {n_buckets} equal buckets and compared "
        "with how often that outcome happened. A negative gap means the event happened less "
        "often than predicted. Reliability and ECE are 0 when probabilities match frequencies.",
        "",
    ]
    for cal in calibrations:
        label = VARIANT_LABELS[cal.variant]
        lines += [f"### {label}: mean up/flat/down Brier {_opt(cal.brier, '.4f')}", ""]
        if cal.variant == "final":
            for event in ("up", "flat", "down"):
                lines += _reliability_rows(cal.events[event])
            lines += [f"- All classes pooled: {_decomposition(cal.events['all'])}", ""]
        else:
            for event in ("up", "flat", "down", "all"):
                table = cal.events[event]
                lines.append(
                    f"- {_EVENT_LABELS[event]}: {_decomposition(table)}; {table.finding}"
                )
            lines.append("")
    return "\n".join(lines)


def render_calibration_history(ticker: str, history: list[CalibrationParams]) -> str:
    """Every stored calibration version for ``ticker``, oldest first (Plan.md §21-22)."""
    lines = [f"## Calibration versions for {ticker}", ""]
    if not history:
        return "\n".join(
            [*lines, "- None stored: the quant baseline is used uncalibrated (multiplier 1.00).", ""]
        )
    for c in history:
        lines += [
            f"- v{c.version} at {c.created_at:%Y-%m-%d %H:%M %Z}: volatility multiplier "
            f"{c.previous_vol_multiplier:.2f} → {c.vol_multiplier:.2f}, P50 shift "
            f"{c.previous_p50_bias_shift_pct:+.2f}% → {c.p50_bias_shift_pct:+.2f}% "
            f"({c.n_samples} forecasts, {c.calibrator_version})",
            *(f"  - {r}" for r in c.reason),
        ]
    lines.append("")
    return "\n".join(lines)
