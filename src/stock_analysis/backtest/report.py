"""Deterministic markdown summary of a backtest (Plan.md §56 demo part 1)."""

from __future__ import annotations

from typing import Optional

from stock_analysis.backtest.runner import BacktestResult
from stock_analysis.learning.report import (
    render_probability_calibration,
    render_scorecards,
    render_track_records,
)
from stock_analysis.snapshots.report import DISCLAIMER


def _opt(value: Optional[float], fmt: str) -> str:
    return "—" if value is None else format(value, fmt)


def _flag(value: Optional[bool], yes: str, no: str) -> str:
    return "—" if value is None else (yes if value else no)


def _cell(text: Optional[str]) -> str:
    """Free text safe inside a markdown table cell."""
    return (text or "").replace("|", "/").replace("\n", " ")


def render_backtest_report(result: BacktestResult) -> str:
    """Week-by-week table, aggregate accuracy, adaptation history and limitations."""
    scored = [w for w in result.weeks if w.status == "scored"]
    lines = [
        f"# Backtest: {result.ticker} ({result.resolved_symbol}), {len(result.weeks)} weeks",
        "",
        f"- Database: `{result.database_path}`",
        f"- LLM: {'enabled' if result.llm_enabled else 'disabled (quant-only)'}",
        *([f"- {result.llm_usage.describe()}"] if result.llm_enabled else []),
        "- Each week was forecast with only the data, lessons and calibration available at "
        "its as-of close, then scored before the next week was forecast.",
        "",
        "| As of | Target | Cal. | P10–P90 | P50 | Actual | Band | Direction | Final loss "
        "| Quant loss | Uncal. loss | Cause |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for w in result.weeks:
        if w.forecast_id is None:
            lines.append(
                f"| {w.as_of_date} | {w.target_date} | — | no forecast: {_cell(w.reason)} "
                "| | | | | | | | |"
            )
            continue
        band = f"{_opt(w.p10_price, '.2f')}–{_opt(w.p90_price, '.2f')}"
        actual = _opt(w.actual_close, ".2f")
        if w.actual_return_pct is not None:
            actual += f" ({w.actual_return_pct:+.2f}%)"
        if w.status not in ("scored",):
            actual = f"{w.status}: {_cell(w.reason)}"
        lines.append(
            f"| {w.as_of_date} | {w.target_date} | v{w.calibration_version} | {band} "
            f"| {_opt(w.p50_price, '.2f')} | {actual} "
            f"| {_flag(w.in_80pct_band, 'inside', 'OUTSIDE')} "
            f"| {_flag(w.direction_correct, 'correct', 'wrong')} "
            f"| {_opt(w.final_loss, '.4f')} | {_opt(w.baseline_loss, '.4f')} "
            f"| {_opt(w.uncalibrated_loss, '.4f')} | {w.primary_cause or '—'} |"
        )

    lines += ["", "## Accuracy", ""]
    if scored:
        inside = sum(bool(w.in_80pct_band) for w in scored)
        hits = sum(bool(w.direction_correct) for w in scored)
        lines += [
            f"- Scored weeks: {len(scored)} of {len(result.weeks)}",
            f"- Inside the 80% band: {inside} of {len(scored)} (a calibrated band is "
            "inside about 8 in 10 weeks)",
            f"- Direction correct: {hits} of {len(scored)}",
        ]
    else:
        lines.append("- No week could be scored.")
    b = result.benchmarks
    lines += [
        f"- Mean Brier loss — uncalibrated quant: {_opt(b.mean_uncalibrated_loss, '.4f')}, "
        f"calibrated quant: {_opt(b.mean_calibrated_loss, '.4f')}, "
        f"final forecast: {_opt(b.mean_final_loss, '.4f')}",
        f"- {b.finding}",
        f"- {len(scored)} weeks is a small sample: these numbers do not establish that any "
        "variant is better in general.",
        "",
        "## Adaptation",
        "",
    ]
    if result.calibration_history:
        for c in result.calibration_history:
            lines.append(
                f"- v{c.version} at {c.created_at:%Y-%m-%d}: volatility multiplier "
                f"{c.previous_vol_multiplier:.2f} → {c.vol_multiplier:.2f}, P50 shift "
                f"{c.previous_p50_bias_shift_pct:+.2f}% → {c.p50_bias_shift_pct:+.2f}% "
                f"({c.n_samples} forecasts)"
            )
    else:
        lines.append("- No calibration change: not enough evidence accumulated.")
    lines.append(f"- Active lessons at the end: {result.active_lessons}")
    evaluation = result.evaluation
    final = [c for c in evaluation.probability_calibration if c.variant == "final"]
    lines += [
        "",
        render_track_records(evaluation).rstrip(),
        "",
        render_probability_calibration(
            final, n_buckets=evaluation.n_buckets, ticker=result.ticker, sources=evaluation.sources
        ).rstrip(),
        "",
        render_scorecards(result.scorecards).rstrip(),
    ]

    lines += ["", "## Limitations", ""]
    lines += [f"- {name} analyst disabled: {why}" for name, why in result.disabled_analysts.items()]
    lines += [
        "- Prices are today's split/dividend-adjusted history, so price levels can differ "
        "slightly from what was quoted on each as-of date; returns are unaffected.",
        "- Postmortems have no hindsight news in a backtest.",
    ]
    if result.llm_enabled:
        lines.append(
            "- The LLM may already know how these dates played out from its training data; "
            "the code cannot prevent that. A quant-only run (--no-llm) is free of it."
        )
    lines += [f"- Error: {e}" for e in result.errors]
    lines += ["", "## Disclaimer", "", DISCLAIMER, ""]
    return "\n".join(lines)
