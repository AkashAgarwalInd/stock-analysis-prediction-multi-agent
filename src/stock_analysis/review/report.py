"""The "LAST FORECAST VS ACTUAL" section (Plan.md §17, §29).

Deterministic formatting of a stored forecast snapshot and its outcome; no
number here is computed beyond what the scorer stored.
"""

from __future__ import annotations

from stock_analysis.schemas.outcome import ForecastOutcome, OutcomeStatus
from stock_analysis.schemas.snapshot import ForecastSnapshot

_BASIS_NOTE_TOLERANCE = 0.005

SINGLE_OBSERVATION_NOTE = (
    "This is one observation: it does not show whether the LLM adjustment is "
    "generally better or worse than the quant baseline."
)


def _price(v: float) -> str:
    return f"₹{v:.2f}"


def _pct(v: float) -> str:
    return f"{v:+.2f}%"


def _value_added_text(value: float) -> str:
    if value > 0:
        return f"{value:+.4f} (the final forecast had lower loss than the quant baseline)"
    if value < 0:
        return f"{value:+.4f} (the final forecast had higher loss than the quant baseline)"
    return "0.0000 (same loss as the quant baseline)"


def _shadow_line(outcome: ForecastOutcome) -> str:
    """Uncalibrated quant vs calibrated quant (Plan.md §23 shadow forecast)."""
    if outcome.calibration_version == 0 or outcome.uncalibrated_loss is None:
        return "- Calibration: none applied; the quant baseline is the uncalibrated model"
    gain = outcome.calibration_value_added or 0.0
    verdict = "helped" if gain > 0 else ("hurt" if gain < 0 else "made no difference")
    return (
        f"- Uncalibrated quant loss ({outcome.loss_metric}): {outcome.uncalibrated_loss:.4f}; "
        f"calibration v{outcome.calibration_version} {verdict} on this forecast "
        f"({gain:+.4f})"
    )


def render_last_forecast_vs_actual(snapshot: ForecastSnapshot, outcome: ForecastOutcome) -> str:
    """Render the review of one evaluated forecast as markdown."""
    final = snapshot.final_forecast
    header = [
        "## Last forecast vs actual",
        "",
        f"- Forecast `{snapshot.forecast_id}` for {snapshot.ticker}, made "
        f"{snapshot.made_at.date().isoformat()} (as of {snapshot.as_of_date.isoformat()}), "
        f"target {snapshot.target_date.isoformat()}"
        + ("; a simulated backtest week" if snapshot.effective_source == "backtest" else ""),
    ]
    if outcome.status != OutcomeStatus.SCORED:
        label = "Not evaluated" if outcome.status == OutcomeStatus.INVALID else "Not yet evaluated"
        return "\n".join([*header, f"- {label}: {outcome.invalid_reason}", ""])

    if outcome.actual_close is None or outcome.actual_return_pct is None:
        raise ValueError(f"Scored outcome {outcome.forecast_id} has no actual close/return")
    actual = _price(outcome.actual_close)
    basis = outcome.actual_close_on_forecast_basis
    factor = outcome.adjustment_factor
    if basis is not None and factor is not None and abs(factor - 1.0) > _BASIS_NOTE_TOLERANCE:
        # Price history was re-adjusted (dividend/split) after the forecast was made
        actual += f" ({_price(basis)} on the forecast's price basis)"
    band = "inside" if outcome.in_80pct_band else "OUTSIDE"
    verdict = "correct" if outcome.direction_correct else "wrong"
    nifty = _pct(outcome.nifty_return_pct) if outcome.nifty_return_pct is not None else "n/a"
    relative = (
        _pct(outcome.excess_vs_nifty_pct) if outcome.excess_vs_nifty_pct is not None else "n/a"
    )

    lines = [
        *header,
        "",
        "| | Forecast | Actual |",
        "|---|---|---|",
        f"| P50 / close | {_price(final.p50_price)} | {actual} |",
        f"| Weekly return | {_pct(final.expected_return_pct)} expected | "
        f"{_pct(outcome.actual_return_pct)} |",
        f"| 80% band (P10–P90) | {_price(final.p10_price)}–{_price(final.p90_price)} | {band} |",
        f"| Direction | {str(outcome.predicted_direction).upper()} "
        f"(P(up) {final.prob_up * 100:.1f}%) | {str(outcome.realized_direction).upper()} "
        f"— {verdict} |",
        f"| Nifty 50 return | — | {nifty} |",
        f"| Stock vs Nifty | — | {relative} |",
        "",
        f"- Baseline loss ({outcome.loss_metric}): {outcome.baseline_loss:.4f}",
        f"- Final loss ({outcome.loss_metric}): {outcome.final_loss:.4f}",
        f"- LLM value added: {_value_added_text(outcome.llm_value_added or 0.0)}",
        _shadow_line(outcome),
        f"- Adjustment gate: {outcome.adjustment_gate_decision or 'n/a'}; adjustment applied: "
        f"{'yes' if outcome.adjustment_applied else 'no'}",
    ]
    if outcome.vol_ratio is not None:
        lines.append(f"- Realized / predicted volatility: {outcome.vol_ratio:.2f}x")
    if outcome.daily_band_breaches is not None:
        lines.append(f"- Daily band breaches (quant daily path): {outcome.daily_band_breaches}")
    lines += [f"- Data check: {c}" for c in outcome.validity_checks]
    lines += ["", SINGLE_OBSERVATION_NOTE, ""]
    return "\n".join(lines)
