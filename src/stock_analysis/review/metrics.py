"""Plan.md §15 outcome metrics: pure, deterministic functions (no LLM)."""

from __future__ import annotations

import math
import statistics
from typing import Optional, Sequence

from stock_analysis.schemas.outcome import Direction

# Standard normal quantile at 0.9: P10/P90 sit this many sigmas from the median
_Z90 = 1.2815515655446004
# The quant baseline scales daily volatility to weekly with sqrt(5)
_TRADING_DAYS_PER_WEEK = 5


def pct_return(start: float, end: float) -> float:
    """Simple return from ``start`` to ``end`` in percent."""
    return (end / start - 1.0) * 100.0


def realized_direction(return_pct: float, flat_threshold_pct: float) -> Direction:
    """Direction of an actual move, with the same flat band the forecast probabilities use."""
    if return_pct > flat_threshold_pct:
        return "up"
    if return_pct < -flat_threshold_pct:
        return "down"
    return "flat"


def predicted_direction(prob_up: float, prob_flat: float, prob_down: float) -> Direction:
    """Most probable direction; an exact tie is treated as no directional call (flat)."""
    probs: dict[Direction, float] = {"up": prob_up, "flat": prob_flat, "down": prob_down}
    best = max(probs.values())
    winners = [k for k, v in probs.items() if v == best]
    return winners[0] if len(winners) == 1 else "flat"


def brier_score(prob_up: float, prob_flat: float, prob_down: float, realized: Direction) -> float:
    """Multi-class Brier score over up/flat/down (0 = perfect, 2 = worst)."""
    outcome = {"up": 0.0, "flat": 0.0, "down": 0.0}
    outcome[realized] = 1.0
    return (
        (prob_up - outcome["up"]) ** 2
        + (prob_flat - outcome["flat"]) ** 2
        + (prob_down - outcome["down"]) ** 2
    )


def pinball_loss(quantile: float, predicted: float, actual: float) -> float:
    """Quantile (pinball) loss of one predicted quantile."""
    diff = actual - predicted
    return max(quantile * diff, (quantile - 1.0) * diff)


def quantile_losses(
    last_close: float, p10: float, p50: float, p90: float, actual_return_pct: float
) -> dict[str, float]:
    """Pinball losses for P10/P50/P90 in return percentage points, plus their mean.

    Working in returns (not prices) keeps losses comparable across tickers.
    """
    losses = {
        name: pinball_loss(q, pct_return(last_close, price), actual_return_pct)
        for name, q, price in (("p10", 0.1, p10), ("p50", 0.5, p50), ("p90", 0.9, p90))
    }
    losses["mean"] = (losses["p10"] + losses["p50"] + losses["p90"]) / 3.0
    return losses


def pit_percentile(actual: float, p10: float, p50: float, p90: float) -> float:
    """Where ``actual`` fell in the forecast distribution (0-1).

    Uses a split-normal on log price fitted exactly through P10, P50 and P90,
    so it returns 0.1 at P10, 0.5 at P50 and 0.9 at P90 and extends smoothly
    into the tails. Requires 0 < P10 < P50 < P90.
    """
    if not 0 < p10 < p50 < p90 or actual <= 0:
        raise ValueError("PIT needs positive prices with P10 < P50 < P90")
    x, median = math.log(actual), math.log(p50)
    sigma = (median - math.log(p10)) / _Z90 if x <= median else (math.log(p90) - median) / _Z90
    z = (x - median) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def realized_weekly_vol_pct(closes: Sequence[float]) -> Optional[float]:
    """Sample volatility of daily log returns, scaled to a week, in percent.

    Needs at least three closes (two returns); a five-day window is a noisy
    estimate, so read single ratios with care.
    """
    if len(closes) < 3:
        return None
    returns = [math.log(b / a) for a, b in zip(closes, closes[1:], strict=False)]
    return statistics.stdev(returns) * math.sqrt(_TRADING_DAYS_PER_WEEK) * 100.0


def vol_ratio(realized_pct: Optional[float], predicted_pct: float) -> Optional[float]:
    """Realized / predicted weekly volatility (>1 means volatility was underestimated)."""
    if realized_pct is None or predicted_pct <= 0:
        return None
    return realized_pct / predicted_pct
