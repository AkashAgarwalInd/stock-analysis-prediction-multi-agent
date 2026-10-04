"""Plan.md Phase 13: analyst, regime and decision scorecards (evaluation only).

Everything is computed from immutable outcome records with a point-in-time
cutoff, so a scorecard "as of" a past date is exactly what was knowable then.
Forecasts whose windows overlap a later forecast of the same ticker are left
out (the newest is kept): overlapping forecasts see the same price move, and
counting them separately would overstate the evidence.
Small samples are shown with their uncertainty: hit rates carry a 95% Wilson
interval and decision effects a 95% Student-t interval, and no finding is drawn
below the configured minimum sample size. Analyst weighting is never applied:
the scorecard only reports whether it *would* be eligible (enough samples and
the feature flag on).
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Optional

from stock_analysis.config.settings import Settings, get_settings
from stock_analysis.database.learning_store import LearningStore
from stock_analysis.database.outcome_store import OutcomeStore
from stock_analysis.schemas.learning import DecisionOutcome
from stock_analysis.schemas.scorecard import (
    AnalystScorecard,
    DecisionEvaluation,
    GroupScore,
    HitRate,
    Scorecards,
    ScoredForecast,
)

ANALYSTS = ("technical", "fundamental", "sentiment", "context")
_Z95 = 1.96
_UNKNOWN = "unknown"


def hit_rate(hits: int, n: int) -> HitRate:
    """Hit rate with a 95% Wilson score interval (None when there are no samples)."""
    if n == 0:
        return HitRate(n=0, hits=0)
    p = hits / n
    denom = 1 + _Z95**2 / n
    centre = (p + _Z95**2 / (2 * n)) / denom
    half = _Z95 * math.sqrt(p * (1 - p) / n + _Z95**2 / (4 * n * n)) / denom
    return HitRate(
        n=n, hits=hits, rate=p, ci_low=max(0.0, centre - half), ci_high=min(1.0, centre + half)
    )


def _grouped_hits(pairs: Iterable[tuple[str, bool]]) -> dict[str, HitRate]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for key, hit in pairs:
        counts[key][0] += int(hit)
        counts[key][1] += 1
    return {key: hit_rate(h, n) for key, (h, n) in sorted(counts.items())}


def analyst_scorecards(
    forecasts: list[ScoredForecast], settings: Optional[Settings] = None
) -> list[AnalystScorecard]:
    """Plan.md §24: each analyst's directional calls, pooled and by ticker/sector/regime.

    Only forecasts where the analyst made a call count: missing, degraded and
    deliberately disabled analysts have no entry in ``analyst_hits``.
    """
    settings = settings or get_settings()
    cards = []
    for analyst in ANALYSTS:
        calls = [(f, f.analyst_hits[analyst]) for f in forecasts if analyst in f.analyst_hits]
        pooled = hit_rate(sum(hit for _, hit in calls), len(calls))
        enough = pooled.n >= settings.min_samples_analyst_weights
        if not settings.analyst_weighting_enabled:
            note = "automatic analyst weighting is disabled (ANALYST_WEIGHTING_ENABLED=false)"
        elif not enough:
            note = (
                f"only {pooled.n} calls; weighting needs at least "
                f"{settings.min_samples_analyst_weights}"
            )
        else:
            note = "enough calls for weighting"
        cards.append(
            AnalystScorecard(
                analyst=analyst,
                pooled=pooled,
                by_ticker=_grouped_hits((f.ticker, hit) for f, hit in calls),
                by_sector=_grouped_hits((f.sector or _UNKNOWN, hit) for f, hit in calls),
                by_regime=_grouped_hits((f.market_regime, hit) for f, hit in calls),
                weighting_eligible=settings.analyst_weighting_enabled and enough,
                weighting_note=note,
            )
        )
    return cards


def group_scores(
    forecasts: list[ScoredForecast], key: Callable[[ScoredForecast], str]
) -> list[GroupScore]:
    """Plan.md §25: forecast quality per group (regime, ticker or sector)."""
    groups: dict[str, list[ScoredForecast]] = defaultdict(list)
    for f in forecasts:
        groups[key(f)].append(f)
    return [
        GroupScore(
            group=name,
            n=len(items),
            direction=hit_rate(sum(f.direction_correct for f in items), len(items)),
            mean_abs_error_pct=statistics.fmean(f.abs_error_pct for f in items),
            mean_signed_error_pct=statistics.fmean(f.signed_error_pct for f in items),
            brier=statistics.fmean(f.brier for f in items),
            coverage_80pct=statistics.fmean(float(f.in_80pct_band) for f in items),
        )
        for name, items in sorted(groups.items())
    ]


def independent_forecasts(
    forecasts: list[ScoredForecast],
) -> tuple[list[ScoredForecast], list[str]]:
    """Per ticker, keep the newest forecast and drop any whose window overlaps one kept.

    Returns the kept forecasts (oldest first) and the IDs left out.
    """
    kept: list[ScoredForecast] = []
    excluded: list[str] = []
    newest_first = sorted(forecasts, key=lambda f: (f.target_date, f.forecast_id), reverse=True)
    for f in newest_first:
        if any(
            k.ticker == f.ticker and f.as_of_date < k.target_date and k.as_of_date < f.target_date
            for k in kept
        ):
            excluded.append(f.forecast_id)
        else:
            kept.append(f)
    return kept[::-1], excluded


# Two-sided 95% Student-t critical values for 1..30 degrees of freedom
_T95_TABLE = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip


def _t95(df: int) -> float:
    """Two-sided 95% Student-t critical value: table to 30 df, Cornish-Fisher beyond."""
    if df <= len(_T95_TABLE):
        return _T95_TABLE[df - 1]
    z = _Z95
    return (
        z
        + (z**3 + z) / (4 * df)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)
    )


def mean_ci(values: list[float]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Mean with a two-sided 95% Student-t interval (no interval below two values)."""
    if not values:
        return None, None, None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None, None
    half = _t95(len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def _decision_finding(
    n: int,
    n_adjusted: int,
    mean: Optional[float],
    low: Optional[float],
    high: Optional[float],
    minimum: int,
) -> str:
    if n and n_adjusted == 0:
        return "no forecast was adjusted under this decision: the final forecast was the quant baseline"
    if n < minimum or mean is None or low is None or high is None:
        return f"insufficient evidence: {n} forecasts, at least {minimum} needed"
    if low > 0:
        return "forecasts under this decision improved on the quant baseline on average"
    if high < 0:
        return "forecasts under this decision were worse than the quant baseline on average"
    return "no measurable effect on forecast quality"


def decision_evaluations(
    outcomes: list[DecisionOutcome], settings: Optional[Settings] = None
) -> list[DecisionEvaluation]:
    """Does any decision-engine decision correlate with forecast improvement?

    "Improvement" is ``llm_value_added`` (quant-baseline Brier loss minus final
    loss). This is an association across forecasts, not a causal effect, and it
    never changes how decisions are made.
    """
    settings = settings or get_settings()
    groups: dict[tuple[str, str], list[DecisionOutcome]] = defaultdict(list)
    for o in outcomes:
        groups[(o.decision_type, o.decision)].append(o)
    evaluations = []
    for (decision_type, decision), items in sorted(groups.items()):
        added = [o.llm_value_added for o in items if o.llm_value_added is not None]
        pinball = [
            o.llm_value_added_pinball for o in items if o.llm_value_added_pinball is not None
        ]
        calls = [o.direction_correct for o in items if o.direction_correct is not None]
        bands = [o.in_80pct_band for o in items if o.in_80pct_band is not None]
        mean, low, high = mean_ci(added)
        n_adjusted = sum(o.adjustment_applied for o in items)
        evaluations.append(
            DecisionEvaluation(
                decision_type=decision_type,
                decision=decision,
                n=len(items),
                n_adjusted=n_adjusted,
                mean_llm_value_added=mean,
                ci_low=low,
                ci_high=high,
                share_improved=(sum(v > 0 for v in added) / len(added)) if added else None,
                mean_llm_value_added_pinball=statistics.fmean(pinball) if pinball else None,
                direction=hit_rate(sum(calls), len(calls)),
                coverage_80pct=statistics.fmean(map(float, bands)) if bands else None,
                finding=_decision_finding(
                    len(added),
                    n_adjusted,
                    mean,
                    low,
                    high,
                    settings.min_samples_decision_evaluation,
                ),
            )
        )
    return evaluations


def build_scorecards(
    outcome_store: OutcomeStore,
    learning_store: LearningStore,
    *,
    as_of: datetime,
    ticker: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> Scorecards:
    """All scorecards from the records that existed at ``as_of``."""
    settings = settings or get_settings()
    forecasts, excluded = independent_forecasts(
        outcome_store.scored_forecasts(before=as_of, ticker=ticker)
    )
    kept = {f.forecast_id for f in forecasts}
    decisions = [
        d
        for d in learning_store.list_decision_outcomes(before=as_of, ticker=ticker)
        if d.forecast_id in kept
    ]
    return Scorecards(
        as_of=as_of,
        ticker=ticker,
        n_forecasts=len(forecasts),
        overlapping_excluded=len(excluded),
        min_samples=settings.min_samples_analyst_weights,
        analysts=analyst_scorecards(forecasts, settings),
        by_regime=group_scores(forecasts, lambda f: f.market_regime),
        by_ticker=group_scores(forecasts, lambda f: f.ticker),
        by_sector=group_scores(forecasts, lambda f: f.sector or _UNKNOWN),
        decisions=decision_evaluations(decisions, settings),
    )
