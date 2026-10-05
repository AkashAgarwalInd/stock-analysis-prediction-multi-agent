"""End-to-end demo on several stocks (Plan.md Phase 18, §56 demo flow, §76 acceptance).

For each ticker, in its own database (a backtest refuses a database that
already holds later history, so tickers cannot share one):

1. ``backtest``: weekly forecasts simulated point-in-time, each scored and
   learned from before the next one;
2. ``evaluate``: track record against the naive and quant benchmarks plus the
   final forecast's probability calibration;
3. ``calibration``: the calibration versions and every variant's probability
   calibration;
4. ``analyze``: today's forecast, which reviews the backtest's forecasts first
   and shows the last forecast vs actual, the adaptation and the track record.

Every report is written under the output directory, and the analysis report is
checked against the §76 list of what it must visibly contain.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from stock_analysis.analysis import AnalysisResult, InputSource, run_analysis
from stock_analysis.backtest import (
    BacktestError,
    BacktestResult,
    render_backtest_report,
    run_backtest,
)
from stock_analysis.backtest.data import load_yfinance_history
from stock_analysis.database import (
    Database,
    ForecastSnapshotError,
    ForecastSnapshotStore,
    LearningStore,
    LearningStoreError,
    MemoryStoreError,
    OutcomeStore,
    OutcomeStoreError,
)
from stock_analysis.learning import (
    build_evaluation,
    render_calibration_history,
    render_probability_calibration,
    render_track_record,
)
from stock_analysis.llm.usage import LLMUsageSummary
from stock_analysis.logging import get_logger
from stock_analysis.market.resolver import resolve_nse_ticker
from stock_analysis.snapshots import DISCLAIMER

logger = get_logger(__name__)

# Plan.md §42 default symbols, without TATAMOTORS (demerged in 2025)
DEFAULT_TICKERS = ("RELIANCE", "INFY", "HDFCBANK")

# Plan.md §76: what the final analysis must visibly contain, and the report text showing it
ACCEPTANCE_ITEMS: tuple[tuple[str, str], ...] = (
    ("Previous forecast", "| P50 / close |"),
    ("Actual outcome", "- Actual return:"),
    ("Forecast error", "| Error vs P50 |"),
    ("Baseline error", "| | Quant baseline | Final forecast | Actual |"),
    ("LLM value added", "- LLM value added:"),
    ("Postmortem", "- Primary cause:"),
    ("Learning/adaptation", "## System adaptation"),
    ("Current market regime", "- Market regime:"),
    ("Analyst perspectives", "## Current analysis"),
    ("Quant baseline", "## Quant baseline"),
    ("Final constrained forecast", "## Final forecast"),
    ("Track record", "## Track record"),
    ("Benchmark comparison", "## Benchmark comparison"),
    ("Risks/invalidation triggers", "## Invalidation triggers"),
    ("Disclaimer", DISCLAIMER),
)


def acceptance_checklist(report: str) -> dict[str, bool]:
    """Which §76 items ``report`` visibly contains."""
    return {item: marker in report for item, marker in ACCEPTANCE_ITEMS}


@dataclass
class DemoStockResult:
    """The demo's outcome for one ticker."""

    ticker: str
    database_path: Path
    reports: dict[str, Path] = field(default_factory=dict)
    backtest: Optional[BacktestResult] = None
    analysis: Optional[AnalysisResult] = None
    checklist: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def scored_weeks(self) -> int:
        return sum(w.status == "scored" for w in self.backtest.weeks) if self.backtest else 0

    @property
    def missing_items(self) -> list[str]:
        return [item for item, ok in self.checklist.items() if not ok]

    @property
    def passed(self) -> bool:
        return not self.errors and bool(self.checklist) and not self.missing_items

    @property
    def llm_usage(self) -> LLMUsageSummary:
        records = [
            u
            for u in (
                self.backtest.llm_usage if self.backtest else None,
                self.analysis.llm_usage if self.analysis else None,
            )
            if u is not None
        ]
        return LLMUsageSummary(
            calls=sum(u.calls for u in records),
            failed_calls=sum(u.failed_calls for u in records),
            prompt_tokens=sum(u.prompt_tokens for u in records),
            completion_tokens=sum(u.completion_tokens for u in records),
            cost_est=round(sum(u.cost_est for u in records), 6),
        )


def default_output_dir(now: datetime) -> Path:
    return Path("data/demo") / f"{now:%Y%m%d-%H%M%S}"


def _write(result: DemoStockResult, name: str, text: str, directory: Path) -> None:
    path = directory / f"{name}.md"
    path.write_text(text)
    result.reports[name] = path


def _evaluation_reports(db: Database, ticker: str, as_of: datetime) -> tuple[str, str]:
    """The ``evaluate`` and ``calibration`` command output for ``ticker``, as markdown."""
    report = build_evaluation(
        ForecastSnapshotStore(db), OutcomeStore(db), as_of=as_of, ticker=ticker
    )
    final = [c for c in report.probability_calibration if c.variant == "final"]
    evaluate = "\n\n".join(
        (
            render_track_record(report.track_record),
            render_probability_calibration(final, n_buckets=report.n_buckets, ticker=ticker),
        )
    )
    calibration = "\n\n".join(
        (
            render_calibration_history(ticker, LearningStore(db).calibration_history(ticker)),
            render_probability_calibration(
                report.probability_calibration, n_buckets=report.n_buckets, ticker=ticker
            ),
        )
    )
    return evaluate, calibration


def run_stock_demo(
    query: str,
    *,
    weeks: int,
    output_dir: Path,
    use_llm: bool,
    llm_factory: Any = None,
    now: Optional[datetime] = None,
    history_loader: Callable[..., Any] = load_yfinance_history,
    input_source: Optional[InputSource] = None,
    progress: Optional[Callable[[str], None]] = None,
    analysis_options: Optional[dict[str, Any]] = None,
    **backtest_options: Any,
) -> DemoStockResult:
    """Backtest, evaluate, inspect the calibration and analyze one stock.

    A failing step is recorded in ``errors`` and the later steps that do not
    depend on it still run; expected backtest, store and analysis errors never raise.
    ``now`` is the clock of the evaluation and the analysis (default: now).
    ``analysis_options`` go to ``run_analysis`` (e.g. the review's
    ``price_source``), ``backtest_options`` to ``run_backtest``.
    """
    say = progress or (lambda _msg: None)
    ticker, _, _ = resolve_nse_ticker(query)
    directory = output_dir / ticker
    directory.mkdir(parents=True, exist_ok=True)
    result = DemoStockResult(ticker=ticker, database_path=directory / f"{ticker}.db")
    clock = now or datetime.now(UTC)

    say(f"[{ticker}] backtest: {weeks} weeks")
    try:
        result.backtest = run_backtest(
            ticker,
            weeks,
            database_path=result.database_path,
            llm_factory=llm_factory,
            use_llm=use_llm,
            now=now,
            history_loader=history_loader,
            **backtest_options,
        )
    except (BacktestError, ValueError) as err:
        result.errors.append(f"backtest: {err}")
        return result
    _write(result, "backtest", render_backtest_report(result.backtest), directory)
    result.errors += [f"backtest week: {e}" for e in result.backtest.errors]

    db = Database(result.database_path)
    try:
        say(f"[{ticker}] evaluate and calibration")
        try:
            evaluate, calibration = _evaluation_reports(db, ticker, clock)
            _write(result, "evaluate", evaluate, directory)
            _write(result, "calibration", calibration, directory)
        except (ForecastSnapshotError, OutcomeStoreError, LearningStoreError) as err:
            result.errors.append(f"evaluation: {err}")

        say(f"[{ticker}] analyze")
        try:
            result.analysis = run_analysis(
                ticker,
                db,
                use_llm=use_llm,
                llm_factory=llm_factory,
                source=input_source,
                run_at=now,
                **(analysis_options or {}),
            )
        except (
            ForecastSnapshotError,
            OutcomeStoreError,
            LearningStoreError,
            MemoryStoreError,
        ) as err:
            result.errors.append(f"analyze: {err}")
            return result
    finally:
        db.close()

    report = result.analysis.report
    if report is None:
        result.errors.append(f"analyze: no forecast ({result.analysis.error})")
        return result
    _write(result, "analysis", report, directory)
    result.checklist = acceptance_checklist(report)
    return result


def run_demo(
    tickers: Sequence[str] = DEFAULT_TICKERS,
    *,
    weeks: int = 12,
    output_dir: Optional[Path] = None,
    use_llm: bool = False,
    progress: Optional[Callable[[str], None]] = None,
    **options: Any,
) -> list[DemoStockResult]:
    """:func:`run_stock_demo` for each ticker, in order, into ``output_dir``; a stock that
    fails unexpectedly is reported in its result and the others still run."""
    output_dir = output_dir or default_output_dir(datetime.now(UTC))
    results = []
    for query in tickers:
        try:
            result = run_stock_demo(
                query,
                weeks=weeks,
                output_dir=output_dir,
                use_llm=use_llm,
                progress=progress,
                **options,
            )
        except Exception as err:  # one stock's unexpected failure must not stop the others
            logger.exception("demo_stock_failed", query=query)
            result = DemoStockResult(ticker=query.upper(), database_path=output_dir / query)
            result.errors.append(f"unexpected {type(err).__name__}: {err}"[:300])
        logger.info(
            "demo_stock_completed",
            ticker=result.ticker,
            scored_weeks=result.scored_weeks,
            passed=result.passed,
            errors=len(result.errors),
        )
        results.append(result)
    return results


def render_demo_summary(results: Sequence[DemoStockResult]) -> str:
    """Markdown summary: one row per stock, then any missing §76 items, errors and reports."""
    lines = [
        "# Demo summary",
        "",
        "| Ticker | Weeks scored | Calibrations | Today P(up) | §76 items | LLM calls | Result |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        weeks = f"{r.scored_weeks}/{len(r.backtest.weeks)}" if r.backtest else "—"
        versions = str(len(r.backtest.calibration_history)) if r.backtest else "—"
        forecast = r.analysis.state.final_forecast if r.analysis and r.analysis.report else None
        prob_up = f"{forecast['prob_up']:.2f}" if forecast else "—"
        shown = f"{sum(r.checklist.values())}/{len(ACCEPTANCE_ITEMS)}" if r.checklist else "—"
        lines.append(
            f"| {r.ticker} | {weeks} | {versions} | {prob_up} | {shown} | {r.llm_usage.calls} "
            f"| {'pass' if r.passed else 'FAIL'} |"
        )
    lines.append("")
    for r in results:
        for item in r.missing_items:
            lines.append(f"- {r.ticker}: the analysis report does not show “{item}”")
        lines += [f"- {r.ticker}: {e}" for e in r.errors]
        if r.reports:
            names = ", ".join(f"{name}.md" for name in r.reports)
            lines.append(f"- {r.ticker} reports ({names}): `{r.database_path.parent}`")
    usage = LLMUsageSummary(
        calls=sum(r.llm_usage.calls for r in results),
        failed_calls=sum(r.llm_usage.failed_calls for r in results),
        prompt_tokens=sum(r.llm_usage.prompt_tokens for r in results),
        completion_tokens=sum(r.llm_usage.completion_tokens for r in results),
        cost_est=round(sum(r.llm_usage.cost_est for r in results), 6),
    )
    lines += [f"- {usage.describe()}", "", DISCLAIMER, ""]
    return "\n".join(lines)
