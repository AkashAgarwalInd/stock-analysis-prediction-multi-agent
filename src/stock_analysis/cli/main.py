from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from stock_analysis.config import get_settings
from stock_analysis.database import (
    Database,
    ForecastSnapshotError,
    ForecastSnapshotStore,
    LearningStore,
    LearningStoreError,
    MemoryStore,
    MemoryStoreError,
    OutcomeStore,
    OutcomeStoreError,
    close_database,
    init_database,
    run_migrations,
)
from stock_analysis.learning import (
    RssHindsightNewsSource,
    build_evaluation,
    build_scorecards,
    render_adaptation,
    render_calibration_history,
    render_lesson_actions,
    render_postmortem,
    render_probability_calibration,
    render_scorecards,
    render_track_record,
)
from stock_analysis.logging import configure_logging, get_logger
from stock_analysis.review import render_last_forecast_vs_actual

app = typer.Typer(
    name="stock-analysis",
    help="Multi-Agent Indian Stock Analyst + Adaptive Weekly Forecaster",
    add_completion=False,
)

console = Console()
logger = get_logger(__name__)


@app.callback()
def callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    configure_logging()
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)


@app.command()
def init() -> None:
    """Initialize the database and run migrations."""
    settings = get_settings()
    console.print(f"[green]Initializing database at {settings.database_path}[/green]")
    init_database()
    try:
        run_migrations()
    except RuntimeError:
        logger.warning("migrations_skipped", message="Migrations skipped during init")
    close_database()
    console.print("[green]Database initialized successfully[/green]")


@app.command()
def migrate() -> None:
    """Run database migrations."""
    console.print("[yellow]Running migrations...[/yellow]")
    run_migrations()
    console.print("[green]Migrations completed[/green]")


@app.command()
def review(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Only this ticker"),
    llm: bool = typer.Option(
        True,
        "--llm/--no-llm",
        help="LLM postmortem with hindsight news (otherwise rules only, no network)",
    ),
) -> None:
    """Run the review graph: score matured forecasts, diagnose them, update lessons and
    calibration, then show the scorecards and track record."""
    from stock_analysis.langgraph.review_graph import run_review

    db = init_database()
    try:
        snapshots, outcomes = ForecastSnapshotStore(db), OutcomeStore(db)
        memory, learning = MemoryStore(db), LearningStore(db)
        llm_factory = news_source = None
        if llm:
            from stock_analysis.llm.factory import get_llm_factory

            llm_factory, news_source = get_llm_factory(), RssHindsightNewsSource()
        base = _base_ticker(ticker)
        state = run_review(
            datetime.now(UTC),
            ticker=base,
            snapshot_store=snapshots,
            outcome_store=outcomes,
            memory_store=memory,
            learning_store=learning,
            llm_factory=llm_factory,
            news_source=news_source,
        )
        if not state.outcomes:
            console.print("No matured forecasts awaiting evaluation.")
        for outcome in state.outcomes:
            snapshot = snapshots.get(outcome.forecast_id)
            if snapshot is not None:
                console.print(render_last_forecast_vs_actual(snapshot, outcome), markup=False)
        for postmortem in state.postmortems:
            console.print(f"Forecast `{postmortem.forecast_id}`", markup=False)
            console.print(render_postmortem(postmortem), markup=False)
        if state.lesson_actions:
            console.print(render_lesson_actions(state.lesson_actions), markup=False)
        for update in state.calibration_updates:
            console.print(
                render_adaptation(update, state.benchmarks.get(update.ticker)), markup=False
            )
        if state.scorecards is not None:
            console.print(render_scorecards(state.scorecards), markup=False)
        if state.evaluation is not None:
            console.print(render_track_record(state.evaluation.track_record), markup=False)
        for error in state.errors:
            console.print(f"Review error: {error}", markup=False)
    except (ForecastSnapshotError, OutcomeStoreError, LearningStoreError, MemoryStoreError) as err:
        console.print(f"Review failed: {err}", markup=False)
        raise typer.Exit(1) from err
    finally:
        close_database()


_DATABASE_HELP = "Database to read, e.g. a backtest's (default: the app database)"


def _base_ticker(ticker: Optional[str]) -> Optional[str]:
    """Tickers are stored without the exchange suffix (RELIANCE, not RELIANCE.NS)."""
    return ticker.upper().split(".")[0] if ticker else None


@contextmanager
def _read_database(database: Optional[str]) -> Iterator[Database]:
    """Open ``database`` (or the app database) with the evaluation tables checked."""
    if database and not Path(database).exists():
        console.print(f"No database at {database}", markup=False)
        raise typer.Exit(1)
    db = Database(Path(database)) if database else init_database()
    try:
        try:
            for store in (ForecastSnapshotStore(db), OutcomeStore(db), LearningStore(db)):
                store.ensure_schema()
        except (ForecastSnapshotError, OutcomeStoreError, LearningStoreError) as err:
            console.print(str(err), markup=False)
            raise typer.Exit(1) from err
        try:
            yield db
        except ForecastSnapshotError as err:  # e.g. a stored forecast failed its hash check
            console.print(f"Cannot read stored forecasts: {err}", markup=False)
            raise typer.Exit(1) from err
    finally:
        if database:
            db.close()
        else:
            close_database()


@app.command()
def scorecard(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Only this ticker"),
    database: Optional[str] = typer.Option(None, "--database", help=_DATABASE_HELP),
) -> None:
    """Show analyst, regime and decision scorecards (evaluation only)."""
    with _read_database(database) as db:
        cards = build_scorecards(
            OutcomeStore(db),
            LearningStore(db),
            as_of=datetime.now(UTC),
            ticker=_base_ticker(ticker),
        )
        console.print(render_scorecards(cards), markup=False)


@app.command()
def evaluate(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Only this ticker"),
    database: Optional[str] = typer.Option(None, "--database", help=_DATABASE_HELP),
) -> None:
    """Show the forecast track record against the naive and quant benchmarks."""
    with _read_database(database) as db:
        base = _base_ticker(ticker)
        report = build_evaluation(
            ForecastSnapshotStore(db), OutcomeStore(db), as_of=datetime.now(UTC), ticker=base
        )
        console.print(render_track_record(report.track_record), markup=False)
        final = [c for c in report.probability_calibration if c.variant == "final"]
        console.print(
            render_probability_calibration(final, n_buckets=report.n_buckets, ticker=base),
            markup=False,
        )


@app.command()
def calibration(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Only this ticker"),
    database: Optional[str] = typer.Option(None, "--database", help=_DATABASE_HELP),
) -> None:
    """Show calibration versions and predicted probability vs observed frequency."""
    with _read_database(database) as db:
        base = _base_ticker(ticker)
        learning = LearningStore(db)
        tickers = [base] if base else learning.calibrated_tickers()
        for name in tickers:
            console.print(
                render_calibration_history(name, learning.calibration_history(name)),
                markup=False,
            )
        report = build_evaluation(
            ForecastSnapshotStore(db), OutcomeStore(db), as_of=datetime.now(UTC), ticker=base
        )
        console.print(
            render_probability_calibration(
                report.probability_calibration, n_buckets=report.n_buckets, ticker=base
            ),
            markup=False,
        )


@app.command()
def backtest(
    ticker: str = typer.Option(..., "--ticker", "-t", help="NSE ticker, e.g. RELIANCE"),
    weeks: int = typer.Option(12, "--weeks", "-w", min=1, max=104, help="Weeks to simulate"),
    end: Optional[str] = typer.Option(
        None, "--end", help="Last target date (YYYY-MM-DD); default: latest completed week"
    ),
    database: Optional[str] = typer.Option(
        None, "--database", help="Database to write (default: a new file in data/backtests/)"
    ),
    llm: bool = typer.Option(
        True, "--llm/--no-llm", help="Use LLM analysts and postmortems (otherwise quant only)"
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write the report here"),
) -> None:
    """Simulate weekly forecasts point-in-time, scoring and learning week by week."""
    from stock_analysis.backtest import (
        BacktestError,
        default_database_path,
        render_backtest_report,
        run_backtest,
    )

    path = Path(database) if database else default_database_path(ticker.upper(), datetime.now(UTC))
    try:
        result = run_backtest(
            ticker,
            weeks,
            database_path=path,
            use_llm=llm,
            end=date.fromisoformat(end) if end else None,
            progress=lambda w: console.print(
                f"{w.as_of_date} → {w.target_date}: {w.status}", markup=False
            ),
        )
    except (BacktestError, ValueError) as err:
        console.print(f"Backtest failed: {err}", markup=False)
        raise typer.Exit(1) from err
    report = render_backtest_report(result)
    console.print(report, markup=False)
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(report)
        console.print(f"Report written to {output}", markup=False)


@app.command()
def analyze(
    query: str = typer.Argument(
        ..., help='NSE ticker or company name, e.g. RELIANCE or "tata motors"'
    ),
    database: Optional[str] = typer.Option(
        None, "--database", help="Database to use, e.g. a backtest's (default: the app database)"
    ),
    llm: bool = typer.Option(
        True,
        "--llm/--no-llm",
        help="LLM analysts, predictor and postmortems (otherwise quant only)",
    ),
    review: bool = typer.Option(
        True, "--review/--no-review", help="Review matured forecasts before this one"
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write the report here"),
) -> None:
    """Forecast one stock: last forecast vs actual, adaptation, current analysis, quant
    baseline, final forecast, track record and benchmarks."""
    from rich.markdown import Markdown

    from stock_analysis.analysis import run_analysis

    if database and not Path(database).exists():
        console.print(f"No database at {database}", markup=False)
        raise typer.Exit(1)
    if llm and not get_settings().gemini_api_key:
        console.print(
            "Warning: GEMINI_API_KEY is not set, so the LLM analysts will fail and the "
            "forecast stays the quant baseline; use --no-llm for a quant-only run.",
            markup=False,
        )
    db = Database(Path(database)) if database else init_database()
    try:
        with console.status(f"Analyzing {query}…"):
            result = run_analysis(query, db, use_llm=llm, review=review)
    except (ForecastSnapshotError, OutcomeStoreError, LearningStoreError, MemoryStoreError) as err:
        console.print(f"Analysis failed: {err}", markup=False)
        if not database:
            console.print("Run `stock-analysis init` to create the database.", markup=False)
        raise typer.Exit(1) from err
    finally:
        if database:
            db.close()
        else:
            close_database()

    for part, reason in result.unavailable_inputs.items():
        console.print(f"Input not available ({part}): {reason}", markup=False)
    report = result.report
    if report is None:
        console.print(f"No forecast for {result.resolved_symbol}: {result.error}", markup=False)
        raise typer.Exit(1)
    console.print(Markdown(report))
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(report)
        console.print(f"Report written to {output}", markup=False)


@app.command()
def status() -> None:
    """Show application status."""
    settings = get_settings()
    table = Table(title="Application Status")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("App Name", settings.app_name)
    table.add_row("Environment", settings.app_env)
    table.add_row("Log Level", settings.log_level)
    table.add_row("Database Path", str(settings.database_path))
    table.add_row("Gemini Model (Primary)", settings.gemini_model_primary)
    table.add_row("Gemini Model (Critic)", settings.gemini_model_critic)
    table.add_row("Gemini Model (Fallback)", settings.gemini_model_fallback)
    table.add_row("Forecast Horizon (days)", str(settings.forecast_horizon_days))

    console.print(table)


@app.command()
def config() -> None:
    """Show current configuration."""
    settings = get_settings()
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    for field_name in settings.model_fields:
        value = getattr(settings, field_name)
        if field_name == "gemini_api_key" and value:
            value = "***" + value[-4:] if len(value) > 4 else "***"
        table.add_row(field_name, str(value))

    console.print(table)


if __name__ == "__main__":
    app()
