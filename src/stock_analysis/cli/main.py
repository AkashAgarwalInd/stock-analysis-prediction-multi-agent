from datetime import UTC, date, datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from stock_analysis.config import get_settings
from stock_analysis.database import (
    Database,
    ForecastSnapshotStore,
    LearningStore,
    LearningStoreError,
    MemoryStore,
    OutcomeStore,
    OutcomeStoreError,
    close_database,
    init_database,
    run_migrations,
)
from stock_analysis.learning import (
    LearningCycle,
    RssHindsightNewsSource,
    build_scorecards,
    compare_benchmarks,
    render_adaptation,
    render_lesson_actions,
    render_postmortem,
    render_scorecards,
)
from stock_analysis.logging import configure_logging, get_logger
from stock_analysis.review import OutcomeReviewer, render_last_forecast_vs_actual

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
    """Score matured forecasts, diagnose them, update lessons and calibration."""
    db = init_database()
    try:
        snapshots, outcomes = ForecastSnapshotStore(db), OutcomeStore(db)
        memory, learning = MemoryStore(db), LearningStore(db)
        for store in (snapshots, outcomes, memory, learning):
            store.ensure_schema()
        now = datetime.now(UTC)
        results = OutcomeReviewer(snapshots, outcomes).review_matured(now, ticker=ticker)
        if not results:
            console.print("No matured forecasts awaiting evaluation.")
        for outcome in results:
            snapshot = snapshots.get(outcome.forecast_id)
            if snapshot is not None:
                console.print(render_last_forecast_vs_actual(snapshot, outcome), markup=False)

        llm_factory = news_source = None
        if llm:
            from stock_analysis.llm.factory import get_llm_factory

            llm_factory, news_source = get_llm_factory(), RssHindsightNewsSource()
        report = LearningCycle(
            snapshots, outcomes, memory, learning, llm_factory=llm_factory, news_source=news_source
        ).run(now, ticker=ticker)
        for postmortem in report.postmortems:
            console.print(f"Forecast `{postmortem.forecast_id}`", markup=False)
            console.print(render_postmortem(postmortem), markup=False)
        if report.lesson_actions:
            console.print(render_lesson_actions(report.lesson_actions), markup=False)
        for update in report.calibration_updates:
            scored = [
                o
                for fid in learning.scored_forecast_ids(update.ticker, before=now, limit=10_000)
                if (o := outcomes.get(fid)) is not None
            ]
            console.print(render_adaptation(update, compare_benchmarks(scored)), markup=False)
        for error in report.errors:
            console.print(f"Learning error: {error}", markup=False)
    finally:
        close_database()


@app.command()
def scorecard(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Only this ticker"),
    database: Optional[str] = typer.Option(
        None, "--database", help="Database to read, e.g. a backtest's (default: the app database)"
    ),
) -> None:
    """Show analyst, regime and decision scorecards (evaluation only)."""
    if database and not Path(database).exists():
        console.print(f"No database at {database}", markup=False)
        raise typer.Exit(1)
    db = Database(Path(database)) if database else init_database()
    try:
        outcomes, learning = OutcomeStore(db), LearningStore(db)
        try:
            outcomes.ensure_schema()
            learning.ensure_schema()
        except (OutcomeStoreError, LearningStoreError) as err:
            console.print(str(err), markup=False)
            raise typer.Exit(1) from err
        # Tickers are stored without the exchange suffix (RELIANCE, not RELIANCE.NS)
        base = ticker.upper().split(".")[0] if ticker else None
        cards = build_scorecards(outcomes, learning, as_of=datetime.now(UTC), ticker=base)
        console.print(render_scorecards(cards), markup=False)
    finally:
        if database:
            db.close()
        else:
            close_database()


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
