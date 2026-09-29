import typer
from rich.console import Console
from rich.table import Table

from stock_analysis.config import get_settings
from stock_analysis.database import init_database, close_database, run_migrations
from stock_analysis.logging import configure_logging, get_logger

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

    for field_name, field_info in settings.model_fields.items():
        value = getattr(settings, field_name)
        if field_name == "gemini_api_key" and value:
            value = "***" + value[-4:] if len(value) > 4 else "***"
        table.add_row(field_name, str(value))

    console.print(table)


if __name__ == "__main__":
    app()