import subprocess
from pathlib import Path

from stock_analysis.config import get_settings
from stock_analysis.logging import get_logger

logger = get_logger(__name__)

ALEMBIC_DIR = Path(__file__).parent.parent.parent.parent / "alembic"


def run_migrations() -> None:
    settings = get_settings()
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        cwd=ALEMBIC_DIR.parent,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error("migration_failed", stdout=result.stdout, stderr=result.stderr)
        raise RuntimeError(f"Migration failed: {result.stderr}")

    logger.info("migrations_completed", stdout=result.stdout)


def get_migration_status() -> dict:
    result = subprocess.run(
        ["alembic", "current"],
        cwd=ALEMBIC_DIR.parent,
        capture_output=True,
        text=True,
    )

    return {
        "current_revision": result.stdout.strip() if result.returncode == 0 else None,
        "error": result.stderr if result.returncode != 0 else None,
    }


def create_migration(message: str) -> None:
    result = subprocess.run(
        ["alembic", "revision", "--autogenerate", "-m", message],
        cwd=ALEMBIC_DIR.parent,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error("migration_creation_failed", stdout=result.stdout, stderr=result.stderr)
        raise RuntimeError(f"Migration creation failed: {result.stderr}")

    logger.info("migration_created", stdout=result.stdout)