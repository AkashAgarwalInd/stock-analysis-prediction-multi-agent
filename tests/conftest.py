import os
import tempfile
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

os.environ.setdefault("APP_ENV", "testing")
os.environ.setdefault("GEMINI_API_KEY", "test-key-12345")
# Phase 18: tests never wait for rate limits or retry backoff
for _name in ("LLM_REQUESTS_PER_MINUTE", "YFINANCE_REQUESTS_PER_MINUTE", "NEWS_REQUESTS_PER_MINUTE"):
    os.environ.setdefault(_name, "0")
for _name in ("LLM_RETRY_BASE_DELAY_SECONDS", "EXTERNAL_RETRY_BASE_DELAY_SECONDS"):
    os.environ.setdefault(_name, "0")


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    from stock_analysis.config.settings import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def temp_db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    yield path
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Forecast pipeline fixtures (snapshots, memory, outcomes). Imports stay inside
# the fixtures so the environment above is set before stock_analysis loads.
# ---------------------------------------------------------------------------


@pytest.fixture
def price_history():
    """300 synthetic daily closes ending on LAST_BAR, patched into the quant node."""
    import numpy as np

    from tests.forecast_helpers import LAST_BAR

    rng = np.random.default_rng(0)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 300)))
    bars = [
        SimpleNamespace(close=Decimal(str(round(c, 4))), date=LAST_BAR - timedelta(days=299 - i))
        for i, c in enumerate(closes)
    ]
    history = SimpleNamespace(data=bars)
    collector = SimpleNamespace(fetch_history=lambda *a, **k: history)
    with patch("stock_analysis.market.collector.get_price_collector", return_value=collector):
        yield history


@pytest.fixture
def initial_state():
    from stock_analysis.schemas.graph_state import GraphState

    return GraphState(
        symbol="RELIANCE",
        resolved_symbol="RELIANCE.NS",
        company_name="Reliance Industries Ltd",
        technical_indicators_summary={"rsi_14": 55.2, "macd_histogram": 1.4},
        fundamentals_summary={"pe_ratio": 25.3},
        news_summary={"article_count": 10},
        market_context_summary={"india_vix": 14.5},
    )


@pytest.fixture
def migrated_db(temp_db_path, monkeypatch):
    """A temp SQLite DB upgraded to alembic head (the production schema path)."""
    from alembic.config import Config

    from alembic import command
    from stock_analysis.config.settings import get_settings
    from stock_analysis.database import Database
    from tests.forecast_helpers import ALEMBIC_DIR

    monkeypatch.setenv("DATABASE_PATH", str(temp_db_path))
    get_settings.cache_clear()
    cfg = Config()
    cfg.set_main_option("script_location", str(ALEMBIC_DIR))
    command.upgrade(cfg, "head")
    db = Database(temp_db_path)
    yield db
    db.close()
    for suffix in ("-wal", "-shm"):
        Path(f"{temp_db_path}{suffix}").unlink(missing_ok=True)


@pytest.fixture
def store(migrated_db):
    from stock_analysis.database import ForecastSnapshotStore

    return ForecastSnapshotStore(migrated_db)


@pytest.fixture
def memory_store(migrated_db):
    from stock_analysis.database import MemoryStore

    return MemoryStore(migrated_db)


@pytest.fixture
def outcome_store(migrated_db):
    from stock_analysis.database import OutcomeStore

    return OutcomeStore(migrated_db)


@pytest.fixture
def adjusted_state(price_history, initial_state):
    from tests.forecast_helpers import FakeLLM, run_graph

    return run_graph(FakeLLM("valid"), initial_state)


@pytest.fixture
def fallback_state(price_history, initial_state):
    from tests.forecast_helpers import FakeLLM, run_graph

    return run_graph(FakeLLM("invalid"), initial_state)


@pytest.fixture
def guardrail_state(price_history, initial_state):
    from tests.forecast_helpers import DownLLM, run_graph

    return run_graph(DownLLM(), initial_state)


@pytest.fixture
def snapshot(adjusted_state):
    from stock_analysis.snapshots import build_forecast_snapshot
    from tests.forecast_helpers import MADE_AT

    return build_forecast_snapshot(adjusted_state, made_at=MADE_AT)
