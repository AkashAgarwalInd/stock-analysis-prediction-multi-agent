"""Shared Phase 7 fixtures: a migrated SQLite DB and completed forecast states."""

from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from alembic.config import Config
from phase7_helpers import ALEMBIC_DIR, LAST_BAR, MADE_AT, DownLLM, FakeLLM, run_graph

from alembic import command
from stock_analysis.config.settings import get_settings
from stock_analysis.database import Database, ForecastSnapshotStore, MemoryStore
from stock_analysis.schemas.graph_state import GraphState
from stock_analysis.snapshots import build_forecast_snapshot


@pytest.fixture
def price_history():
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
    return ForecastSnapshotStore(migrated_db)


@pytest.fixture
def memory_store(migrated_db):
    return MemoryStore(migrated_db)


@pytest.fixture
def adjusted_state(price_history, initial_state):
    return run_graph(FakeLLM("valid"), initial_state)


@pytest.fixture
def fallback_state(price_history, initial_state):
    return run_graph(FakeLLM("invalid"), initial_state)


@pytest.fixture
def guardrail_state(price_history, initial_state):
    return run_graph(DownLLM(), initial_state)


@pytest.fixture
def snapshot(adjusted_state):
    return build_forecast_snapshot(adjusted_state, made_at=MADE_AT)
