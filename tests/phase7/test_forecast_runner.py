"""Phase 7: the production runner persists every completed forecast."""

import pytest

from stock_analysis.database import (
    ForecastSnapshotError,
    ForecastSnapshotStore,
    close_database,
    get_database,
)
from stock_analysis.langgraph.runner import run_forecast
from stock_analysis.schemas.graph_state import GraphState
from tests.forecast_helpers import DownLLM, FakeLLM


@pytest.fixture
def default_database():
    """Reset the application DB singleton around a test that relies on it."""
    close_database()
    yield
    close_database()


@pytest.mark.usefixtures("price_history", "default_database")
class TestRunForecast:
    def test_persists_to_configured_database(self, migrated_db, initial_state):
        state = run_forecast(initial_state, llm_factory=FakeLLM("valid"))

        assert state.snapshot_persisted is True
        stored = ForecastSnapshotStore(migrated_db).get(state.forecast_id)
        assert stored is not None
        assert stored.final_forecast.model_dump(mode="json") == state.final_forecast
        sha = stored.data_inputs["price"]["history_sha256"]
        assert ForecastSnapshotStore(migrated_db).get_price_history(sha) is not None
        assert get_database().path == migrated_db.path

    def test_persists_degraded_quant_only_forecast(self, migrated_db, initial_state):
        state = run_forecast(initial_state, llm_factory=DownLLM())
        assert state.snapshot_persisted is True
        assert ForecastSnapshotStore(migrated_db).get(state.forecast_id) is not None

    def test_fills_company_name_from_symbol_index(self, migrated_db):
        state = GraphState(symbol="RELIANCE", resolved_symbol="RELIANCE.NS")
        result = run_forecast(state, llm_factory=FakeLLM("valid"))

        stored = ForecastSnapshotStore(migrated_db).get(result.forecast_id)
        assert stored.company_name == "Reliance Industries Ltd"

    def test_unmigrated_database_fails_fast(self, temp_db_path, monkeypatch, initial_state):
        monkeypatch.setenv("DATABASE_PATH", str(temp_db_path))
        with pytest.raises(ForecastSnapshotError, match="stock-analysis migrate"):
            run_forecast(initial_state, llm_factory=FakeLLM("valid"))
