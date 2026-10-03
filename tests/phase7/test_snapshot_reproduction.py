"""Phase 7: a stored snapshot's quant baseline can be reproduced from stored data."""

import sqlite3
from dataclasses import asdict

import pytest

from stock_analysis.database import SnapshotIntegrityError
from stock_analysis.schemas.snapshot import PriceHistorySnapshot
from stock_analysis.snapshots import (
    ReproductionError,
    build_forecast_snapshot,
    reproduce_quant_baseline,
)
from tests.forecast_helpers import MADE_AT, FakeLLM, run_graph


@pytest.fixture
def persisted(price_history, initial_state, store):
    state = run_graph(FakeLLM("valid"), initial_state, store)
    return store.get(state.forecast_id)


class TestPriceHistoryStorage:
    def test_graph_run_stores_exact_close_series(self, persisted, store, price_history):
        sha = persisted.data_inputs["price"]["history_sha256"]
        history = store.get_price_history(sha)

        assert history is not None
        assert history.closes == [str(b.close) for b in price_history.data]
        assert history.fingerprint() == persisted.data_inputs["price"]

    def test_saving_same_series_twice_is_idempotent(
        self, price_history, initial_state, store, migrated_db
    ):
        run_graph(FakeLLM("valid"), initial_state, store)
        run_graph(FakeLLM("valid"), initial_state, store)
        assert migrated_db.fetchone("SELECT COUNT(*) AS n FROM price_history_snapshots")["n"] == 1

    @pytest.mark.parametrize(
        "statement",
        [
            "UPDATE price_history_snapshots SET closes_json = '[]'",
            "DELETE FROM price_history_snapshots",
        ],
    )
    def test_price_history_is_immutable(self, persisted, migrated_db, statement):
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            migrated_db.execute(statement)

    def test_tampered_series_is_detected(self, persisted, store, migrated_db):
        sha = persisted.data_inputs["price"]["history_sha256"]
        migrated_db.execute("DROP TRIGGER price_history_snapshots_no_update")
        migrated_db.execute(
            "UPDATE price_history_snapshots SET closes_json = '[\"1.0\"]', dates_json = '[\"x\"]'"
        )
        with pytest.raises(SnapshotIntegrityError):
            store.get_price_history(sha)

    def test_hash_must_match_series(self):
        history = PriceHistorySnapshot.from_points("X.NS", [])
        with pytest.raises(ValueError, match="history_sha256"):
            PriceHistorySnapshot(
                history_sha256=history.history_sha256, symbol="X.NS", dates=["d"], closes=["1"]
            )


class TestReproduction:
    def test_reproduces_baseline_and_daily_path_exactly(self, persisted, store):
        baseline, daily = reproduce_quant_baseline(persisted, store)

        assert asdict(baseline) == persisted.quant_baseline
        assert [
            (d.p10_price, d.p50_price, d.p90_price, d.prob_up, d.predicted_return_pct)
            for d in daily
        ] == [
            (p.p10_price, p.p50_price, p.p90_price, p.prob_up, p.predicted_return_pct)
            for p in persisted.daily_predictions
        ]

    def test_missing_price_history_is_reported(self, adjusted_state, store):
        snap = build_forecast_snapshot(adjusted_state, made_at=MADE_AT)
        store.save(snap)
        with pytest.raises(ReproductionError, match="not stored"):
            reproduce_quant_baseline(snap, store)

    def test_changed_quant_config_is_refused(self, persisted, store, monkeypatch):
        monkeypatch.setattr("stock_analysis.versions.QUANT_SEED", 7)
        with pytest.raises(ReproductionError, match="Quant model changed"):
            reproduce_quant_baseline(persisted, store)
