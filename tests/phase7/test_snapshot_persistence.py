"""Phase 7: forecast snapshot persistence, versions and daily predictions."""

import hashlib
import json
from datetime import date

import numpy as np
import pytest
from phase7_helpers import MADE_AT, FakeLLM, run_graph
from pydantic import ValidationError

from stock_analysis.config.settings import get_settings
from stock_analysis.market.calendar import get_trading_calendar
from stock_analysis.quant import QuantForecaster
from stock_analysis.schemas.analyst_reports import AnalystReport, MarketRegime, RiskCategory
from stock_analysis.schemas.forecast_pipeline import FinalForecast, PredictorResult
from stock_analysis.schemas.snapshot import ForecastSnapshot, compute_data_snapshot_id
from stock_analysis.snapshots import SnapshotNotReadyError, build_forecast_snapshot
from stock_analysis.versions import ANALYST_PROMPT_FILES, PREDICTOR_PROMPT_VERSION, PROMPTS_DIR


class TestForecastPersistence:
    def test_round_trip_is_identical(self, store, snapshot):
        store.save(snapshot)
        loaded = store.get(snapshot.forecast_id)

        assert loaded == snapshot
        assert loaded.model_dump(mode="json") == snapshot.model_dump(mode="json")

    def test_required_fields_persisted(self, store, snapshot, adjusted_state):
        store.save(snapshot)
        loaded = store.get(snapshot.forecast_id)

        assert loaded.ticker == "RELIANCE"
        assert loaded.company_name == "Reliance Industries Ltd"
        assert loaded.made_at == MADE_AT
        assert loaded.as_of_date == date(2026, 9, 29)
        # 5 NSE trading days after Tue 2026-09-29, skipping the 2026-10-02 holiday
        assert loaded.target_date == date(2026, 10, 7)
        assert loaded.last_close == adjusted_state.quant_baseline["last_close"]
        assert loaded.market_regime == MarketRegime.TRENDING_UP
        assert loaded.risk_category == RiskCategory.LOW
        assert loaded.quant_baseline == adjusted_state.quant_baseline
        assert loaded.final_forecast == FinalForecast(**adjusted_state.final_forecast)
        assert loaded.analyst_reports["technical"] == adjusted_state.technical_report
        assert loaded.critic_verdict is not None and loaded.critic_verdict.passed

    def test_flat_numeric_columns_match_final_forecast(self, store, snapshot, migrated_db):
        store.save(snapshot)
        row = migrated_db.fetchone(
            "SELECT * FROM forecast_snapshots WHERE forecast_id = ?", (snapshot.forecast_id,)
        )
        final = snapshot.final_forecast
        for col in ("prob_up", "prob_flat", "prob_down", "p10_price", "p50_price", "p90_price"):
            assert row[col] == getattr(final, col)
        assert row["adjustment_applied"] == 1
        assert len(row["snapshot_hash"]) == 64

    def test_unknown_forecast_returns_none(self, store):
        assert store.get("does-not-exist") is None
        assert store.get_latest("does-not-exist") is None
        assert store.list_versions("does-not-exist") == []

    def test_list_for_ticker(self, store, adjusted_state):
        first = build_forecast_snapshot(adjusted_state, made_at=MADE_AT)
        second = build_forecast_snapshot(adjusted_state, made_at=MADE_AT.replace(hour=13))
        store.save(first)
        store.save(second)

        assert [s.forecast_id for s in store.list_for_ticker("RELIANCE")] == [
            second.forecast_id,
            first.forecast_id,
        ]
        assert store.list_for_ticker("TCS") == []

    def test_graph_persists_every_completed_forecast(self, price_history, initial_state, store):
        state = run_graph(FakeLLM("valid"), initial_state, store)

        assert state.snapshot_persisted is True
        persisted = store.get(state.forecast_id)
        assert persisted is not None
        assert persisted.data_snapshot_id == state.data_snapshot_id
        assert persisted.final_forecast == FinalForecast(**state.final_forecast)

    def test_graph_without_store_builds_but_does_not_persist(self, adjusted_state):
        assert adjusted_state.forecast_id is not None
        assert adjusted_state.snapshot_persisted is False
        assert adjusted_state.forecast_report is not None

    def test_snapshot_preserves_forecast_numbers(self, adjusted_state, fallback_state):
        for state in (adjusted_state, fallback_state):
            snap = build_forecast_snapshot(state, made_at=MADE_AT)
            assert snap.final_forecast.model_dump(mode="json") == state.final_forecast

    def test_no_snapshot_without_completed_forecast(self, adjusted_state):
        broken = adjusted_state.model_copy(update={"final_forecast": {"error": "no forecast"}})
        with pytest.raises(SnapshotNotReadyError):
            build_forecast_snapshot(broken)


class TestDataSnapshotId:
    def test_deterministic_for_same_inputs(self, adjusted_state):
        a = build_forecast_snapshot(adjusted_state, made_at=MADE_AT)
        b = build_forecast_snapshot(adjusted_state, made_at=MADE_AT)
        assert a.forecast_id != b.forecast_id
        assert a.data_snapshot_id == b.data_snapshot_id

    def test_changes_when_inputs_change(self, adjusted_state):
        base = build_forecast_snapshot(adjusted_state, made_at=MADE_AT)
        changed = adjusted_state.model_copy(update={"news_summary": {"article_count": 11}})
        assert build_forecast_snapshot(changed).data_snapshot_id != base.data_snapshot_id

    def test_covers_price_history_fingerprint(self, snapshot):
        price = snapshot.data_inputs["price"]
        assert price["last_date"] == "2026-09-29"
        assert price["bars"] == 300
        assert len(price["history_sha256"]) == 64

    def test_id_must_match_stored_inputs(self, snapshot):
        data = snapshot.model_dump()
        data["data_inputs"] = {**data["data_inputs"], "news": {"article_count": 999}}
        with pytest.raises(ValidationError, match="data_snapshot_id"):
            ForecastSnapshot.model_validate(data)
        assert compute_data_snapshot_id(snapshot.data_inputs) == snapshot.data_snapshot_id


class TestVersionPersistence:
    def test_model_versions_record_configured_models(self, monkeypatch, adjusted_state, store):
        monkeypatch.setenv("GEMINI_MODEL_PRIMARY", "gemini-test-primary")
        get_settings.cache_clear()
        snap = build_forecast_snapshot(adjusted_state, made_at=MADE_AT)
        store.save(snap)

        versions = store.get(snap.forecast_id).model_versions
        assert versions["llm_primary"] == "gemini-test-primary"
        assert versions["llm_fallback"] == get_settings().gemini_model_fallback
        assert versions["quant"] == (
            "ewma_monte_carlo(seed=42,n_paths=10000,ewma_lambda=0.94,"
            "flat_threshold_pct=0.5,history=2y)"
        )
        assert versions["critic"].startswith("deterministic@")

    def test_prompt_versions_are_content_hashes(self, store, snapshot):
        store.save(snapshot)
        versions = store.get(snapshot.forecast_id).prompt_versions

        for node, filename in ANALYST_PROMPT_FILES.items():
            digest = hashlib.sha256((PROMPTS_DIR / filename).read_bytes()).hexdigest()
            assert versions[node] == f"sha256:{digest[:16]}"
        assert versions["predictor"] == PREDICTOR_PROMPT_VERSION
        for key, schema in (
            ("analyst_response_schema", AnalystReport),
            ("predictor_response_schema", PredictorResult),
        ):
            digest = hashlib.sha256(json.dumps(schema.model_json_schema()).encode()).hexdigest()
            assert versions[key] == f"sha256:{digest[:16]}"

    def test_prompt_version_changes_with_prompt_text(self, monkeypatch, tmp_path, adjusted_state):
        before = build_forecast_snapshot(adjusted_state).prompt_versions
        for filename in ANALYST_PROMPT_FILES.values():
            (tmp_path / filename).write_bytes((PROMPTS_DIR / filename).read_bytes())
        (tmp_path / "technical_analyst.txt").write_text("edited prompt")
        monkeypatch.setattr("stock_analysis.versions.PROMPTS_DIR", tmp_path)

        after = build_forecast_snapshot(adjusted_state).prompt_versions
        assert after["technical_analyst"] != before["technical_analyst"]
        assert after["fundamental_analyst"] == before["fundamental_analyst"]

    def test_calibration_version_persisted(self, store, adjusted_state):
        snap = build_forecast_snapshot(adjusted_state, made_at=MADE_AT, calibration_version=3)
        store.save(snap)
        assert store.get(snap.forecast_id).calibration_version == 3

    def test_default_calibration_version_is_uncalibrated(self, snapshot):
        assert snapshot.calibration_version == 0


class TestDailyPredictions:
    def test_persisted_one_row_per_trading_day(self, store, snapshot, migrated_db):
        store.save(snapshot)
        rows = migrated_db.fetchall(
            "SELECT * FROM forecast_daily_predictions WHERE forecast_id = ? ORDER BY day_index",
            (snapshot.forecast_id,),
        )

        assert [r["day_index"] for r in rows] == [1, 2, 3, 4, 5]
        dates = [date.fromisoformat(r["target_date"]) for r in rows]
        assert dates == sorted(dates)
        assert date(2026, 10, 2) not in dates
        assert all(get_trading_calendar().is_trading_day(d) for d in dates)
        assert dates[-1] == snapshot.target_date
        for row in rows:
            assert row["p10_price"] < row["p50_price"] < row["p90_price"]
            assert 0.0 <= row["prob_up"] <= 1.0
            assert row["source"] == "quant_baseline"

    def test_round_trip(self, store, snapshot):
        store.save(snapshot)
        assert store.get(snapshot.forecast_id).daily_predictions == snapshot.daily_predictions

    def test_last_day_matches_quant_endpoint(self, snapshot):
        last = snapshot.daily_predictions[-1]
        quant = snapshot.quant_baseline
        assert last.p10_price == quant["p10_price"]
        assert last.p50_price == quant["p50_price"]
        assert last.p90_price == quant["p90_price"]
        assert last.prob_up == quant["prob_up"]
        assert last.predicted_return_pct == pytest.approx(quant["expected_return_pct"])

    def test_daily_path_does_not_change_quant_baseline(self, price_history):
        closes = [float(b.close) for b in price_history.data]

        prices = np.array(closes)
        forecaster = QuantForecaster(seed=42, n_paths=10000, horizon=5)
        baseline, daily = forecaster.forecast_with_daily_path(prices)
        assert QuantForecaster(seed=42, n_paths=10000, horizon=5).forecast(prices) == baseline
        assert len(daily) == 5

    def test_stored_json_has_no_dataframes_or_article_bodies(self, store, snapshot, migrated_db):
        store.save(snapshot)
        row = migrated_db.fetchone(
            "SELECT data_inputs_json FROM forecast_snapshots WHERE forecast_id = ?",
            (snapshot.forecast_id,),
        )
        inputs = json.loads(row["data_inputs_json"])
        assert set(inputs) == {
            "symbol",
            "resolved_symbol",
            "price",
            "technical_indicators",
            "fundamentals",
            "news",
            "market_context",
            "memory",
        }
        assert len(row["data_inputs_json"]) < 2000
