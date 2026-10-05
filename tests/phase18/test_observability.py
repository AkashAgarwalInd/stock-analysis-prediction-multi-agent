"""Phase 18: LLM call log, cost tracking and node-level structured logs (Plan.md §50-51)."""

import json
import os
import subprocess
import sys

import pytest
import structlog
from typer.testing import CliRunner

from stock_analysis.analysis import run_analysis
from stock_analysis.cli.main import app
from stock_analysis.database import LLMCallStore
from stock_analysis.langgraph.observability import instrument_node, node_status
from stock_analysis.llm.usage import LLMCallRecord, current_node
from stock_analysis.schemas.graph_state import GraphState
from tests.forecast_helpers import MADE_AT
from tests.hardening_helpers import stub_factory
from tests.outcome_helpers import NOW_MATURED, FakePriceSource
from tests.phase17.test_analyze import FakeSource, _rally_prices


def _analyze(db, **options):
    return run_analysis(
        "reliance",
        db,
        llm_factory=stub_factory(),
        source=FakeSource(),
        run_at=MADE_AT,
        news_source=None,
        price_source=FakePriceSource({}),
        **options,
    )


class TestLLMCallLog:
    def test_store_round_trip_and_summary(self, migrated_db):
        store = LLMCallStore(migrated_db)
        store.ensure_schema()
        record = LLMCallRecord(
            node="predictor", role="primary", model="m", prompt_tokens=100,
            completion_tokens=20, latency_ms=50, cost_est=0.00002, status="ok",
            run_id="r1", ticker="RELIANCE",
        )  # fmt: skip
        store.record(record)
        store.record(LLMCallRecord(**{**record.__dict__, "run_id": "r2", "status": "error"}))
        assert store.calls(run_id="r1") == [record]
        summary = store.summary(ticker="RELIANCE")
        assert (summary.calls, summary.failed_calls, summary.prompt_tokens) == (2, 1, 200)

        store.attach_forecast("r1", "f-1")
        row = migrated_db.fetchone("SELECT forecast_id FROM llm_calls WHERE run_id = 'r1'")
        assert row["forecast_id"] == "f-1"

    def test_missing_table_is_reported(self, temp_db_path):
        from stock_analysis.database import Database, LLMCallStoreError

        db = Database(temp_db_path)
        try:
            assert not LLMCallStore(db).available()
            with pytest.raises(LLMCallStoreError, match="migrate"):
                LLMCallStore(db).ensure_schema()
        finally:
            db.close()


class TestForecastRunAccounting:
    def test_every_call_is_tagged_stored_and_linked(self, migrated_db):
        result = _analyze(migrated_db)
        usage = result.llm_usage
        assert result.report is not None
        # four analysts and at least one predictor call, each through the real factory
        assert usage.calls >= 5 and usage.failed_calls == 0
        assert {"technical_analyst", "fundamental_analyst", "sentiment_analyst",
                "context_analyst", "predictor"} <= usage.by_node.keys()  # fmt: skip
        assert usage.prompt_tokens > 0 and usage.cost_est > 0

        rows = migrated_db.fetchall("SELECT * FROM llm_calls")
        assert len(rows) == usage.calls
        assert {r["forecast_id"] for r in rows} == {result.state.forecast_id}
        assert {r["ticker"] for r in rows} == {"RELIANCE"}
        assert len({r["run_id"] for r in rows}) == 1
        assert {r["model"] for r in rows} == {"gemini-stub"}

    def test_budget_degrades_the_run_instead_of_failing_it(self, migrated_db, monkeypatch):
        from stock_analysis.config.settings import get_settings

        monkeypatch.setenv("LLM_MAX_CALLS_PER_RUN", "2")
        get_settings.cache_clear()
        result = _analyze(migrated_db)
        assert result.report is not None  # the forecast still ships
        assert result.llm_usage.calls == 2
        degraded = [
            r for r in (result.state.technical_report, result.state.fundamental_report,
                        result.state.sentiment_report, result.state.context_report)
            if r["confidence"] == 0.0
        ]  # fmt: skip
        assert len(degraded) == 2
        assert result.state.final_forecast["adjustment_applied"] is False

    def test_quant_only_run_makes_no_calls(self, migrated_db):
        result = _analyze(migrated_db, use_llm=False)
        assert result.llm_usage.calls == 0
        assert migrated_db.fetchone("SELECT COUNT(*) AS n FROM llm_calls")["n"] == 0


@pytest.fixture
def wide_console(monkeypatch):
    from stock_analysis.cli import main

    monkeypatch.setattr(main.console, "width", 250)


class TestReviewAccounting:
    def test_review_postmortem_calls_are_logged_and_tagged(self, migrated_db, store, outcome_store):
        from stock_analysis.database import LearningStore, MemoryStore
        from stock_analysis.langgraph.review_graph import run_review

        first = _analyze(migrated_db, use_llm=False)
        snapshot = store.get(first.state.forecast_id)
        state = run_review(
            NOW_MATURED,
            ticker="RELIANCE",
            snapshot_store=store,
            outcome_store=outcome_store,
            memory_store=MemoryStore(migrated_db),
            learning_store=LearningStore(migrated_db),
            price_source=_rally_prices(snapshot),
            llm_factory=stub_factory(),
            news_source=None,
        )
        assert state.postmortems  # the rally is outside the band: an LLM postmortem
        assert state.llm_usage.calls >= 1 and state.llm_usage.by_node == {
            "postmortem": state.llm_usage.calls
        }
        rows = LLMCallStore(migrated_db).calls()
        assert len(rows) == state.llm_usage.calls
        assert all(r.run_id.startswith("review-") and r.node == "postmortem" for r in rows)

    def test_review_inside_a_tracked_run_reports_only_its_own_calls(self, migrated_db, store):
        from stock_analysis.langgraph.review_graph import run_review
        from stock_analysis.llm.usage import LLMUsageTracker, track_llm_usage

        tracker = LLMUsageTracker(run_id="outer")
        tracker.record(
            LLMCallRecord(node="x", role="primary", model="m", prompt_tokens=1,
                          completion_tokens=1, latency_ms=1, cost_est=0.0, status="ok")
        )  # fmt: skip
        with track_llm_usage(tracker):
            state = run_review(NOW_MATURED, snapshot_store=store)
        assert state.llm_usage.calls == 0
        assert LLMCallStore(migrated_db).calls() == []  # the outer tracker has no sink


class TestCli:
    def test_usage_command(self, migrated_db, temp_db_path, wide_console):
        _analyze(migrated_db)
        out = CliRunner().invoke(app, ["usage", "--database", str(temp_db_path)])
        assert out.exit_code == 0, out.output
        assert "LLM usage:" in out.output
        assert "predictor" in out.output and "gemini-stub" in out.output

    def test_history_command(self, migrated_db, temp_db_path, wide_console):
        result = _analyze(migrated_db, use_llm=False)
        out = CliRunner().invoke(
            app, ["history", "--ticker", "RELIANCE.NS", "--database", str(temp_db_path)]
        )
        assert out.exit_code == 0, out.output
        final = result.state.final_forecast
        assert "Forecast history: RELIANCE" in out.output
        assert f"{final['p50_price']:.2f}" in out.output
        assert "pending" in out.output  # not matured yet

    def test_history_without_forecasts(self, migrated_db, temp_db_path):
        out = CliRunner().invoke(app, ["history", "-t", "INFY", "--database", str(temp_db_path)])
        assert out.exit_code == 0 and "No stored forecasts for INFY" in out.output

    def test_review_rejects_ticker_with_all(self):
        out = CliRunner().invoke(app, ["review", "--all", "--ticker", "RELIANCE"])
        assert out.exit_code == 2


class TestNodeInstrumentation:
    def test_node_tags_llm_calls_and_logs(self):
        seen = {}

        def node(state):
            seen["node"] = current_node()
            seen["context"] = structlog.contextvars.get_contextvars().get("node")
            return {"predictor_result": {"prob_up": 0.4}}

        wrapped = instrument_node("predictor", node)
        state = GraphState(symbol="RELIANCE", resolved_symbol="RELIANCE.NS")
        assert wrapped(state) == {"predictor_result": {"prob_up": 0.4}}
        assert seen == {"node": "predictor", "context": "predictor"}
        assert current_node() is None  # reset after the node
        assert wrapped.__wrapped__ is node

    def test_failure_is_reraised(self):
        def node(state):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            instrument_node("critic", node)(
                GraphState(symbol="RELIANCE", resolved_symbol="RELIANCE.NS")
            )

    def test_status(self):
        assert node_status({"predictor_result": {"error": "Invalid predictor output"}}) == "error"
        assert node_status({"predictor_result": {"prob_up": 0.4}}) == "ok"
        assert node_status(None) == "ok"

    def test_logs_are_json_on_stderr(self):
        """Report output owns stdout; logs carry run_id, ticker, node, latency and status."""
        code = (
            "import structlog\n"
            "from stock_analysis.logging import configure_logging, get_logger\n"
            "configure_logging()\n"
            "with structlog.contextvars.bound_contextvars(run_id='r1', ticker='RELIANCE'):\n"
            "    get_logger('t').info('node_completed', node='critic', latency_ms=3, status='ok')\n"
        )
        env = {**os.environ, "LOG_FORMAT": "json", "LOG_LEVEL": "INFO"}
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        assert done.stdout == ""
        event = json.loads(done.stderr.strip().splitlines()[-1])
        assert {"run_id", "ticker", "node", "latency_ms", "status"} <= event.keys()
        assert event["event"] == "node_completed"
