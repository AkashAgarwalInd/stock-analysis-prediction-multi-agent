"""End-to-end Phase 6 graph tests with a mocked LLM and synthetic prices."""

import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from stock_analysis.langgraph.workflow import compile_graph, route_after_revision
from stock_analysis.schemas.analyst_reports import AnalystReport
from stock_analysis.schemas.forecast_pipeline import FinalForecast, PredictorResult
from stock_analysis.schemas.graph_state import GraphState

MAX_REVISIONS = 2


def _validation_error() -> ValidationError:
    try:
        PredictorResult.model_validate({})
    except ValidationError as err:
        return err
    raise AssertionError("empty PredictorResult should not validate")


def _analyst_name(prompt: str) -> str:
    for key, name in (
        ("indicators", "technical"),
        ("fundamentals", "fundamental"),
        ("news", "sentiment"),
        ("market_context", "context"),
    ):
        if f'"{key}"' in prompt:
            return name
    raise AssertionError("unrecognised analyst prompt")


class FakeLLM:
    """Mock LLM factory: valid analyst reports, configurable predictor behaviour."""

    def __init__(self, predictor_mode: str):
        self.predictor_mode = predictor_mode
        self.predictor_calls = 0

    def generate_structured(self, role, prompt, response_schema, **_):
        if response_schema is AnalystReport:
            return AnalystReport(
                analyst=_analyst_name(prompt),
                stance="bullish",
                confidence=0.7,
                key_points=["Trend is constructive"],
                evidence=["Close above 20-day average"],
                risks=["Earnings surprise could reverse trend"],
                data_gaps=[],
            )

        self.predictor_calls += 1
        if self.predictor_mode == "invalid":
            raise _validation_error()

        quant = json.loads(prompt.split("QUANT BASELINE:\n")[1].split("\n\n")[0])
        up, down = quant["prob_up"] + 0.05, quant["prob_down"] - 0.05
        return PredictorResult(
            prob_up=up,
            prob_flat=quant["prob_flat"],
            prob_down=down,
            expected_return_pct=quant["expected_return_pct"],
            p10_price=quant["p10_price"],
            p50_price=quant["p50_price"],
            p90_price=quant["p90_price"],
            weekly_vol_pct=quant["weekly_vol_pct"],
            adjustment_applied=True,
            adjustments=[
                {
                    "adjustment_type": "prob_up",
                    "previous_value": quant["prob_up"],
                    "new_value": up,
                    "reason": "Constructive trend",
                    "evidence_refs": ["Close above 20-day average"],
                },
                {
                    "adjustment_type": "prob_down",
                    "previous_value": quant["prob_down"],
                    "new_value": down,
                    "reason": "Constructive trend",
                    "evidence_refs": ["Close above 20-day average"],
                },
            ],
            reasoning="Small tilt toward upside given analyst agreement",
            evidence=["Close above 20-day average"],
            risks=["Earnings surprise could reverse trend"],
            quant_baseline_ref=quant,
        )


@pytest.fixture
def synthetic_prices():
    rng = np.random.default_rng(0)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 300)))
    history = SimpleNamespace(data=[SimpleNamespace(close=Decimal(str(c))) for c in closes])
    collector = SimpleNamespace(fetch_history=lambda *a, **k: history)
    with patch("stock_analysis.market.collector.get_price_collector", return_value=collector):
        yield


@pytest.fixture
def initial_state():
    return GraphState(
        symbol="RELIANCE",
        resolved_symbol="RELIANCE.NS",
        technical_indicators_summary={"rsi_14": 55},
        fundamentals_summary={"pe_ratio": 25},
        news_summary={"article_count": 10},
        market_context_summary={"india_vix": 14.5},
    )


def _final_forecast(result) -> dict:
    return result["final_forecast"] if isinstance(result, dict) else result.final_forecast


@pytest.mark.usefixtures("synthetic_prices")
def test_graph_produces_adjusted_final_forecast(initial_state):
    llm = FakeLLM("valid")
    final = _final_forecast(compile_graph(llm).invoke(initial_state))

    forecast = FinalForecast(**final)
    assert forecast.adjustment_applied is True
    assert forecast.fallback_to_quant is False
    assert forecast.critic_passed is True
    assert forecast.revision_count == 0
    assert llm.predictor_calls == 1


@pytest.mark.usefixtures("synthetic_prices")
def test_invalid_predictor_output_revises_then_falls_back_to_quant(initial_state):
    llm = FakeLLM("invalid")
    final = _final_forecast(compile_graph(llm).invoke(initial_state))

    forecast = FinalForecast(**final)
    assert forecast.fallback_to_quant is True
    assert forecast.adjustment_applied is False
    assert forecast.critic_passed is False
    # initial attempt + MAX_REVISIONS revisions, then the loop terminates
    assert llm.predictor_calls == 1 + MAX_REVISIONS
    assert forecast.revision_count == MAX_REVISIONS + 1
    assert forecast.prob_up == forecast.quant_baseline["prob_up"]


@pytest.mark.usefixtures("synthetic_prices")
def test_guardrail_failure_blocks_adjustment_but_still_forecasts(initial_state):
    """A failing LLM degrades every analyst, guardrails fail, quant-only forecast ships."""

    class DownLLM:
        def generate_structured(self, *a, **k):
            raise RuntimeError("LLM unavailable")

    final = _final_forecast(compile_graph(DownLLM()).invoke(initial_state))

    forecast = FinalForecast(**final)
    assert forecast.adjustment_applied is False
    assert forecast.critic_passed is True
    assert forecast.prob_up == forecast.quant_baseline["prob_up"]


class TestRouteAfterRevision:
    def test_passed_goes_to_final(self):
        state = GraphState(symbol="X", resolved_symbol="X", critic_result={"passed": True})
        assert route_after_revision(state) == "final_forecast"

    def test_failed_within_budget_revises(self):
        state = GraphState(
            symbol="X",
            resolved_symbol="X",
            critic_result={"passed": False},
            forecast_revision_count=MAX_REVISIONS,
        )
        assert route_after_revision(state) == "predictor"

    def test_failed_past_budget_terminates(self):
        state = GraphState(
            symbol="X",
            resolved_symbol="X",
            critic_result={"passed": False},
            forecast_revision_count=MAX_REVISIONS + 1,
        )
        assert route_after_revision(state) == "final_forecast"
