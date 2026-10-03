"""Shared forecast test helpers: fake LLMs, fixed timestamps and a graph runner."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

from stock_analysis.langgraph.workflow import compile_graph
from stock_analysis.schemas.analyst_reports import AnalystReport
from stock_analysis.schemas.forecast_pipeline import PredictorResult
from stock_analysis.schemas.graph_state import GraphState

ALEMBIC_DIR = Path(__file__).resolve().parents[1] / "alembic"
LAST_BAR = date(2026, 9, 29)  # Tuesday; 2026-10-02 (Gandhi Jayanti) is an NSE holiday
MADE_AT = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)


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
    """Valid analyst reports; predictor tilts prob_up by +0.05 or emits invalid output."""

    def __init__(self, predictor_mode: str = "valid"):
        self.predictor_mode = predictor_mode
        self.predictor_prompts: list[str] = []

    def generate_structured(self, role, prompt, response_schema, **_):
        if response_schema is AnalystReport:
            return AnalystReport(
                analyst=_analyst_name(prompt),
                stance="bullish",
                confidence=0.7,
                key_points=["Trend is constructive"],
                evidence=["Close above 20-day average"],
                risks=["Earnings surprise could reverse trend"],
                data_gaps=[{"description": "No intraday data", "severity": "low"}],
            )
        self.predictor_prompts.append(prompt)
        if self.predictor_mode == "invalid":
            PredictorResult.model_validate({})  # raises ValidationError
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


class DownLLM:
    """Every LLM call fails: analysts degrade and the guardrails block adjustment."""

    def generate_structured(self, *a, **k):
        raise RuntimeError("LLM unavailable")


def run_graph(llm, initial_state, store=None, memory_store=None) -> GraphState:
    """Invoke the full graph and return the final state as a ``GraphState``."""
    return GraphState(**compile_graph(llm, store, memory_store).invoke(initial_state))
