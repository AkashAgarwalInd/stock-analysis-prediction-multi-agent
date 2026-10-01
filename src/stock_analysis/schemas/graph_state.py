from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# Single source of truth for analyst/decision types lives in analyst_reports.
# Re-exported here so existing ``graph_state`` imports keep working.
from stock_analysis.schemas.analyst_reports import (
    AdjustmentGateDecision,
    AnalystReport,
    AnalystStance,
    AnalystType,
    DataGap,
    DataGapSeverity,
    DecisionResult,
    DecisionType,
    MarketRegime,
    RiskCategory,
)

__all__ = [
    "AdjustmentGateDecision",
    "AnalystReport",
    "AnalystStance",
    "AnalystType",
    "DataGap",
    "DataGapSeverity",
    "DecisionResult",
    "DecisionType",
    "GraphState",
    "MarketRegime",
    "RiskCategory",
]


class GraphState(BaseModel):
    """Compact structured state for the LangGraph workflow.

    Large datasets (OHLCV, full news, fundamentals) are kept OUT of state.
    Only references, IDs, summaries, and structured outputs are stored.
    """

    model_config = ConfigDict(extra="forbid")

    symbol: str
    resolved_symbol: str
    collectors_complete: bool = False

    # Collector outputs (references/IDs/summaries, not large data)
    technical_indicators_summary: Optional[dict] = None
    fundamentals_summary: Optional[dict] = None
    news_summary: Optional[dict] = None
    market_context_summary: Optional[dict] = None

    # Analyst reports
    technical_report: Optional[dict] = None
    fundamental_report: Optional[dict] = None
    sentiment_report: Optional[dict] = None
    context_report: Optional[dict] = None

    # Guardrail output
    guardrail_violations: list[dict] = Field(default_factory=list)
    guardrail_warnings: list[dict] = Field(default_factory=list)

    # Decision engine output
    decision: Optional[DecisionResult] = None
    adjustment_gate_decision: Optional[DecisionResult] = None

    # Forecast pipeline outputs
    quant_baseline: Optional[dict] = None
    predictor_result: Optional[dict] = None
    critic_result: Optional[dict] = None
    forecast_revision_count: int = 0
    final_forecast: Optional[dict] = None
    final_report: Optional[dict] = None
