from typing import Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

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
from stock_analysis.schemas.snapshot import ForecastSource

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
    company_name: Optional[str] = None
    collectors_complete: bool = False
    # Point-in-time clock: a historical run (backtest) sets this to when the forecast
    # is made, so memory, calibration and the snapshot see that time instead of now.
    run_at: Optional[AwareDatetime] = None
    # Analysts switched off on purpose (name -> reason), e.g. no point-in-time data
    # in a backtest. They get a deterministic placeholder report, no LLM call, and are
    # left out of guardrail coverage/consensus and the decision engine.
    disabled_analysts: dict[str, str] = Field(default_factory=dict)
    # Recorded on the snapshot: "backtest" for a simulated week, "live" otherwise
    forecast_source: ForecastSource = "live"

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
    risk_decision: Optional[DecisionResult] = None
    adjustment_gate_decision: Optional[DecisionResult] = None

    # Forecast pipeline outputs
    price_snapshot: Optional[dict] = None  # fingerprint of the price history used
    quant_baseline: Optional[dict] = None
    quant_daily_path: Optional[list[dict]] = None
    # Shadow forecast: the quant baseline before calibration, and the calibration applied
    quant_baseline_uncalibrated: Optional[dict] = None
    calibration: Optional[dict] = None
    predictor_result: Optional[dict] = None
    critic_result: Optional[dict] = None
    forecast_revision_count: int = 0
    final_forecast: Optional[dict] = None
    final_report: Optional[dict] = None

    # Phase 7 snapshot outputs (the snapshot itself lives in SQLite, not state)
    forecast_id: Optional[str] = None
    data_snapshot_id: Optional[str] = None
    snapshot_persisted: bool = False
    forecast_report: Optional[str] = None

    # Plan.md Phase 16: review of matured forecasts run before this forecast (PreRunReview)
    review_summary: Optional[dict] = None
    # Phase 7 memory (Plan.md): prior context loaded at the start, insight written at the end
    memory_context: Optional[dict] = None
    memory_written: bool = False
