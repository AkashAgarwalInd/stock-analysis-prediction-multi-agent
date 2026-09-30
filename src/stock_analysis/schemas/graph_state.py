from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class AnalystType(str, Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    CONTEXT = "context"


class AnalystStance(str, Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


class DecisionType(str, Enum):
    MARKET_REGIME = "market_regime"
    RISK_CATEGORY = "risk_category"
    ANALYST_STANCE = "analyst_stance"


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    MARKET_STRESS = "market_stress"
    EVENT_RISK = "event_risk"
    MIXED = "mixed"


class RiskCategory(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AnalystReport(BaseModel):
    analyst: AnalystType
    stance: AnalystStance
    confidence: float = Field(ge=0.0, le=1.0)
    key_points: List[str] = Field(max_items=5, description="Max 5 concise observations")
    evidence: List[str] = Field(max_items=10, description="Supporting data points")
    risks: List[str] = Field(max_items=3, description="Max 3 identified risks")
    data_gaps: List[str] = Field(default_factory=list, description="Missing required data")

    @validator("confidence")
    def confidence_bounds(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return v


class DecisionResult(BaseModel):
    decision_type: DecisionType
    result: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    model: str = "rules"
    model_version: str = "0.1.0"


class GraphState(BaseModel):
    """Compact structured state for the LangGraph workflow.

    Large datasets (OHLCV, full news, fundamentals) are kept OUT of state.
    Only references, IDs, summaries, and structured outputs are stored.
    """

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

    # Decision engine output
    decision: Optional[DecisionResult] = None

    # Final outputs
    quant_baseline: Optional[dict] = None
    final_report: Optional[dict] = None

    class Config:
        extra = "forbid"