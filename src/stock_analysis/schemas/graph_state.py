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


class DataGapSeverity(str, Enum):
    """Severity levels for data gaps."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DataGap(BaseModel):
    """Structured data gap with severity."""
    description: str = Field(min_length=1, max_length=500)
    severity: DataGapSeverity = DataGapSeverity.MEDIUM
    source_analyst: Optional[AnalystType] = None
    required_for: Optional[str] = None


class AnalystReport(BaseModel):
    analyst: AnalystType
    stance: AnalystStance
    confidence: float = Field(ge=0.0, le=1.0)
    key_points: List[str] = Field(max_items=5, description="Max 5 concise observations")
    evidence: List[str] = Field(max_items=10, description="Supporting data points")
    risks: List[str] = Field(max_items=3, description="Max 3 identified risks")
    data_gaps: List[DataGap] = Field(default_factory=list, description="Missing required data")

    @validator("confidence")
    def confidence_bounds(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return v

    @validator("key_points")
    def key_points_conciseness(cls, v):
        """Validate each key point is concise (<= 200 chars) and non-empty."""
        for i, point in enumerate(v):
            if not point or not point.strip():
                raise ValueError(f"key_points[{i}] must not be empty")
            if len(point) > 200:
                raise ValueError(f"key_points[{i}] exceeds 200 character limit")
        return v

    @validator("evidence")
    def evidence_quality(cls, v):
        """Validate evidence items are non-empty and reasonably sized."""
        for i, item in enumerate(v):
            if not item or not item.strip():
                raise ValueError(f"evidence[{i}] must not be empty")
            if len(item) > 300:
                raise ValueError(f"evidence[{i}] exceeds 300 character limit")
        return v

    @validator("risks")
    def risks_specificity(cls, v):
        """Validate risks are specific (not generic) and non-empty."""
        generic_risks = {"market risk", "volatility", "uncertainty", "risk", "general risk"}
        for i, risk in enumerate(v):
            if not risk or not risk.strip():
                raise ValueError(f"risks[{i}] must not be empty")
            if len(risk) < 10:
                raise ValueError(f"risks[{i}] too short (min 10 chars): be specific")
            if risk.lower().strip() in generic_risks:
                raise ValueError(f"risks[{i}] too generic: '{risk}' - be specific")
        return v

    @validator("data_gaps", pre=True)
    def data_gaps_structure(cls, v):
        """Validate and normalize data gaps - accept strings, dicts, or DataGap objects."""
        if not isinstance(v, list):
            return v
        normalized = []
        for i, gap in enumerate(v):
            if isinstance(gap, str):
                # Convert string to DataGap with inferred severity
                severity = DataGapSeverity.MEDIUM
                lower_gap = gap.lower()
                if any(kw in lower_gap for kw in ["critical", "essential", "required", "missing price", "no data"]):
                    severity = DataGapSeverity.CRITICAL
                elif any(kw in lower_gap for kw in ["limited", "incomplete", "partial"]):
                    severity = DataGapSeverity.HIGH
                normalized.append(DataGap(description=gap, severity=severity))
            elif isinstance(gap, dict):
                if "description" not in gap:
                    raise ValueError(f"data_gaps[{i}] must have 'description' field")
                normalized.append(DataGap(**gap))
            elif isinstance(gap, DataGap):
                normalized.append(gap)
            else:
                raise ValueError(f"data_gaps[{i}] must be string, dict, or DataGap object")
        return normalized

    @validator("stance", "confidence")
    def stance_confidence_consistency(cls, v, values):
        """Validate that stance matches confidence level."""
        if "stance" in values and "confidence" in values:
            stance = values["stance"]
            confidence = values["confidence"]
            if isinstance(stance, AnalystStance):
                if stance in (AnalystStance.STRONG_BULLISH, AnalystStance.STRONG_BEARISH):
                    if confidence < 0.7:
                        raise ValueError(
                            f"Strong stance ({stance.value}) requires confidence >= 0.7, got {confidence}"
                        )
                elif stance in (AnalystStance.BULLISH, AnalystStance.BEARISH):
                    if confidence < 0.5:
                        raise ValueError(
                            f"Directional stance ({stance.value}) requires confidence >= 0.5, got {confidence}"
                        )
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