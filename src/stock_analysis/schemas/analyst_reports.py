from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

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
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0.0-1.0")
    key_points: List[str] = Field(
        max_items=5,
        description="Max 5 concise technical/fundamental/sentiment/context observations",
    )
    evidence: List[str] = Field(
        max_items=10,
        description="Supporting data points, price levels, indicator values",
    )
    risks: List[str] = Field(
        max_items=3,
        description="Max 3 identified risks specific to analyst type",
    )
    data_gaps: List[str] = Field(
        default_factory=list,
        description="Missing required data needed for analysis",
    )

    @validator("confidence")
    def confidence_bounds(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return v


class DecisionResult(BaseModel):
    decision_type: DecisionType
    result: str
    confidence: float = Field(ge=0.0, le=1.0, description="Decision confidence 0.0-1.0")
    rationale: str = Field(description="Human-readable rationale for the decision")
    model: str = Field(default="rules", description="Decision engine model identifier")
    model_version: str = Field(default="0.1.0", description="Decision engine model version")


class DecisionEngineInterface(BaseModel):
    """Provider-independent interface for decision engines.

    Subclasses must implement the `decide` method.
    """

    name: str = "decision_engine"

    def decide(
        self,
        decision_type: DecisionType,
        context: dict,
        options: dict | None = None,
    ) -> DecisionResult:
        """Make a bounded classification decision.

        Args:
            decision_type: Type of decision to make
            context: Structured context from analysts (no large datasets)
            options: Optional decision-specific options

        Returns:
            DecisionResult with the classification
        """
        raise NotImplementedError("Subclasses must implement decide()")


class RulesDecisionEngine(DecisionEngineInterface):
    """Deterministic rules-based decision engine.

    Uses analyst reports and structured data to classify
    bounded decisions without LLM calls.
    """

    name: str = "rules"

    def decide(
        self,
        decision_type: DecisionType,
        context: dict,
        options: dict | None = None,
    ) -> DecisionResult:
        """Classify decision using deterministic rules from analyst reports."""

        # Gather analyst stances and data gaps from context
        reports = context.get("analyst_reports", [])
        all_gaps: list[str] = []
        stances: dict[AnalystStance, float] = {}

        for rpt in reports:
            try:
                a_rpt = AnalystReport(**rpt)
                stances[AnalystStance(a_rpt.stance)] = a_rpt.confidence
                all_gaps.extend(a_rpt.data_gaps)
            except Exception:
                continue

        # -- Classify based on decision_type --

        primary_confidence = 0.0

        try:
            if decision_type == DecisionType.MARKET_REGIME:
                bullish = sum(
                    1 for s in stances
                    if s in (AnalystStance.BULLISH, AnalystStance.STRONG_BULLISH)
                )
                bearish = sum(
                    1 for s in stances
                    if s in (AnalystStance.BEARISH, AnalystStance.STRONG_BEARISH)
                )
                neutral = sum(
                    1 for s in stances if s == AnalystStance.NEUTRAL
                )

                if bullish > bearish and bullish > neutral:
                    result = MarketRegime.TRENDING_UP.value
                    rationale = f"Bullish consensus ({bullish} bullish vs {bearish} bearish)"
                elif bearish > bullish and bearish > neutral:
                    result = MarketRegime.TRENDING_DOWN.value
                    rationale = f"Bearish consensus ({bearish} bearish vs {bullish} bullish)"
                elif any(
                    s in (AnalystStance.STRONG_BULLISH, AnalystStance.STRONG_BEARISH)
                    for s in stances
                ):
                    result = MarketRegime.HIGH_VOLATILITY.value
                    rationale = "High-confidence extreme stance detected"
                else:
                    result = MarketRegime.SIDWAYS.value
                    rationale = "Mixed/neutral consensus"

            elif decision_type == DecisionType.RISK_CATEGORY:
                # High risk if critical data gaps exist
                critical_gaps = [
                    g for g in all_gaps if "critical" in g.lower()
                ]
                if critical_gaps:
                    result = RiskCategory.HIGH.value
                    rationale = f"Critical data gaps detected: {critical_gaps}"
                elif any(
                    s in (AnalystStance.STRONG_BULLISH, AnalystStance.STRONG_BEARISH)
                    for s in stances
                ):
                    result = RiskCategory.MEDIUM.value
                    rationale = "Extreme stances with moderate data"
                else:
                    result = RiskCategory.LOW.value
                    rationale = "Consensus stance with sufficient data"

            elif decision_type == DecisionType.ANALYST_STANCE:
                if not stances:
                    result = AnalystStance.NEUTRAL.value
                    rationale = "No analyst reports available"
                    primary_confidence = 0.0
                else:
                    primary = max(stances, key=stances.get)
                    result = primary.value
                    primary_confidence = stances[primary]
                    rationale = f"Primary stance: {primary.value} (confidence={primary_confidence:.2f})"

            else:
                result = MarketRegime.SIDWAYS.value
                rationale = f"Unsupported decision type: {decision_type}"
                primary_confidence = 0.0

        except (ValueError, Exception):
            # Unsupported decision type or other error - degrade gracefully
            result = MarketRegime.SIDWAYS.value
            rationale = f"Unsupported decision type: {decision_type}"
            primary_confidence = 0.0
            decision_type = DecisionType.MARKET_REGIME  # use valid enum for pydantic

        return DecisionResult(
            decision_type=effective_decision_type if 'effective_decision_type' in dir() else DecisionType.MARKET_REGIME,
            result=result,
            confidence=primary_confidence,
            rationale=rationale,
            model=self.name,
            model_version="0.1.0",
        )

    def classify_market_regime(
        self,
        reports: list[dict],
    ) -> tuple[str, str]:
        """Convenience: classify market regime from analyst reports.

        Returns (regime_string, rationale).
        """
        return self.decide(
            DecisionType.MARKET_REGIME,
            {"analyst_reports": reports},
        ).model_dump().values()


class JevDecisionEngine(DecisionEngineInterface):
    """Decision engine using Jev provider.

    Placeholder - not fully implemented for Phase 5.
    """

    name: str = "jev"

    def decide(
        self,
        decision_type: DecisionType,
        context: dict,
        options: dict | None = None,
    ) -> DecisionResult:
        # Placeholder: degrade gracefully
        return DecisionResult(
            decision_type=decision_type,
            result="not_implemented",
            confidence=0.0,
            rationale="JevDecisionEngine not configured for Phase 5",
            model=self.name,
            model_version="0.1.0",
        )


class LayaDecisionEngine(DecisionEngineInterface):
    """Decision engine using Laya provider.

    Placeholder - not fully implemented for Phase 5.
    """

    name: str = "laya"

    def decide(
        self,
        decision_type: DecisionType,
        context: dict,
        options: dict | None = None,
    ) -> DecisionResult:
        # Placeholder: degrade gracefully
        return DecisionResult(
            decision_type=decision_type,
            result="not_implemented",
            confidence=0.0,
            rationale="LayaDecisionEngine not configured for Phase 5",
            model=self.name,
            model_version="0.1.0",
        )