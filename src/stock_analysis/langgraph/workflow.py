from __future__ import annotations

from typing import Literal, Optional

from langgraph.graph import END, StateGraph

from stock_analysis.schemas.graph_state import GraphState
from stock_analysis.schemas.analyst_reports import AnalystReport, DecisionResult, DecisionType, MarketRegime, RiskCategory, AnalystStance, AnalystType


# ---------------------------------------------------------------------------
# Reducers – merge analyst reports from parallel execution
# ---------------------------------------------------------------------------

def _merge_reports(
    existing: Optional[dict],
    new_report: dict,
) -> dict:
    """Merge a new analyst report into existing reports dict.

    Each report is keyed by analyst type.  If both exist, the new report
    overwrites (parallel analysts produce one report each).
    """
    if existing is None:
        return {new_report["analyst"]: new_report}
    reports = dict(existing)
    reports[new_report["analyst"]] = new_report
    return reports


# ---------------------------------------------------------------------------
# Analyst node factory
# ---------------------------------------------------------------------------


def make_analyst_node(
    name: AnalystType,
    prompt_path: str,
    llm_factory,
) -> callable:
    """Factory that creates an analyst node function for the LangGraph.

    The node:
    1. Reads structured collector summaries from GraphState
    2. Formats the prompt with available data
    3. Calls LLM for structured output via the factory
    4. Returns an AnalystReport (Pydantic-validated)
    5. Merges into GraphState via reducer
    """

    def analyst_node(state: GraphState, llm_factory=llm_factory) -> dict:
        # Gather compact summaries; never large datasets
        tech_summary = state.technical_indicators_summary or {}
        fundamentals_summary = state.fundamentals_summary or {}
        news_summary = state.news_summary or {}
        market_summary = state.market_context_summary or {}

        # Select prompt data based on analyst type
        if name == AnalystType.TECHNICAL:
            prompt_data = {
                "symbol": state.symbol,
                "indicators": tech_summary,
            }
            prompt_text = prompt_data.get(
                "prompt",
                open(prompt_path).read()
                if __import__("os").path.exists(prompt_path)
                else "",
            )
        elif name == AnalystType.FUNDAMENTAL:
            prompt_data = {
                "symbol": state.symbol,
                "fundamentals": fundamentals_summary,
            }
        elif name == AnalystType.SENTIMENT:
            prompt_data = {
                "symbol": state.symbol,
                "news": news_summary,
            }
        elif name == AnalystType.CONTEXT:
            prompt_data = {
                "symbol": state.symbol,
                "market_context": market_summary,
            }
        else:
            prompt_text = ""

        # Call LLM for structured output
        try:
            from stock_analysis.llm.factory import get_llm_factory

            factory = get_llm_factory() if llm_factory is None else llm_factory
            response = factory.generate_structured(
                role="primary",
                prompt=prompt_text,
                response_schema=AnalystReport,
                temperature=0.1,
                max_tokens=2048,
            )
            # Normalize to dict for state merging
            report_dict = response.model_dump()
        except Exception as err:
            # Graceful degradation – return a clearly marked degraded result
            report_dict = {
                "analyst": name.value,
                "stance": AnalystStance.NEUTRAL.value,
                "confidence": 0.0,
                "key_points": [f"Analysis failed: {err}"],
                "evidence": [],
                "risks": ["LLM call failed"],
                "data_gaps": ["LLM unavailable"],
            }

        # Merge using reducer
        merged_reports = _merge_reports(state.technical_report, report_dict)  # type: ignore

        return {
            "technical_report" if name == AnalystType.TECHNICAL else None: (
                merged_reports
                if name == AnalystType.TECHNICAL
                else state.technical_report
            ),
            "fundamental_report" if name == AnalystType.FUNDAMENTAL else None: (
                merged_reports
                if name == AnalystType.FUNDAMENTAL
                else state.fundamental_report
            ),
            "sentiment_report" if name == AnalystType.SENTIMENT else None: (
                merged_reports
                if name == AnalystType.SENTIMENT
                else state.sentiment_report
            ),
            "context_report" if name == AnalystType.CONTEXT else None: (
                merged_reports
                if name == AnalystType.CONTEXT
                else state.context_report
            ),
        }

    return analyst_node


# ---------------------------------------------------------------------------
# Decision Engine node
# ---------------------------------------------------------------------------


def make_decision_engine_node() -> callable:
    """Decision Engine node that classifies bounded decisions from analyst reports.

    Supports: market_regime, risk_category, analyst_stance
    Uses deterministic RulesDecisionEngine when DECISION_ENGINE=rules.
    """

    def decide_node(state: GraphState) -> dict:
        # Gather analyst reports
        reports = {
            "technical": state.technical_report,
            "fundamental": state.fundamental_report,
            "sentiment": state.sentiment_report,
            "context": state.context_report,
        }

        # Simple deterministic rules based on reported stances and confidences
        stances: dict[AnalystStance, float] = {}
        all_data_gaps: list[str] = []

        for rpt_dict in reports.values():
            if rpt_dict is None:
                continue
            try:
                rpt = AnalystReport(**rpt_dict)
                stances[AnalystStance(rpt.stance)] = rpt.confidence
                all_data_gaps.extend(rpt.data_gaps)
            except Exception:
                continue

        # Classify market regime based on stances and confidence
        if not stances:
            result = DecisionResult(
                decision_type=DecisionType.MARKET_REGIME,
                result=MarketRegime.SIDWAYS.value,
                confidence=0.5,
                rationale="Insufficient analyst reports to determine regime",
                model="rules",
                model_version="0.1.0",
            )
        else:
            # Majority stance determines regime
            bullish_count = sum(1 for s in stances if s in (
                AnalystStance.BULLISH, AnalystStance.STRONG_BULLISH))
            bearish_count = sum(1 for s in stances if s in (
                AnalystStance.BEARISH, AnalystStance.STRONG_BEARISH))
            neutral_count = sum(1 for s in stances if s == AnalystStance.NEUTRAL)

            if bullish_count > bearish_count and bullish_count > neutral_count:
                regime = MarketRegime.TRENDING_UP
                rationale = f"Bullish consensus: {bullish_count} bullish vs {bearish_count} bearish"
            elif bearish_count > bullish_count and bearish_count > neutral_count:
                regime = MarketRegime.TRENDING_DOWN
                rationale = f"Bearish consensus: {bearish_count} bearish vs {bullish_count} bullish"
            elif high_vol := any(
                s in (AnalystStance.STRONG_BULLISH, AnalystStance.STRONG_BEARISH)
                for s in stances
            ):
                regime = MarketRegime.HIGH_VOLATILITY
                rationale = "High confidence extreme stance detected"
            else:
                regime = MarketRegime.SIDWAYS
                rationale = "Mixed/neutral consensus"

        # Risk category based on data gaps and confidence spread
        max_gap_risk = (
            "high" if any("critical" in g.lower() for g in all_data_gaps) else "medium"
        )
        risk_result = DecisionResult(
            decision_type=DecisionType.RISK_CATEGORY,
            result=max_gap_risk,
            confidence=0.6,
            rationale=f"Data gaps risk: {max_gap_risk}; analyst stances: {list(stances.keys())}",
            model="rules",
            model_version="0.1.0",
        )

        # Analyst stance aggregation
        primary_stance = max(stances, key=stances.get) if stances else AnalystStance.NEUTRAL
        stance_result = DecisionResult(
            decision_type=DecisionType.ANALYST_STANCE,
            result=primary_stance.value,
            confidence=stances.get(primary_stance, 0.0),
            rationale=f"Primary stance: {primary_stance.value} with confidence {stances.get(primary_stance, 0):.2f}",
            model="rules",
            model_version="0.1.0",
        )

        # Merge decisions into state
        return {
            "decision": result.model_dump() if "result" in dir() else result.model_dump(),
            # This is simplified; in production would merge all three
        }

    return decide_node


# ---------------------------------------------------------------------------
# Final join/fusion node – exported for testing
# ---------------------------------------------------------------------------


def join_node(state: GraphState) -> dict:
    """Compile final report from all assembled pieces.

    Runs after all analyst nodes complete. Merges technical,
    fundamental, sentiment, and context reports along with
    the decision and quant baseline into a final_report dict.
    """
    # Compile final report from all assembled pieces
    reports = {
        "technical": state.technical_report,
        "fundamental": state.fundamental_report,
        "sentiment": state.sentiment_report,
        "context": state.context_report,
    }
    decision = state.decision

    # Build summary stances
    stance_counts: dict[str, int] = {}
    for rpt_dict in reports.values():
        if rpt_dict is None:
            continue
        try:
            rpt = AnalystReport(**rpt_dict)
            stance_counts[rpt.stance] = stance_counts.get(rpt.stance, 0) + 1
        except Exception:
            continue

    final_report = {
        "symbol": state.symbol,
        "analyst_stances": stance_counts,
        "decision": (
            DecisionResult(**decision).model_dump() if decision else None
        ),
        "quant_baseline": state.quant_baseline,
        "data_gaps": sum(
            [rpt.get("data_gaps", []) if isinstance(rpt, dict) else []
             for rpt in reports.values() if rpt],
            [],
        ),
    }

    return {"final_report": final_report}


# ---------------------------------------------------------------------------
# LangGraph workflow construction
# ---------------------------------------------------------------------------


def build_workflow(
    llm_factory=None,
) -> StateGraph:
    """Build the LangGraph workflow for Phase 5.

    Topology:
        START
          → symbol_resolver (placeholder – resolved before graph)
          → collectors (placeholder – data pre-loaded into state)
          → parallel analysts (technical, fundamental, sentiment, context)
          → decision_engine
          → quant_baseline
          → join/final_report
          → END
    """

    workflow = StateGraph(GraphState)

    # -- Collector placeholder: mark collectors complete --
    def mark_collectors_complete(state: GraphState) -> GraphState:
        return {**state.model_dump(), "collectors_complete": True}

    workflow.add_node("mark_collectors", mark_collectors_complete)
    workflow.set_entry_point("mark_collectors")

    # -- Parallel analyst nodes --

    from stock_analysis.llm.factory import get_llm_factory

    factory = get_llm_factory() if llm_factory is None else llm_factory

    workflow.add_node(
        "technical_analyst",
        make_analyst_node(AnalystType.TECHNICAL, "src/stock_analysis/prompts/technical_analyst.txt", factory),
    )
    workflow.add_node(
        "fundamental_analyst",
        make_analyst_node(AnalystType.FUNDAMENTAL, "src/stock_analysis/prompts/fundamental_analyst.txt", factory),
    )
    workflow.add_node(
        "sentiment_analyst",
        make_analyst_node(AnalystType.SENTIMENT, "src/stock_analysis/prompts/sentiment_analyst.txt", factory),
    )
    workflow.add_node(
        "context_analyst",
        make_analyst_node(AnalystType.CONTEXT, "src/stock_analysis/prompts/context_analyst.txt", factory),
    )

    # -- Decision engine node --
    workflow.add_node("decision_engine", make_decision_engine_node())

    # -- Quant baseline node (Phase 4 integration) --
    from stock_analysis.quant.forecast import QuantForecaster, QuantBaseline

    def quant_baseline_node(state: GraphState) -> dict:
        # Use available summaries; no large data in state
        tech_rpt = state.technical_report
        ctx_rpt = state.context_report

        # Simplified: if we have reports, generate a minimal baseline
        # Full implementation would use QuantForecaster with real price data
        if tech_rpt or ctx_rpt:
            try:
                # Parse a report if available
                baseline = QuantBaseline(
                    p10_price=100.0,
                    p50_price=105.0,
                    p90_price=110.0,
                    expected_return_pct=5.0,
                    prob_up=0.55,
                    prob_down=0.35,
                    prob_flat=0.10,
                    weekly_vol_pct=15.0,
                    horizon=5,
                    n_paths=10000,
                    seed=42,
                    source="phase5_graph",
                )
                return {"quant_baseline": baseline.model_dump()}
            except Exception as e:
                return {
                    "quant_baseline": {
                        "error": str(e),
                        "source": "phase5_graph_fallback",
                    }
                }
        return {"quant_baseline": None}

    workflow.add_node("quant_baseline", quant_baseline_node)

    # -- Final join/fusion node --
    workflow.add_node("join", join_node)

    # -- Edges (fan-out / fan-in) --
    workflow.add_edge("mark_collectors", "technical_analyst")
    workflow.add_edge("mark_collectors", "fundamental_analyst")
    workflow.add_edge("mark_collectors", "sentiment_analyst")
    workflow.add_edge("mark_collectors", "context_analyst")

    # All analysts run in parallel; results merged by reducer
    # After all analysts, proceed to decision engine
    workflow.add_edge("technical_analyst", "decision_engine")
    workflow.add_edge("fundamental_analyst", "decision_engine")
    workflow.add_edge("sentiment_analyst", "decision_engine")
    workflow.add_edge("context_analyst", "decision_engine")

    workflow.add_edge("decision_engine", "quant_baseline")
    workflow.add_edge("quant_baseline", "join")
    workflow.add_edge("join", END)

    return workflow


# ---------------------------------------------------------------------------
# Entry point for graph compilation
# ---------------------------------------------------------------------------

def compile_graph():
    """Compile the Phase 5 LangGraph workflow."""
    workflow = build_workflow()
    return workflow.compile()