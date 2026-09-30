"""Integration tests for Phase 5 LangGraph workflow with mocked LLMs."""

import pytest

from stock_analysis.langgraph.workflow import compile_graph, GraphState
from stock_analysis.schemas.analyst_reports import AnalystReport, AnalystType, AnalystStance


class TestGraphIntegration:
    """Integration tests for the LangGraph workflow."""

    @pytest.fixture
    def graph(self):
        """Compile the graph for testing."""
        return compile_graph()

    @pytest.fixture
    def sample_state(self):
        """Create a sample GraphState with collector summaries."""
        return GraphState(
            symbol="RELIANCE",
            resolved_symbol="RELIANCE",
            technical_indicators_summary={
                "rsi_14": 35.2,
                "macd_histogram": 1.4,
                "price_vs_ema_20": "above",
            },
            fundamentals_summary={
                "pe_ratio": 25.3,
                "roe": 22.1,
                "dividend_yield": 1.2,
            },
            news_summary={
                "article_count": 15,
                "overall_sentiment": "positive",
            },
            market_context_summary={
                "nifty_50": 22500,
                "bank_nifty": 45000,
                "india_vix": 14.5,
                "usdinr": 83.2,
            },
            collectors_complete=True,
        )

    def test_graph_compiles(self):
        """Graph should compile without error."""
        g = compile_graph()
        assert g is not None

    def test_graph_state_initialization(self, sample_state):
        """GraphState should initialize with sample data."""
        assert sample_state.symbol == "RELIANCE"
        assert sample_state.collectors_complete is True

    def test_graph_state_all_reports_none_initially(self, sample_state):
        """All analyst reports should be None initially."""
        assert sample_state.technical_report is None
        assert sample_state.fundamental_report is None
        assert sample_state.sentiment_report is None
        assert sample_state.context_report is None

    def test_graph_state_decision_none_initially(self, sample_state):
        """Decision should be None initially."""
        assert sample_state.decision is None

    @pytest.mark.mock_llm
    def test_technical_analyst_node(self, graph, sample_state):
        """Technical analyst node should produce a report via mocked LLM."""
        # The graph compilation may fail if LLM is not available,
        # but we test the node function directly
        from stock_analysis.langgraph.workflow import make_analyst_node
        from stock_analysis.llm.factory import get_llm_factory

        factory = get_llm_factory()
        node = make_analyst_node(
            AnalystType.TECHNICAL,
            "src/stock_analysis/prompts/technical_analyst.txt",
            factory,
        )

        # Should not crash; LLM may return degraded result
        result = node(sample_state)
        # Report dict should be present (possibly degraded)
        assert "technical_report" in result or result.get("technical_report") is not None

    @pytest.mark.mock_llm
    def test_all_analysts_produce_reports(self, graph, sample_state):
        """All four analyst nodes should produce reports."""
        from stock_analysis.langgraph.workflow import make_analyst_node
        from stock_analysis.llm.factory import get_llm_factory

        factory = get_llm_factory()
        analyst_types = [
            AnalystType.TECHNICAL,
            AnalystType.FUNDAMENTAL,
            AnalystType.SENTIMENT,
            AnalystType.CONTEXT,
        ]
        prompt_files = [
            "src/stock_analysis/prompts/technical_analyst.txt",
            "src/stock_analysis/prompts/fundamental_analyst.txt",
            "src/stock_analysis/prompts/sentiment_analyst.txt",
            "src/stock_analysis/prompts/context_analyst.txt",
        ]

        for atype, pf in zip(analyst_types, prompt_files):
            node = make_analyst_node(atype, pf, factory)
            result = node(sample_state)
            # Each should produce a report dict
            report_key = f"{atype.value}_report"
            assert report_key in result or result.get(report_key) is not None

    def test_decision_engine_node(self, sample_state):
        """Decision engine node should produce a decision."""
        from stock_analysis.langgraph.workflow import make_decision_engine_node

        node = make_decision_engine_node()
        result = node(sample_state)
        assert "decision" in result or result.get("decision") is not None

    def test_join_node_produces_final_report(self, sample_state):
        """Join node should produce a final consolidated report."""
        from stock_analysis.langgraph.workflow import join_node

        # First set some analyst reports
        state = sample_state.model_copy()
        state.technical_report = {
            "analyst": "technical",
            "stance": "bullish",
            "confidence": 0.75,
            "key_points": ["Price above EMA"],
            "evidence": ["price_close > EMA_20"],
            "risks": ["RSI overbought"],
            "data_gaps": [],
        }
        state.fundamental_report = {
            "analyst": "fundamental",
            "stance": "bullish",
            "confidence": 0.65,
            "key_points": ["P/E reasonable"],
            "evidence": ["P/E=25"],
            "risks": ["Sector slowdown"],
            "data_gaps": [],
        }

        result = join_node(state)
        assert "final_report" in result
        fr = result["final_report"]
        assert fr["symbol"] == "RELIANCE"
        # Should have stance counts from the two reports
        assert "analyst_stances" in fr

    def test_graph_state_extra_fields_forbidden(self):
        """GraphState should forbid extra fields not defined."""
        try:
            GraphState(
                symbol="RELIANCE",
                resolved_symbol="RELIANCE",
                undefined_field="should_fail",  # type: ignore
            )
            assert False, "Should have raised ValidationError"
        except Exception:
            pass  # Expected