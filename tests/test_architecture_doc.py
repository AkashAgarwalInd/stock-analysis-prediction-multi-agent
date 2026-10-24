"""docs/architecture.md must name every node of the forecast and review graphs."""

from pathlib import Path

from stock_analysis.langgraph.review_graph import REVIEW_STEPS
from stock_analysis.langgraph.workflow import compile_graph
from tests.forecast_helpers import FakeLLM

ARCHITECTURE_DOC = Path(__file__).resolve().parent.parent / "docs" / "architecture.md"


def test_forecast_graph_nodes_are_documented():
    doc = ARCHITECTURE_DOC.read_text(encoding="utf-8")
    nodes = set(compile_graph(FakeLLM()).get_graph().nodes) - {"__start__", "__end__"}
    missing = sorted(node for node in nodes if f"`{node}`" not in doc)
    assert not missing, f"Add these forecast graph nodes to docs/architecture.md: {missing}"


def test_review_graph_steps_are_documented():
    doc = ARCHITECTURE_DOC.read_text(encoding="utf-8")
    missing = [step for step in REVIEW_STEPS if f"`{step}`" not in doc]
    assert not missing, f"Add these review graph steps to docs/architecture.md: {missing}"
