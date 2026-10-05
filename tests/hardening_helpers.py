"""A stand-in Gemini model for tests that exercise the real ``LLMFactory`` (retries, rate
limits, usage tracking) end to end."""

from types import SimpleNamespace

from stock_analysis.llm import LLMFactory, LLMModel
from stock_analysis.schemas.analyst_reports import AnalystReport
from stock_analysis.schemas.forecast_pipeline import PredictorResult
from stock_analysis.schemas.learning import PostmortemDiagnosis
from tests.backtest_helpers import BacktestLLM

# Matched by the schema title the factory appends to the prompt
SCHEMAS = (PostmortemDiagnosis, PredictorResult, AnalystReport)


class GeminiStub:
    """Answers with ``BacktestLLM``'s output for the schema named in the prompt and reports
    token usage like the Gemini SDK."""

    def __init__(self, name: str = "gemini-stub"):
        self.model_name = name
        self.fake = BacktestLLM()

    def generate_content(self, prompt, generation_config=None, request_options=None):
        schema = next(s for s in SCHEMAS if f'"title": "{s.__name__}"' in prompt)
        result = self.fake.generate_structured(LLMModel.PRIMARY, prompt, schema)
        return SimpleNamespace(
            text=result.model_dump_json(),
            usage_metadata=SimpleNamespace(
                prompt_token_count=len(prompt) // 4, candidates_token_count=50
            ),
        )


def stub_factory() -> LLMFactory:
    """A real ``LLMFactory`` whose every role is a ``GeminiStub`` (no backoff sleeps)."""
    factory = LLMFactory(sleep=lambda _s: None)
    factory._models = {role: GeminiStub() for role in LLMModel}
    factory._initialized = True
    return factory
