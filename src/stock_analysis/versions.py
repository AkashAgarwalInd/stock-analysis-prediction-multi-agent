"""Model and prompt version metadata recorded with every forecast snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

from stock_analysis import __version__
from stock_analysis.config.settings import get_settings
from stock_analysis.quant.forecast import EWMAVolatility, QuantForecaster
from stock_analysis.schemas.analyst_reports import AnalystReport
from stock_analysis.schemas.forecast_pipeline import PredictorResult
from stock_analysis.schemas.learning import PostmortemDiagnosis

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# Graph node name -> prompt file in PROMPTS_DIR
ANALYST_PROMPT_FILES = {
    "technical_analyst": "technical_analyst.txt",
    "fundamental_analyst": "fundamental_analyst.txt",
    "sentiment_analyst": "sentiment_analyst.txt",
    "context_analyst": "context_analyst.txt",
}
POSTMORTEM_PROMPT_FILE = "postmortem.txt"

# The predictor prompt is assembled in code (workflow.build_predictor_prompt);
# bump this whenever that function's wording or constraints change.
PREDICTOR_PROMPT_VERSION = "4"
# Bump when the deterministic critic's checks or tolerances change.
CRITIC_VERSION = "1"
# Bump when the report layout changes.
REPORT_WRITER_VERSION = "4"
# Bump when ForecastInsight distillation (schemas/memory.py) changes.
MEMORY_WRITER_VERSION = "1"
# Bump when the calibration estimator, shrinkage or eligibility rules change
# (learning/calibration.py).
CALIBRATOR_VERSION = "1"

# Quant baseline settings used by the graph's quant_baseline node.
QUANT_SEED = 42
QUANT_N_PATHS = 10_000
QUANT_HISTORY_PERIOD = "2y"


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _schema_digest(schema: type[BaseModel]) -> str:
    # Hash the exact text LLMFactory.generate_structured appends to the prompt
    text = json.dumps(schema.model_json_schema())
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()[:16]


def get_prompt_versions() -> dict[str, str]:
    """Versions of everything sent to the LLM: prompt files, predictor prompt, output schemas."""
    versions = {
        node: _file_digest(PROMPTS_DIR / filename)
        for node, filename in ANALYST_PROMPT_FILES.items()
    }
    versions["predictor"] = PREDICTOR_PROMPT_VERSION
    versions["analyst_response_schema"] = _schema_digest(AnalystReport)
    versions["predictor_response_schema"] = _schema_digest(PredictorResult)
    return versions


def get_postmortem_version() -> str:
    """Version of the postmortem prompt file and its response schema (content hashes)."""
    return (
        f"prompt={_file_digest(PROMPTS_DIR / POSTMORTEM_PROMPT_FILE)},"
        f"schema={_schema_digest(PostmortemDiagnosis)}"
    )


def get_model_versions(quant_method: str) -> dict[str, str]:
    """Configured LLM model IDs and the versions of the deterministic components."""
    settings = get_settings()
    return {
        "app": __version__,
        "llm_primary": settings.gemini_model_primary,
        "llm_critic": settings.gemini_model_critic,
        "llm_fallback": settings.gemini_model_fallback,
        "quant": (
            f"{quant_method}(seed={QUANT_SEED},n_paths={QUANT_N_PATHS},"
            f"ewma_lambda={EWMAVolatility().lambda_},"
            f"flat_threshold_pct={QuantForecaster.FLAT_THRESHOLD_PCT},"
            f"history={QUANT_HISTORY_PERIOD})"
        ),
        "critic": f"deterministic@{CRITIC_VERSION}",
        "report_writer": f"deterministic@{REPORT_WRITER_VERSION}",
        "memory_writer": f"deterministic@{MEMORY_WRITER_VERSION}",
        "calibrator": f"deterministic@{CALIBRATOR_VERSION}",
    }
