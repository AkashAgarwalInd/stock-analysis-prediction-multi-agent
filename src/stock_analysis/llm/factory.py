import json
import time
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from stock_analysis.config import get_settings
from stock_analysis.llm.models import LLMModel, LLMResponse
from stock_analysis.llm.usage import (
    LLMBudgetExceeded,
    LLMCallRecord,
    current_node,
    current_tracker,
    estimate_cost,
)
from stock_analysis.logging import get_logger
from stock_analysis.reliability import RetryPolicy, get_rate_limiter, is_transient_error, retry_call

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

__all__ = ["LLMFactory", "LLMModel", "LLMResponse", "get_llm", "get_llm_factory"]


class LLMFactory:
    """Creates Gemini models per role and runs structured / text generation."""

    def __init__(self, sleep: Callable[[float], None] = time.sleep) -> None:
        self._models: dict[LLMModel, Any] = {}
        self._initialized = False
        self._sleep = sleep  # backoff between retries; injectable for tests

    def initialize(self) -> None:
        """Build one Gemini model per role. A no-op until an API key is configured."""
        if self._initialized:
            return

        settings = get_settings()
        if not settings.gemini_api_key:
            logger.warning("gemini_api_key_not_set")
            return

        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        self._models = {
            LLMModel.PRIMARY: genai.GenerativeModel(settings.gemini_model_primary),
            LLMModel.CRITIC: genai.GenerativeModel(settings.gemini_model_critic),
            LLMModel.FALLBACK: genai.GenerativeModel(settings.gemini_model_fallback),
        }
        self._initialized = True
        logger.info("llm_factory_initialized", roles=[r.value for r in self._models])

    def get_model(self, role: LLMModel) -> Any:
        """Return the Gemini model for ``role``; raises if no API key is configured."""
        self.initialize()
        if not self._models:
            raise RuntimeError("Gemini API key not configured")
        return self._models[role]

    @staticmethod
    def _generation_config(
        temperature: float | None, max_tokens: int | None, json_mode: bool
    ) -> Any:
        import google.generativeai as genai

        settings = get_settings()
        kwargs: dict[str, Any] = {
            "temperature": settings.gemini_temperature if temperature is None else temperature,
            "max_output_tokens": settings.gemini_max_tokens if max_tokens is None else max_tokens,
        }
        if json_mode:
            kwargs["response_mime_type"] = "application/json"
        return genai.GenerationConfig(**kwargs)

    def _call_model(
        self, role: LLMModel, prompt: str, config: Any, purpose: str
    ) -> tuple[Any, Any]:
        """One logical call to the model for ``role``: rate limited, retried with
        exponential backoff on 429/5xx/timeouts (Plan.md §48), and recorded."""
        model = self.get_model(role)
        tracker = current_tracker()
        if tracker is not None:
            tracker.start_call()
        model_name = str(getattr(model, "model_name", role.value))
        attempts = 0

        def count(attempt: int) -> None:
            nonlocal attempts
            attempts = attempt

        options = {"timeout": get_settings().llm_request_timeout_seconds}
        started = time.perf_counter()
        try:
            response = retry_call(
                lambda: model.generate_content(
                    prompt, generation_config=config, request_options=options
                ),
                operation=f"llm:{role.value}",
                policy=RetryPolicy.for_llm(),
                retryable=is_transient_error,
                limiter=get_rate_limiter("llm"),
                sleep=self._sleep,
                on_attempt=count,
            )
        except Exception as err:
            self._record(role, model_name, None, started, attempts, purpose, err)
            raise
        self._record(role, model_name, response, started, attempts, purpose, None)
        return model, response

    @staticmethod
    def _record(
        role: LLMModel,
        model_name: str,
        response: Any,
        started: float,
        attempts: int,
        purpose: str,
        error: Exception | None,
    ) -> None:
        """Log the call and report it to the active usage tracker (Plan.md §51)."""
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = _token_count(usage, "prompt_token_count")
        completion_tokens = _token_count(usage, "candidates_token_count")
        record = LLMCallRecord(
            node=current_node(),
            role=role.value,
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=int((time.perf_counter() - started) * 1000),
            cost_est=estimate_cost(prompt_tokens, completion_tokens),
            status="ok" if error is None else "error",
            attempts=max(attempts, 1),
            purpose=purpose,
            error=None if error is None else f"{type(error).__name__}: {error}"[:300],
        )
        log = logger.info if error is None else logger.warning
        log(
            "llm_call",
            node=record.node,
            role=record.role,
            model=record.model,
            purpose=purpose,
            status=record.status,
            attempts=record.attempts,
            latency_ms=record.latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=round(record.cost_est, 6),
            error=record.error,
        )
        tracker = current_tracker()
        if tracker is not None:
            tracker.record(record)

    def _generate(
        self,
        role: LLMModel,
        prompt: str,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
        purpose: str = "generate",
    ) -> tuple[Any, Any]:
        """Call the model for ``role``; if it still fails after its retries, use the
        FALLBACK model (with its own retries)."""
        config = self._generation_config(temperature, max_tokens, json_mode)
        try:
            return self._call_model(role, prompt, config, purpose)
        except LLMBudgetExceeded:
            raise
        except Exception as exc:  # SDK raises many unrelated error types
            if role == LLMModel.FALLBACK:
                raise
            logger.warning("llm_call_failed_using_fallback", role=role.value, error=str(exc))
            return self._call_model(LLMModel.FALLBACK, prompt, config, purpose)

    def generate_structured(
        self,
        role: LLMModel,
        prompt: str,
        response_schema: type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Generate JSON and validate it against ``response_schema``.

        The schema is appended to the prompt instead of passed as Gemini's
        ``response_schema`` because the SDK rejects free-form ``dict`` fields.
        Output that violates the schema gets one repair attempt (the errors are
        sent back to the model, Plan.md §48); a second violation surfaces as
        ``pydantic.ValidationError`` so the caller can fall back.
        """
        schema = json.dumps(response_schema.model_json_schema())
        full_prompt = f"{prompt}\n\nRespond with JSON only, matching this JSON schema:\n{schema}"
        logger.debug("llm_generate_structured", role=role.value)

        _, response = self._generate(role, full_prompt, temperature, max_tokens, json_mode=True)
        text = _response_text(response)
        try:
            return response_schema.model_validate_json(text)
        except ValidationError as err:
            if not get_settings().llm_structured_output_repair:
                raise
            logger.warning(
                "llm_structured_output_invalid",
                role=role.value,
                schema=response_schema.__name__,
                errors=err.error_count(),
            )
            repair_prompt = _repair_prompt(full_prompt, text, err)
        _, response = self._generate(
            role, repair_prompt, temperature, max_tokens, json_mode=True, purpose="repair"
        )
        return response_schema.model_validate_json(_response_text(response))

    def generate_text(
        self,
        role: LLMModel,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate plain text. Returns an empty response when no API key is configured."""
        try:
            self.get_model(role)
        except RuntimeError:
            return LLMResponse(content="", model="", usage={})

        logger.debug("llm_generate_text", role=role.value)
        model, response = self._generate(role, prompt, temperature, max_tokens, json_mode=False)

        try:
            content = _response_text(response)
        except RuntimeError:  # empty or blocked: an empty text, as before
            content = ""
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            content=content,
            model=getattr(model, "model_name", "unknown"),
            usage={
                "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                "completion_tokens": getattr(usage, "candidates_token_count", 0),
            }
            if usage
            else None,
        )


def _response_text(response: Any) -> str:
    try:
        text = getattr(response, "text", None)
    except ValueError as err:  # Gemini raises when a response was blocked or has no parts
        raise RuntimeError(f"LLM returned no text: {err}") from err
    if not text:
        raise RuntimeError("LLM returned empty response")
    return str(text)


def _token_count(usage: Any, name: str) -> int:
    value = getattr(usage, name, 0) if usage is not None else 0
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


_MAX_REPAIR_ECHO_CHARS = 6000


def _repair_prompt(prompt: str, previous: str, err: ValidationError) -> str:
    """Re-ask once with the validation errors of the previous answer."""
    problems = "\n".join(
        f"- {'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}"
        for e in err.errors(include_url=False)[:20]
    )
    return (
        f"{prompt}\n\nYOUR PREVIOUS RESPONSE WAS REJECTED because it does not match the "
        f"schema or its constraints:\n{problems}\n\nPrevious response:\n"
        f"{previous[:_MAX_REPAIR_ECHO_CHARS]}\n\nReturn a corrected JSON object only."
    )


_factory_instance: LLMFactory | None = None


def get_llm_factory() -> LLMFactory:
    """Get or create the LLM factory singleton."""
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = LLMFactory()
    return _factory_instance


def get_llm(role: LLMModel) -> Any:
    """Get an LLM model by role."""
    return get_llm_factory().get_model(role)


class DisabledLLM:
    """Stands in for the LLM in a quant-only run (every analyst disabled); any call is a bug."""

    def generate_structured(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("LLM calls are disabled for this run")
