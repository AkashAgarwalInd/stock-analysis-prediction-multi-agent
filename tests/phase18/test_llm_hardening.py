"""Phase 18: LLM retries, structured-output repair, usage tracking and the call budget
(Plan.md §48, §50-51)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from stock_analysis.llm import (
    LLMBudgetExceeded,
    LLMFactory,
    LLMModel,
    LLMUsageSummary,
    LLMUsageTracker,
    llm_node,
    track_llm_usage,
)
from stock_analysis.llm.usage import LLMCallRecord, estimate_cost


class Answer(BaseModel):
    verdict: str
    score: float


def _response(text, prompt_tokens=120, completion_tokens=30):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens, candidates_token_count=completion_tokens
        ),
    )


def _model(*outcomes, name="fake-model"):
    """A model whose successive calls return (or raise) ``outcomes``."""
    model = MagicMock()
    model.model_name = name
    model.generate_content.side_effect = list(outcomes)
    return model


def _status_error(code):
    err = Exception(f"{code} error")
    err.code = code
    return err


def _factory(sleeps=None, **models):
    factory = LLMFactory(sleep=(sleeps.append if sleeps is not None else lambda _s: None))
    factory._models = {LLMModel(role): model for role, model in models.items()}
    factory._initialized = True
    return factory


GOOD = json.dumps({"verdict": "ok", "score": 0.5})


class TestRetries:
    def test_transient_error_is_retried_on_the_same_model(self, monkeypatch):
        from stock_analysis.config.settings import get_settings

        monkeypatch.setenv("LLM_RETRY_BASE_DELAY_SECONDS", "2")
        get_settings.cache_clear()
        sleeps = []
        primary = _model(_status_error(429), _status_error(503), _response(GOOD))
        fallback = _model(_response(GOOD))
        result = _factory(sleeps, primary=primary, fallback=fallback).generate_structured(
            LLMModel.PRIMARY, "x", Answer
        )
        assert result.verdict == "ok"
        assert primary.generate_content.call_count == 3
        fallback.generate_content.assert_not_called()
        assert len(sleeps) == 2 and 1.8 <= sleeps[0] <= 2.2 and 3.6 <= sleeps[1] <= 4.4

    def test_exhausted_retries_move_to_the_fallback_model(self):
        primary = _model(*[_status_error(503)] * 3)
        fallback = _model(_response(json.dumps({"verdict": "backup", "score": 1})))
        result = _factory(primary=primary, fallback=fallback).generate_structured(
            LLMModel.PRIMARY, "x", Answer
        )
        assert result.verdict == "backup"
        assert primary.generate_content.call_count == 3  # LLM_RETRY_ATTEMPTS default

    def test_non_transient_error_goes_to_fallback_without_retry(self):
        primary = _model(_status_error(400))
        fallback = _model(_response(GOOD))
        _factory(primary=primary, fallback=fallback).generate_structured(
            LLMModel.PRIMARY, "x", Answer
        )
        assert primary.generate_content.call_count == 1

    def test_calls_are_rate_limited(self, monkeypatch):
        acquired = []
        monkeypatch.setattr(
            "stock_analysis.llm.factory.get_rate_limiter",
            lambda name: SimpleNamespace(acquire=lambda: acquired.append(name)),
        )
        _factory(primary=_model(TimeoutError(), _response(GOOD))).generate_structured(
            LLMModel.PRIMARY, "x", Answer
        )
        assert acquired == ["llm", "llm"]  # once per attempt


class TestStructuredOutputRepair:
    def test_invalid_output_is_repaired_once(self):
        primary = _model(_response('{"verdict": "ok"}'), _response(GOOD))
        result = _factory(primary=primary).generate_structured(LLMModel.PRIMARY, "Rate it.", Answer)
        assert result == Answer(verdict="ok", score=0.5)
        repair_prompt = primary.generate_content.call_args_list[1].args[0]
        assert "Rate it." in repair_prompt
        assert "REJECTED" in repair_prompt and "score: Field required" in repair_prompt
        assert '{"verdict": "ok"}' in repair_prompt  # the rejected answer is shown

    def test_second_violation_raises_validation_error(self):
        primary = _model(_response("not json"), _response('{"verdict": 1}'))
        fallback = _model(_response(GOOD))
        with pytest.raises(ValidationError):
            _factory(primary=primary, fallback=fallback).generate_structured(
                LLMModel.PRIMARY, "x", Answer
            )
        assert primary.generate_content.call_count == 2
        fallback.generate_content.assert_not_called()

    def test_repair_can_be_disabled(self, monkeypatch):
        from stock_analysis.config.settings import get_settings

        monkeypatch.setenv("LLM_STRUCTURED_OUTPUT_REPAIR", "false")
        get_settings.cache_clear()
        primary = _model(_response('{"verdict": "ok"}'), _response(GOOD))
        with pytest.raises(ValidationError):
            _factory(primary=primary).generate_structured(LLMModel.PRIMARY, "x", Answer)
        assert primary.generate_content.call_count == 1


class TestUsageTracking:
    def test_calls_are_recorded_with_node_tokens_and_cost(self):
        tracker = LLMUsageTracker(run_id="run-1", ticker="RELIANCE")
        factory = _factory(primary=_model(_response(GOOD, 1000, 200)))
        with track_llm_usage(tracker), llm_node("technical_analyst"):
            factory.generate_structured(LLMModel.PRIMARY, "x", Answer)

        (record,) = tracker.records
        assert (record.run_id, record.ticker, record.node) == ("run-1", "RELIANCE", "technical_analyst")
        assert (record.model, record.role, record.status) == ("fake-model", "primary", "ok")
        assert (record.prompt_tokens, record.completion_tokens) == (1000, 200)
        assert record.cost_est == pytest.approx(estimate_cost(1000, 200))
        assert record.cost_est == pytest.approx((1000 * 0.10 + 200 * 0.40) / 1e6)
        assert record.attempts == 1 and record.purpose == "generate"

    def test_retries_repair_and_failures_are_visible(self):
        tracker = LLMUsageTracker()
        primary = _model(TimeoutError(), _response('{"verdict": "ok"}'), _response(GOOD))
        with track_llm_usage(tracker):
            _factory(primary=primary).generate_structured(LLMModel.PRIMARY, "x", Answer)
        first, repair = tracker.records
        assert first.attempts == 2 and first.purpose == "generate"
        assert repair.purpose == "repair"

        tracker = LLMUsageTracker()
        primary, fallback = _model(*[TimeoutError()] * 3), _model(_response(GOOD))
        with track_llm_usage(tracker):
            _factory(primary=primary, fallback=fallback).generate_structured(
                LLMModel.PRIMARY, "x", Answer
            )
        failed, ok = tracker.records
        assert (failed.status, failed.attempts, failed.role) == ("error", 3, "primary")
        assert "TimeoutError" in failed.error
        assert (ok.status, ok.role) == ("ok", "fallback")
        assert tracker.summary().failed_calls == 1

    def test_no_tracker_means_no_accounting(self):
        _factory(primary=_model(_response(GOOD))).generate_structured(
            LLMModel.PRIMARY, "x", Answer
        )  # just logs

    def test_failing_sink_never_fails_the_call(self):
        def sink(_record):
            raise RuntimeError("disk full")

        tracker = LLMUsageTracker(sink=sink)
        with track_llm_usage(tracker):
            _factory(primary=_model(_response(GOOD))).generate_structured(
                LLMModel.PRIMARY, "x", Answer
            )
        assert len(tracker.records) == 1

    def test_summary(self):
        records = [
            LLMCallRecord(node="a", role="primary", model="m", prompt_tokens=10,
                          completion_tokens=5, latency_ms=3, cost_est=0.001, status="ok"),
            LLMCallRecord(node="a", role="primary", model="m", prompt_tokens=7,
                          completion_tokens=0, latency_ms=3, cost_est=0.0, status="error"),
            LLMCallRecord(node=None, role="fallback", model="m", prompt_tokens=1,
                          completion_tokens=1, latency_ms=3, cost_est=0.0005, status="ok"),
        ]  # fmt: skip
        summary = LLMUsageSummary.of(records)
        assert (summary.calls, summary.failed_calls) == (3, 1)
        assert (summary.prompt_tokens, summary.completion_tokens, summary.total_tokens) == (18, 6, 24)
        assert summary.cost_est == pytest.approx(0.0015)
        assert summary.by_node == {"a": 2, "unknown": 1}
        assert "3 calls (1 failed)" in summary.describe()
        assert LLMUsageSummary().describe() == "LLM usage: no calls"


class TestBudget:
    def test_budget_stops_further_calls(self):
        tracker = LLMUsageTracker(max_calls=2)
        primary = _model(*[_response(GOOD)] * 3)
        factory = _factory(primary=primary)
        with track_llm_usage(tracker):
            factory.generate_structured(LLMModel.PRIMARY, "x", Answer)
            factory.generate_structured(LLMModel.PRIMARY, "x", Answer)
            with pytest.raises(LLMBudgetExceeded):
                factory.generate_structured(LLMModel.PRIMARY, "x", Answer)
        assert primary.generate_content.call_count == 2

    def test_exceeded_budget_does_not_fall_back(self):
        tracker = LLMUsageTracker(max_calls=1)
        primary = _model(*[_response(GOOD)] * 2)
        fallback = _model(_response(GOOD))
        factory = _factory(primary=primary, fallback=fallback)
        with track_llm_usage(tracker):
            factory.generate_structured(LLMModel.PRIMARY, "x", Answer)
            with pytest.raises(LLMBudgetExceeded):
                factory.generate_structured(LLMModel.PRIMARY, "x", Answer)
        fallback.generate_content.assert_not_called()

    def test_zero_means_unlimited(self):
        tracker = LLMUsageTracker(max_calls=0)
        for _ in range(5):
            tracker.start_call()


class TestRequestHandling:
    def test_every_request_has_a_timeout(self, monkeypatch):
        from stock_analysis.config.settings import get_settings

        monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "45")
        get_settings.cache_clear()
        primary = _model(_response(GOOD))
        _factory(primary=primary).generate_structured(LLMModel.PRIMARY, "x", Answer)
        assert primary.generate_content.call_args.kwargs["request_options"] == {"timeout": 45.0}

    def test_blocked_response_is_an_llm_failure(self):
        class Blocked:
            usage_metadata = None

            @property
            def text(self):
                raise ValueError("response was blocked (finish_reason SAFETY)")

        primary = _model(Blocked())
        with pytest.raises(RuntimeError, match="no text"):
            _factory(primary=primary).generate_structured(LLMModel.PRIMARY, "x", Answer)

        response = _factory(primary=_model(Blocked())).generate_text(LLMModel.PRIMARY, "x")
        assert response.content == ""
