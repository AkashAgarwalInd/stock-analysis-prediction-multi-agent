"""Phase 18: rate limits and retries with exponential backoff (Plan.md §43, §48)."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from stock_analysis.reliability import (
    RateLimiter,
    RetryPolicy,
    get_rate_limiter,
    is_retryable_data_error,
    is_transient_error,
    retry_call,
    retry_call_async,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _status_error(code: int) -> Exception:
    err = Exception(f"HTTP {code}")
    err.code = code
    return err


class TestRateLimiter:
    def test_burst_then_steady_rate(self):
        clock = FakeClock()
        limiter = RateLimiter(60, burst=2, clock=clock, sleep=clock.sleep)  # one per second
        waits = [limiter.acquire() for _ in range(4)]
        assert waits[:2] == [0.0, 0.0]
        assert waits[2:] == pytest.approx([1.0, 1.0])
        assert clock.now == pytest.approx(2.0)

    def test_tokens_refill_while_idle(self):
        clock = FakeClock()
        limiter = RateLimiter(60, burst=2, clock=clock, sleep=clock.sleep)
        limiter.acquire(), limiter.acquire()
        clock.now += 10  # long idle: the bucket refills, but never beyond the burst
        assert [limiter.acquire() for _ in range(3)] == pytest.approx([0.0, 0.0, 1.0])

    def test_zero_rate_is_unlimited(self):
        clock = FakeClock()
        limiter = RateLimiter(0, clock=clock, sleep=clock.sleep)
        assert all(limiter.acquire() == 0.0 for _ in range(100))
        assert clock.sleeps == []

    def test_negative_rate_rejected(self):
        with pytest.raises(ValueError):
            RateLimiter(-1)

    def test_async_acquire_waits(self):
        clock = FakeClock()
        limiter = RateLimiter(6000, burst=1, clock=clock)  # 10 ms per call
        asyncio.run(limiter.acquire_async())
        assert asyncio.run(limiter.acquire_async()) == pytest.approx(0.01)

    def test_shared_limiter_follows_settings(self, monkeypatch):
        from stock_analysis.config.settings import get_settings

        monkeypatch.setenv("LLM_REQUESTS_PER_MINUTE", "12")
        monkeypatch.setenv("LLM_RATE_LIMIT_BURST", "3")
        get_settings.cache_clear()
        limiter = get_rate_limiter("llm")
        assert (limiter.rate_per_minute, limiter.burst) == (12, 3)
        assert get_rate_limiter("llm") is limiter  # shared while the settings hold

        monkeypatch.setenv("LLM_REQUESTS_PER_MINUTE", "24")
        get_settings.cache_clear()
        assert get_rate_limiter("llm").rate_per_minute == 24

    def test_unknown_service(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_rate_limiter("smtp")


class TestTransientErrors:
    @pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504])
    def test_retryable_status(self, code):
        assert is_transient_error(_status_error(code))

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
    def test_client_errors_are_final(self, code):
        assert not is_transient_error(_status_error(code))

    def test_http_response_status(self):
        request = httpx.Request("GET", "https://example.com")
        err = httpx.HTTPStatusError(
            "x", request=request, response=httpx.Response(503, request=request)
        )
        assert is_transient_error(err)

    def test_timeouts_and_connection_errors(self):
        assert is_transient_error(TimeoutError())
        assert is_transient_error(ConnectionResetError())
        assert is_transient_error(httpx.ReadTimeout("slow"))

    def test_sdk_error_names(self):
        ResourceExhausted = type("ResourceExhausted", (Exception,), {})
        DeadlineExceeded = type("DeadlineExceeded", (Exception,), {})
        assert is_transient_error(ResourceExhausted("quota"))
        assert is_transient_error(DeadlineExceeded("late"))

    def test_bad_requests_are_final(self):
        assert not is_transient_error(ValueError("bad schema"))
        assert not is_transient_error(Exception("API key not valid"))

    def test_data_errors_retry_except_programming_errors(self):
        assert is_retryable_data_error(Exception("curl: (35) SSL connect error"))
        assert not is_retryable_data_error(KeyError("Close"))
        assert not is_retryable_data_error(TypeError("bad call"))

    def test_boolean_code_is_not_a_status(self):
        err = Exception("x")
        err.code = True
        assert not is_transient_error(err)


class TestRetryCall:
    def test_retries_with_exponential_backoff(self):
        clock, calls = FakeClock(), []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise _status_error(503)
            return "ok"

        policy = RetryPolicy(attempts=4, base_delay=1.0, max_delay=30.0, jitter=0.0)
        assert retry_call(flaky, operation="t", policy=policy, sleep=clock.sleep) == "ok"
        assert clock.sleeps == [1.0, 2.0]

    def test_backoff_is_capped(self):
        policy = RetryPolicy(attempts=10, base_delay=1.0, max_delay=5.0, jitter=0.0)
        assert [policy.delay(n) for n in range(5)] == [1.0, 2.0, 4.0, 5.0, 5.0]

    def test_jitter_stays_within_bounds(self):
        policy = RetryPolicy(base_delay=10.0, max_delay=10.0, jitter=0.1)
        assert all(9.0 <= policy.delay(0) <= 11.0 for _ in range(50))

    def test_non_retryable_error_raises_at_once(self):
        clock, calls = FakeClock(), []

        def broken():
            calls.append(1)
            raise _status_error(400)

        with pytest.raises(Exception, match="400"):
            retry_call(broken, operation="t", policy=RetryPolicy(attempts=5), sleep=clock.sleep)
        assert len(calls) == 1 and clock.sleeps == []

    def test_last_error_raised_after_attempts(self):
        clock, attempts = FakeClock(), []

        def down():
            raise TimeoutError("still down")

        with pytest.raises(TimeoutError):
            retry_call(
                down,
                operation="t",
                policy=RetryPolicy(attempts=3, base_delay=0.5, jitter=0.0),
                sleep=clock.sleep,
                on_attempt=attempts.append,
            )
        assert attempts == [1, 2, 3]
        assert clock.sleeps == [0.5, 1.0]

    def test_rate_limiter_is_acquired_per_attempt(self):
        acquired = []
        limiter = SimpleNamespace(acquire=lambda: acquired.append(1))
        outcomes = iter([TimeoutError(), "done"])

        def call():
            value = next(outcomes)
            if isinstance(value, Exception):
                raise value
            return value

        retry_call(call, operation="t", limiter=limiter, sleep=lambda _s: None)
        assert len(acquired) == 2

    def test_async_retry(self):
        calls = []

        async def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise httpx.ConnectError("reset")
            return 7

        policy = RetryPolicy(attempts=2, base_delay=0.0, jitter=0.0)
        assert asyncio.run(retry_call_async(flaky, operation="t", policy=policy)) == 7

    def test_policies_come_from_settings(self, monkeypatch):
        from stock_analysis.config.settings import get_settings

        monkeypatch.setenv("LLM_RETRY_ATTEMPTS", "5")
        monkeypatch.setenv("EXTERNAL_RETRY_ATTEMPTS", "2")
        get_settings.cache_clear()
        assert RetryPolicy.for_llm().attempts == 5
        assert RetryPolicy.for_external_data().attempts == 2


class TestDataSourcesRetry:
    def test_actual_prices_retry_a_failed_download(self, monkeypatch):
        import pandas as pd

        from stock_analysis.review.prices import YFinancePriceSource

        frame = pd.DataFrame(
            {"Close": [100.0, 101.0], "Dividends": [0.0, 0.0], "Stock Splits": [0.0, 0.0],
             "Volume": [10, 12]},
            index=pd.DatetimeIndex(["2026-09-28", "2026-09-29"]),
        )  # fmt: skip
        calls = []

        class Ticker:
            def __init__(self, symbol):
                pass

            def history(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    raise ConnectionError("reset by peer")
                return frame

        monkeypatch.setattr("yfinance.Ticker", Ticker)
        from datetime import date

        series = YFinancePriceSource().fetch("RELIANCE.NS", date(2026, 9, 28), date(2026, 9, 29))
        assert len(calls) == 2
        assert calls[0]["timeout"] == 30.0  # EXTERNAL_REQUEST_TIMEOUT_SECONDS
        assert [b.close for b in series.bars] == [100.0, 101.0]

    def test_zero_volume_bar_on_a_holiday_is_dropped(self, monkeypatch):
        """yfinance repeats the previous close on some holidays (2026-05-01 for RELIANCE)."""
        from datetime import date

        import pandas as pd

        from stock_analysis.review.prices import YFinancePriceSource, is_placeholder_bar

        frame = pd.DataFrame(
            {"Close": [1424.2, 1424.2, 1456.4], "Dividends": [0.0] * 3,
             "Stock Splits": [0.0] * 3, "Volume": [30957881, 0, 24035700]},
            index=pd.DatetimeIndex(["2026-04-30", "2026-05-01", "2026-05-04"]),
        )  # fmt: skip
        monkeypatch.setattr(
            "yfinance.Ticker", lambda _s: SimpleNamespace(history=lambda **_: frame)
        )
        series = YFinancePriceSource().fetch("RELIANCE.NS", date(2026, 4, 30), date(2026, 5, 4))
        assert [b.date for b in series.bars] == [date(2026, 4, 30), date(2026, 5, 4)]
        # a zero-volume bar on a trading day (e.g. an index) is kept
        assert not is_placeholder_bar(date(2026, 4, 30), 0)
