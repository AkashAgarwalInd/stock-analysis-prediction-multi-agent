"""Retries with exponential backoff and rate limits for external calls (Plan.md §43, §48).

Every call to an outside service (Gemini, yfinance, Google News RSS) goes
through a named token-bucket :class:`RateLimiter` and, where a failure may be
temporary, :func:`retry_call` / :func:`retry_call_async`. Limits and retry
budgets come from settings, so tests and batch runs can tune them without code
changes.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

import httpx

from stock_analysis.config.settings import get_settings
from stock_analysis.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Services with a configured rate limit (settings: <name>_requests_per_minute / _burst)
RATE_LIMITED_SERVICES = ("llm", "yfinance", "news")

_TRANSIENT_NAME_MARKERS = (
    "timeout",
    "deadlineexceeded",
    "serviceunavailable",
    "resourceexhausted",
    "toomanyrequests",
    "ratelimit",
    "internalservererror",
    "badgateway",
    "gatewaytimeout",
    "connection",
    "temporarily",
)
# Errors that signal a bug or bad input: retrying cannot help
_PROGRAMMING_ERRORS = (TypeError, ValueError, KeyError, AttributeError, IndexError, NotImplementedError)


class RateLimiter:
    """Thread-safe token bucket: ``burst`` calls at once, then ``rate_per_minute``.

    A rate of 0 disables the limit. ``acquire`` blocks until a token is free and
    returns the time waited.
    """

    def __init__(
        self,
        rate_per_minute: float,
        burst: int = 1,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_minute < 0:
            raise ValueError("rate_per_minute must be >= 0")
        self.rate_per_minute = rate_per_minute
        self.burst = max(1, burst)
        self._clock = clock
        self._sleep = sleep
        self._tokens = float(self.burst)
        self._updated = clock()
        self._lock = threading.Lock()

    @property
    def unlimited(self) -> bool:
        return self.rate_per_minute == 0

    def reserve(self) -> float:
        """Take a token now and return how long to wait before using it."""
        if self.unlimited:
            return 0.0
        per_second = self.rate_per_minute / 60.0
        with self._lock:
            now = self._clock()
            self._tokens = min(self.burst, self._tokens + (now - self._updated) * per_second)
            self._updated = now
            self._tokens -= 1.0
            return 0.0 if self._tokens >= 0 else -self._tokens / per_second

    def acquire(self) -> float:
        """Block until a call is allowed; returns the seconds waited."""
        wait = self.reserve()
        if wait > 0:
            self._sleep(wait)
        return wait

    async def acquire_async(self) -> float:
        """Async :meth:`acquire`."""
        wait = self.reserve()
        if wait > 0:
            await asyncio.sleep(wait)
        return wait


_limiters: dict[tuple[str, float, int], RateLimiter] = {}
_limiters_lock = threading.Lock()


def get_rate_limiter(service: str) -> RateLimiter:
    """The shared limiter for ``service`` (one of ``RATE_LIMITED_SERVICES``).

    A changed setting yields a new limiter, so a test or a re-configured
    process never keeps a stale limit.
    """
    if service not in RATE_LIMITED_SERVICES:
        raise ValueError(f"Unknown rate-limited service: {service}")
    settings = get_settings()
    key = (
        service,
        float(getattr(settings, f"{service}_requests_per_minute")),
        int(getattr(settings, f"{service}_rate_limit_burst")),
    )
    with _limiters_lock:
        limiter = _limiters.get(key)
        if limiter is None:
            limiter = _limiters[key] = RateLimiter(key[1], key[2])
        return limiter


@dataclass(frozen=True)
class RetryPolicy:
    """``attempts`` total tries; the delay after try ``n`` (0-based) is
    ``min(base_delay * 2**n, max_delay)`` with ±``jitter`` (fraction) noise."""

    attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 20.0
    jitter: float = 0.1

    def delay(self, attempt: int) -> float:
        """Backoff in seconds after the failed try ``attempt`` (0-based)."""
        delay = min(self.base_delay * 2.0**attempt, self.max_delay)
        if self.jitter and delay > 0:
            delay *= 1 + random.uniform(-self.jitter, self.jitter)
        return max(0.0, delay)

    @classmethod
    def for_llm(cls) -> RetryPolicy:
        """LLM calls (settings ``llm_retry_*``)."""
        s = get_settings()
        return cls(s.llm_retry_attempts, s.llm_retry_base_delay_seconds, s.llm_retry_max_delay_seconds)

    @classmethod
    def for_external_data(cls) -> RetryPolicy:
        """Market data and news (settings ``external_retry_*``)."""
        s = get_settings()
        return cls(
            s.external_retry_attempts,
            s.external_retry_base_delay_seconds,
            s.external_retry_max_delay_seconds,
        )


def _status_code(err: BaseException) -> int | None:
    """An HTTP status carried by ``err`` (google api_core ``code``, httpx/requests response)."""
    for candidate in (
        getattr(err, "code", None),
        getattr(err, "status_code", None),
        getattr(getattr(err, "response", None), "status_code", None),
    ):
        if isinstance(candidate, int) and not isinstance(candidate, bool) and 100 <= candidate < 600:
            return candidate
    return None


def is_transient_error(err: BaseException) -> bool:
    """True for errors worth retrying: HTTP 408/429/5xx, timeouts and connection failures."""
    status = _status_code(err)
    if status is not None:
        return status in (408, 429) or status >= 500
    # httpx.TransportError: connect/read/write failures and timeouts (RSS feeds)
    if isinstance(err, (TimeoutError, ConnectionError, httpx.TransportError)):
        return True
    name = type(err).__name__.lower()
    if any(marker in name for marker in _TRANSIENT_NAME_MARKERS):
        return True
    message = str(err).lower()
    return any(marker in message for marker in ("429", "rate limit", "too many requests", "503"))


def is_retryable_data_error(err: BaseException) -> bool:
    """Data-source errors: retry anything except programming errors, which a retry cannot fix.

    yfinance and feed parsers raise many unrelated exception types for network
    trouble, so the transient check alone would miss most of them.
    """
    return is_transient_error(err) or not isinstance(err, _PROGRAMMING_ERRORS)


def retry_call(
    fn: Callable[[], T],
    *,
    operation: str,
    policy: RetryPolicy | None = None,
    retryable: Callable[[BaseException], bool] = is_transient_error,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_attempt: Callable[[int], None] | None = None,
) -> T:
    """Call ``fn`` with rate limiting and exponential backoff on retryable errors.

    The last error is re-raised once the attempts are spent or when an error is
    not retryable. ``on_attempt`` gets the 1-based attempt number before each try.
    """
    policy = policy or RetryPolicy()
    for attempt in range(policy.attempts):
        if limiter is not None:
            limiter.acquire()
        if on_attempt is not None:
            on_attempt(attempt + 1)
        try:
            return fn()
        except Exception as err:
            if attempt + 1 >= policy.attempts or not retryable(err):
                raise
            delay = policy.delay(attempt)
            logger.warning(
                "external_call_retrying",
                operation=operation,
                attempt=attempt + 1,
                delay_s=round(delay, 3),
                error_type=type(err).__name__,
                error=str(err)[:200],
            )
            sleep(delay)
    raise AssertionError("unreachable: RetryPolicy.attempts must be >= 1")


async def retry_call_async(
    fn: Callable[[], Awaitable[T]],
    *,
    operation: str,
    policy: RetryPolicy | None = None,
    retryable: Callable[[BaseException], bool] = is_transient_error,
    limiter: RateLimiter | None = None,
) -> T:
    """Async :func:`retry_call`."""
    policy = policy or RetryPolicy()
    for attempt in range(policy.attempts):
        if limiter is not None:
            await limiter.acquire_async()
        try:
            return await fn()
        except Exception as err:
            if attempt + 1 >= policy.attempts or not retryable(err):
                raise
            delay = policy.delay(attempt)
            logger.warning(
                "external_call_retrying",
                operation=operation,
                attempt=attempt + 1,
                delay_s=round(delay, 3),
                error_type=type(err).__name__,
                error=str(err)[:200],
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable: RetryPolicy.attempts must be >= 1")
