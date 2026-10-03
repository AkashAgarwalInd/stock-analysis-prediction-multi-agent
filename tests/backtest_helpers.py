"""Synthetic historical data and fakes for backtest tests."""

from datetime import UTC, date, datetime, timedelta

import numpy as np

from stock_analysis.backtest import HistoricalBar, HistoricalMarketData
from stock_analysis.backtest.data import INDIA_VIX_SYMBOL
from stock_analysis.market.calendar import get_trading_calendar
from stock_analysis.review.prices import NIFTY_50_SYMBOL
from stock_analysis.schemas.learning import PostmortemDiagnosis
from tests.forecast_helpers import FakeLLM

SYMBOL = "RELIANCE.NS"
LAST_TARGET = date(2026, 9, 25)  # a Friday
# Thu 2026-10-01 noon IST: every session up to LAST_TARGET has completed
NOW = datetime(2026, 10, 1, 6, 30, tzinfo=UTC)
HISTORY_START = date(2023, 6, 1)


def _bars(days, closes, volume=1_000_000.0):
    return [
        HistoricalBar(date=d, open=c, high=c * 1.005, low=c * 0.995, close=float(c), volume=volume)
        for d, c in zip(days, closes, strict=True)
    ]


def synthetic_history(
    *,
    end: date = LAST_TARGET + timedelta(days=21),
    seed: int = 7,
    weekly_swing_pct: float = 0.0,
    swing_from: date = date(2026, 1, 1),
    poison_after: date | None = None,
) -> HistoricalMarketData:
    """Stock, Nifty 50 and India VIX on NSE trading days.

    ``weekly_swing_pct`` adds a drift of that size per 5-session block (random sign)
    from ``swing_from``: weekly moves then exceed what daily volatility implies, so the
    quant baseline underestimates weekly volatility. ``poison_after`` multiplies every
    later price by 3 (to prove no forecast looks past its as-of date).
    """
    days = get_trading_calendar().get_trading_days(HISTORY_START, end)
    rng = np.random.default_rng(seed)
    n = len(days)
    market = rng.normal(0.0002, 0.008, n)
    stock = 0.9 * market + rng.normal(0.0002, 0.008, n)
    signs = rng.choice([-1.0, 1.0], size=n // 5 + 1)
    for i, d in enumerate(days):
        if weekly_swing_pct and d >= swing_from:
            stock[i] += signs[i // 5] * weekly_swing_pct / 100 / 5
    stock_closes = 2500 * np.exp(np.cumsum(stock))
    nifty_closes = 24000 * np.exp(np.cumsum(market))
    vix = 14 + np.cumsum(rng.normal(0, 0.2, n)).clip(-6, 10)
    if poison_after is not None:
        factor = np.array([3.0 if d > poison_after else 1.0 for d in days])
        stock_closes, nifty_closes, vix = stock_closes * factor, nifty_closes * factor, vix * factor
    return HistoricalMarketData(
        {
            SYMBOL: _bars(days, stock_closes),
            NIFTY_50_SYMBOL: _bars(days, nifty_closes),
            INDIA_VIX_SYMBOL: _bars(days, vix, volume=0.0),
        }
    )


class RecordingLoader:
    """A ``history_loader`` that serves fixed data and records what was requested."""

    def __init__(self, data: HistoricalMarketData):
        self.data = data
        self.requests: list[tuple[list[str], date, date]] = []

    def __call__(self, symbols, start, end) -> HistoricalMarketData:
        self.requests.append((list(symbols), start, end))
        return self.data


class BacktestLLM(FakeLLM):
    """FakeLLM plus a postmortem that blames volatility and proposes one lesson."""

    def __init__(self):
        super().__init__("valid")
        self.analyst_prompts: list[str] = []
        self.postmortem_prompts: list[str] = []

    def generate_structured(self, role, prompt, response_schema, **kwargs):
        if response_schema is PostmortemDiagnosis:
            self.postmortem_prompts.append(prompt)
            return PostmortemDiagnosis.model_validate(
                {
                    "primary_cause": "volatility_underestimated",
                    "explanation": "Weekly swings exceeded the forecast's volatility.",
                    "only_in_hindsight": [{"statement": "Realised move", "fact_ids": ["H1"]}],
                    "lessons": [
                        {
                            "text": "Weekly moves for this ticker can exceed the EWMA volatility "
                            "estimate when daily volatility looks calm",
                            "scope": "ticker",
                            "category": "volatility_underestimated",
                            "evidence_refs": ["F3"],
                        }
                    ],
                    "confidence": "medium",
                }
            )
        if "QUANT BASELINE:" not in prompt:
            self.analyst_prompts.append(prompt)
        return super().generate_structured(role, prompt, response_schema, **kwargs)


class CountingLLM:
    """Records every call; used to prove a quant-only backtest makes none."""

    def __init__(self):
        self.calls = 0

    def generate_structured(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("no LLM call expected")
