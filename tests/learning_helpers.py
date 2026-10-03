"""Helpers for postmortem, lesson and calibration tests: dated forecasts and fake LLMs."""

from datetime import UTC, date, datetime, time, timedelta

from stock_analysis.market.calendar import TradingCalendar
from stock_analysis.review import PriceBar, score_forecast
from stock_analysis.schemas.graph_state import GraphState
from stock_analysis.schemas.learning import PostmortemDiagnosis
from stock_analysis.schemas.outcome import ForecastOutcome
from stock_analysis.schemas.snapshot import ForecastSnapshot
from stock_analysis.snapshots import build_forecast_snapshot
from tests.outcome_helpers import nifty_series, path_series

# Weekends only (the far-future holiday just extends coverage), so any weekday
# can be an as-of date and consecutive weeks never overlap.
WEEKDAYS = TradingCalendar(holidays={date(2099, 1, 1)})
FIRST_MONDAY = date(2026, 1, 5)


def week(n: int) -> date:
    """The Monday ``n`` weeks after FIRST_MONDAY (forecast windows Mon -> next Mon)."""
    return FIRST_MONDAY + timedelta(weeks=n)


def made_at(as_of: date) -> datetime:
    """17:30 IST on the as-of date: after the NSE close."""
    return datetime.combine(as_of, time(12, 0), tzinfo=UTC)


def evaluated_at(snapshot: ForecastSnapshot) -> datetime:
    """The day after the target session, after end-of-day data is published."""
    return datetime.combine(snapshot.target_date + timedelta(days=1), time(12, 0), tzinfo=UTC)


def snapshot_as_of(state: GraphState, as_of: date, **updates) -> ForecastSnapshot:
    """A snapshot of ``state`` re-dated to ``as_of`` (target five weekdays later)."""
    price = {**state.price_snapshot, "last_date": as_of.isoformat()}
    dated = state.model_copy(update={"price_snapshot": price, **updates})
    return build_forecast_snapshot(dated, made_at=made_at(as_of), calendar=WEEKDAYS)


def wiggle_path(final_pct: float, amplitude: float) -> list[float]:
    """Cumulative % path ending at ``final_pct``, zig-zagging by ``amplitude`` around a line."""
    offsets = [amplitude, -amplitude, amplitude, -amplitude, 0.0]
    return [final_pct * (i + 1) / 5 + off for i, off in enumerate(offsets)]


def score(
    snapshot: ForecastSnapshot,
    final_pct: float,
    *,
    amplitude: float = 0.6,
    nifty_pct: float = 0.0,
    now: datetime | None = None,
    dividend_on_day: int | None = None,
) -> ForecastOutcome:
    """Score ``snapshot`` against a synthetic actual path (weekday calendar).

    ``dividend_on_day`` (1-based session index) adds a dividend corporate action.
    """
    actions = None
    if dividend_on_day is not None:
        day = snapshot.daily_predictions[dividend_on_day - 1].target_date
        actions = {day: PriceBar(date=day, close=1.0, dividend=2.0)}
    return score_forecast(
        snapshot,
        path_series(snapshot, wiggle_path(final_pct, amplitude), actions=actions),
        nifty_series(snapshot, nifty_pct),
        now=now or evaluated_at(snapshot),
        calendar=WEEKDAYS,
    )


def save_scored(store, outcome_store, snapshot, final_pct, **kwargs) -> ForecastOutcome:
    store.save(snapshot)
    outcome = score(snapshot, final_pct, **kwargs)
    outcome_store.save(outcome)
    return outcome


def fact_id(facts, source: str, contains: str = "") -> str:
    """ID of the first fact from ``source`` whose text contains ``contains``."""
    return next(
        f.id
        for f in (*facts.forecast_time, *facts.hindsight)
        if f.source == source and contains in f.text
    )


class FakePostmortemLLM:
    """Returns a fixed diagnosis (validated like LLMFactory does) or raises."""

    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.prompts: list[str] = []

    def generate_structured(self, role, prompt, response_schema, **_):
        assert response_schema is PostmortemDiagnosis
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        payload = self.response(prompt) if callable(self.response) else self.response
        return response_schema.model_validate(payload)
