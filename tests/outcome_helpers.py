"""Synthetic actual-price helpers for outcome scoring tests."""

from datetime import UTC, date, datetime

from stock_analysis.review import NIFTY_50_SYMBOL, PriceBar, PriceSeries
from stock_analysis.schemas.snapshot import ForecastSnapshot

# Thu 2026-10-08 17:30 IST: the 2026-10-07 target session is complete
NOW_MATURED = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)


def window_dates(snapshot: ForecastSnapshot) -> list[date]:
    """as_of_date followed by the forecast's trading sessions."""
    return [snapshot.as_of_date] + [p.target_date for p in snapshot.daily_predictions]


def path_series(
    snapshot: ForecastSnapshot,
    path_pct: list[float],
    *,
    symbol: str = "RELIANCE.NS",
    base: float | None = None,
    actions: dict[date, PriceBar] | None = None,
) -> PriceSeries:
    """Closes moving ``path_pct`` (cumulative % from as_of) over the forecast window.

    ``base`` defaults to the snapshot's last close; a different base simulates
    price history re-adjusted after the forecast was made.
    """
    start = snapshot.last_close if base is None else base
    closes = [start * (1 + r / 100) for r in [0.0, *path_pct]]
    bars = []
    for d, close in zip(window_dates(snapshot), closes, strict=True):
        extra = (actions or {}).get(d)
        bars.append(
            PriceBar(
                date=d,
                close=close,
                dividend=extra.dividend if extra else 0.0,
                split=extra.split if extra else 0.0,
            )
        )
    return PriceSeries(symbol=symbol, bars=bars)


def nifty_series(snapshot: ForecastSnapshot, final_return_pct: float) -> PriceSeries:
    """A Nifty 50 series that ends ``final_return_pct`` above its as_of level."""
    step = final_return_pct / snapshot.horizon_trading_days
    path = [step * (i + 1) for i in range(snapshot.horizon_trading_days)]
    return path_series(snapshot, path, symbol=NIFTY_50_SYMBOL, base=24000.0)


class FakePriceSource:
    """Serves fixed series per symbol and records every requested window."""

    def __init__(self, series: dict[str, PriceSeries]):
        self.series = series
        self.calls: list[tuple[str, date, date]] = []

    def fetch(self, symbol: str, start: date, end: date) -> PriceSeries:
        self.calls.append((symbol, start, end))
        return self.series.get(symbol, PriceSeries(symbol=symbol, error="no data"))
