"""Find matured forecasts, score them against actual prices and store the results."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from typing import Optional

from stock_analysis.database.forecast_store import ForecastSnapshotError, ForecastSnapshotStore
from stock_analysis.database.outcome_store import OutcomeStore, OutcomeStoreError
from stock_analysis.logging import get_logger
from stock_analysis.market.calendar import TradingCalendar, get_trading_calendar
from stock_analysis.review.outcome_scorer import last_completed_trading_date, score_forecast
from stock_analysis.review.prices import (
    NIFTY_50_SYMBOL,
    PriceSeries,
    PriceSource,
    YFinancePriceSource,
)
from stock_analysis.schemas.outcome import ForecastOutcome, OutcomeStatus

logger = get_logger(__name__)


class OutcomeReviewer:
    """Score every matured, unevaluated original forecast (Plan.md §5.2 steps 1 and 6)."""

    def __init__(
        self,
        snapshot_store: ForecastSnapshotStore,
        outcome_store: OutcomeStore,
        price_source: Optional[PriceSource] = None,
        calendar: Optional[TradingCalendar] = None,
    ):
        self.snapshot_store = snapshot_store
        self.outcome_store = outcome_store
        self.price_source = price_source or YFinancePriceSource()
        self.calendar = calendar or get_trading_calendar()

    def find_matured(self, now: datetime, *, ticker: Optional[str] = None) -> list[str]:
        """Original forecasts whose target session has completed and that have no outcome yet."""
        completed = last_completed_trading_date(now, self.calendar)
        return self.outcome_store.matured_unevaluated(completed, ticker=ticker)

    def review_matured(
        self, now: datetime, *, ticker: Optional[str] = None
    ) -> list[ForecastOutcome]:
        """Evaluate matured forecasts; scored/invalid results are stored, unresolved are retried later.

        One forecast's failure is logged and skipped so it never blocks the others.
        """
        return self.score(self.find_matured(now, ticker=ticker), now)[0]

    def score(
        self, forecast_ids: list[str], now: datetime
    ) -> tuple[list[ForecastOutcome], list[str]]:
        """Score ``forecast_ids`` (from ``find_matured``) against actual prices at ``now``.

        Returns the outcomes (scored and invalid ones are stored) and one error per
        forecast that could not be reviewed; those are retried next run.
        """
        nifty_cache: dict[tuple[date, date], PriceSeries] = {}
        results = []
        errors: list[str] = []
        for forecast_id in forecast_ids:
            try:
                snapshot = self.snapshot_store.get(forecast_id)
                if snapshot is None:
                    continue
                window = (snapshot.as_of_date, snapshot.target_date)
                # Fetch only the forecast's own window: later prices are never requested
                stock = self.price_source.fetch(snapshot.resolved_symbol, *window)
                if window not in nifty_cache:
                    nifty_cache[window] = self.price_source.fetch(NIFTY_50_SYMBOL, *window)
                outcome = score_forecast(
                    snapshot, stock, nifty_cache[window], now=now, calendar=self.calendar
                )
                if outcome.status != OutcomeStatus.UNRESOLVED:
                    self.outcome_store.save(outcome)
            except (
                ForecastSnapshotError,
                OutcomeStoreError,
                sqlite3.Error,
                ValueError,  # includes pydantic ValidationError
            ) as err:
                logger.error("forecast_review_failed", forecast_id=forecast_id, error=str(err))
                errors.append(f"{forecast_id}: {err}")
                continue
            logger.info(
                "forecast_reviewed",
                forecast_id=forecast_id,
                status=outcome.status.value,
                reason=outcome.invalid_reason,
            )
            results.append(outcome)
        return results, errors
