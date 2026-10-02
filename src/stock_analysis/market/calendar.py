from datetime import date, timedelta
from functools import lru_cache
from typing import Optional

from stock_analysis.logging import get_logger

logger = get_logger(__name__)


NSE_HOLIDAYS_2024 = {
    date(2024, 1, 26),   # Republic Day
    date(2024, 3, 8),    # Mahashivratri
    date(2024, 3, 25),   # Holi
    date(2024, 3, 29),   # Good Friday
    date(2024, 4, 11),   # Id-Ul-Fitr
    date(2024, 4, 17),   # Ram Navami
    date(2024, 5, 1),    # Maharashtra Day
    date(2024, 6, 17),   # Bakri Id
    date(2024, 7, 17),   # Muharram
    date(2024, 8, 15),   # Independence Day
    date(2024, 10, 2),   # Gandhi Jayanti
    date(2024, 11, 1),   # Diwali Laxmi Pujan
    date(2024, 11, 15),  # Guru Nanak Jayanti
    date(2024, 12, 25),  # Christmas
}

NSE_HOLIDAYS_2025 = {
    date(2025, 1, 26),   # Republic Day
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-Ul-Fitr
    date(2025, 4, 10),   # Ram Navami
    date(2025, 4, 14),   # Dr. Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 6, 7),    # Bakri Id
    date(2025, 7, 6),    # Muharram
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 27),   # Ganesh Chaturthi
    date(2025, 10, 2),   # Gandhi Jayanti
    date(2025, 10, 21),  # Diwali Laxmi Pujan
    date(2025, 11, 5),   # Guru Nanak Jayanti
    date(2025, 12, 25),  # Christmas
}

NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 15),   # Mahashivratri
    date(2026, 3, 4),    # Holi
    date(2026, 3, 20),   # Id-Ul-Fitr
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 6),    # Ram Navami
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 27),   # Bakri Id
    date(2026, 7, 26),   # Muharram
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 27),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 10, 10),  # Diwali Laxmi Pujan
    date(2026, 11, 24),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
}

ALL_HOLIDAYS = NSE_HOLIDAYS_2024 | NSE_HOLIDAYS_2025 | NSE_HOLIDAYS_2026


class TradingCalendar:
    def __init__(self, holidays: Optional[set[date]] = None):
        self._holidays = holidays or ALL_HOLIDAYS
        self._weekend_days = {5, 6}

    def is_trading_day(self, dt: date) -> bool:
        if dt.weekday() in self._weekend_days:
            return False
        if dt in self._holidays:
            return False
        return True

    def next_trading_day(self, dt: date, max_lookahead: int = 10) -> Optional[date]:
        for i in range(1, max_lookahead + 1):
            next_day = dt + timedelta(days=i)
            if self.is_trading_day(next_day):
                return next_day
        return None

    def previous_trading_day(self, dt: date, max_lookback: int = 10) -> Optional[date]:
        for i in range(1, max_lookback + 1):
            prev_day = dt - timedelta(days=i)
            if self.is_trading_day(prev_day):
                return prev_day
        return None

    def get_trading_days(self, start: date, end: date) -> list[date]:
        if start > end:
            return []
        trading_days = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                trading_days.append(current)
            current += timedelta(days=1)
        return trading_days

    def get_trading_days_count(self, start: date, end: date) -> int:
        return len(self.get_trading_days(start, end))

    def is_weekend(self, dt: date) -> bool:
        return dt.weekday() in self._weekend_days

    def is_holiday(self, dt: date) -> bool:
        return dt in self._holidays

    def get_holidays_in_range(self, start: date, end: date) -> list[date]:
        return [d for d in self._holidays if start <= d <= end]

    def get_weekends_in_range(self, start: date, end: date) -> list[date]:
        weekends = []
        current = start
        while current <= end:
            if current.weekday() in self._weekend_days:
                weekends.append(current)
            current += timedelta(days=1)
        return weekends

    def add_holiday(self, dt: date) -> None:
        self._holidays.add(dt)

    def remove_holiday(self, dt: date) -> None:
        self._holidays.discard(dt)


@lru_cache(maxsize=1)
def get_trading_calendar() -> TradingCalendar:
    return TradingCalendar()


def is_trading_day(dt: date) -> bool:
    return get_trading_calendar().is_trading_day(dt)


def next_trading_day(dt: date) -> Optional[date]:
    return get_trading_calendar().next_trading_day(dt)


def previous_trading_day(dt: date) -> Optional[date]:
    return get_trading_calendar().previous_trading_day(dt)


def get_trading_days(start: date, end: date) -> list[date]:
    return get_trading_calendar().get_trading_days(start, end)
