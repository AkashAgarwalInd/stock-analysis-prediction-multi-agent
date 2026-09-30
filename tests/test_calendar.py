from datetime import date, timedelta

import pytest

from stock_analysis.market.calendar import (
    TradingCalendar,
    is_trading_day,
    next_trading_day,
    previous_trading_day,
    get_trading_days,
)


class TestTradingCalendar:
    def setup_method(self):
        self.calendar = TradingCalendar()

    def test_is_trading_day_weekday(self):
        monday = date(2024, 1, 15)
        assert self.calendar.is_trading_day(monday) is True

    def test_is_trading_day_saturday(self):
        saturday = date(2024, 1, 20)
        assert self.calendar.is_trading_day(saturday) is False

    def test_is_trading_day_sunday(self):
        sunday = date(2024, 1, 21)
        assert self.calendar.is_trading_day(sunday) is False

    def test_is_trading_day_holiday(self):
        republic_day = date(2024, 1, 26)
        assert self.calendar.is_trading_day(republic_day) is False

    def test_is_trading_day_holiday_2025(self):
        holi = date(2025, 3, 14)
        assert self.calendar.is_trading_day(holi) is False

    def test_is_trading_day_holiday_2026(self):
        republic_day = date(2026, 1, 26)
        assert self.calendar.is_trading_day(republic_day) is False

    def test_next_trading_day_friday_to_monday(self):
        friday = date(2024, 1, 19)
        next_day = self.calendar.next_trading_day(friday)
        assert next_day == date(2024, 1, 22)

    def test_next_trading_day_before_holiday(self):
        thursday = date(2024, 1, 25)
        next_day = self.calendar.next_trading_day(thursday)
        assert next_day == date(2024, 1, 29)

    def test_previous_trading_day_monday_to_friday(self):
        monday = date(2024, 1, 22)
        prev_day = self.calendar.previous_trading_day(monday)
        assert prev_day == date(2024, 1, 19)

    def test_previous_trading_day_after_holiday(self):
        monday = date(2024, 1, 29)
        prev_day = self.calendar.previous_trading_day(monday)
        assert prev_day == date(2024, 1, 25)

    def test_get_trading_days_range(self):
        start = date(2024, 1, 15)
        end = date(2024, 1, 22)
        days = self.calendar.get_trading_days(start, end)
        assert len(days) == 6
        assert days[0] == date(2024, 1, 15)
        assert days[-1] == date(2024, 1, 22)

    def test_get_trading_days_excludes_weekend(self):
        start = date(2024, 1, 20)
        end = date(2024, 1, 21)
        days = self.calendar.get_trading_days(start, end)
        assert len(days) == 0

    def test_get_trading_days_excludes_holiday(self):
        start = date(2024, 1, 25)
        end = date(2024, 1, 26)
        days = self.calendar.get_trading_days(start, end)
        assert len(days) == 1
        assert days[0] == date(2024, 1, 25)

    def test_get_trading_days_count(self):
        start = date(2024, 1, 15)
        end = date(2024, 1, 22)
        count = self.calendar.get_trading_days_count(start, end)
        assert count == 6

    def test_is_weekend(self):
        assert self.calendar.is_weekend(date(2024, 1, 20)) is True
        assert self.calendar.is_weekend(date(2024, 1, 21)) is True
        assert self.calendar.is_weekend(date(2024, 1, 22)) is False

    def test_is_holiday(self):
        assert self.calendar.is_holiday(date(2024, 1, 26)) is True
        assert self.calendar.is_holiday(date(2024, 1, 15)) is False

    def test_get_holidays_in_range(self):
        holidays = self.calendar.get_holidays_in_range(date(2024, 1, 1), date(2024, 12, 31))
        assert len(holidays) >= 10

    def test_get_weekends_in_range(self):
        weekends = self.calendar.get_weekends_in_range(date(2024, 1, 1), date(2024, 1, 31))
        assert len(weekends) == 8

    def test_add_remove_holiday(self):
        custom_date = date(2024, 6, 15)
        assert self.calendar.is_holiday(custom_date) is False
        self.calendar.add_holiday(custom_date)
        assert self.calendar.is_holiday(custom_date) is True
        self.calendar.remove_holiday(custom_date)
        assert self.calendar.is_holiday(custom_date) is False


class TestCalendarHelperFunctions:
    def test_is_trading_day_helper(self):
        assert is_trading_day(date(2024, 1, 15)) is True
        assert is_trading_day(date(2024, 1, 20)) is False

    def test_next_trading_day_helper(self):
        next_day = next_trading_day(date(2024, 1, 19))
        assert next_day == date(2024, 1, 22)

    def test_previous_trading_day_helper(self):
        prev_day = previous_trading_day(date(2024, 1, 22))
        assert prev_day == date(2024, 1, 19)

    def test_get_trading_days_helper(self):
        days = get_trading_days(date(2024, 1, 15), date(2024, 1, 22))
        assert len(days) == 6