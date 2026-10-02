import os
import tempfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from stock_analysis.market.cache import PriceCache
from stock_analysis.market.collector import PriceData, PriceHistory


class TestPriceCache:
    def setup_method(self):
        # Use a temp database for each test
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.cache = PriceCache(db_path=Path(self.temp_db.name))

    def teardown_method(self):
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_init_creates_table(self):
        # Just verify no exception during init
        cache = PriceCache(db_path=Path(self.temp_db.name))
        stats = cache.get_stats()
        assert "entries" in stats

    def test_set_and_get(self):
        history = PriceHistory(
            symbol="RELIANCE.NS",
            data=[
                PriceData(
                    symbol="RELIANCE.NS",
                    date=date(2024, 1, 15),
                    close=Decimal("2530.00"),
                    volume=1000000,
                )
            ],
            source="yfinance",
            fetched_at=datetime(2024, 1, 15, 10, 0, 0),
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 15),
        )

        cache_key = "RELIANCE.NS:2y:1d:None:None"
        self.cache.set(cache_key, history)

        retrieved = self.cache.get(cache_key)

        assert retrieved is not None
        assert retrieved.symbol == "RELIANCE.NS"
        assert len(retrieved.data) == 1
        assert retrieved.data[0].close == Decimal("2530.00")

    def test_get_nonexistent_key(self):
        result = self.cache.get("nonexistent_key")
        assert result is None

    def test_delete(self):
        history = PriceHistory(
            symbol="RELIANCE.NS",
            data=[PriceData(symbol="RELIANCE.NS", date=date(2024, 1, 15), close=Decimal("2530.00"))],
        )
        cache_key = "test_key"
        self.cache.set(cache_key, history)

        deleted = self.cache.delete(cache_key)
        assert deleted is True

        retrieved = self.cache.get(cache_key)
        assert retrieved is None

    def test_delete_nonexistent(self):
        deleted = self.cache.delete("nonexistent")
        assert deleted is False

    def test_clear_symbol(self):
        history_reliance = PriceHistory(symbol="RELIANCE.NS", data=[])
        history_tcs = PriceHistory(symbol="TCS.NS", data=[])
        self.cache.set("test_clear_symbol_key1:RELIANCE.NS", history_reliance)
        self.cache.set("test_clear_symbol_key2:RELIANCE.NS", history_reliance)
        self.cache.set("test_clear_symbol_key3:TCS.NS", history_tcs)

        count = self.cache.clear_symbol("RELIANCE.NS")
        assert count == 2

        assert self.cache.get("test_clear_symbol_key1:RELIANCE.NS") is None
        assert self.cache.get("test_clear_symbol_key2:RELIANCE.NS") is None
        assert self.cache.get("test_clear_symbol_key3:TCS.NS") is not None

    def test_clear_expired(self):
        history = PriceHistory(symbol="RELIANCE.NS", data=[])

        self.cache.set("expired_key", history, ttl_hours=-1)
        self.cache.set("valid_key", history, ttl_hours=24)

        count = self.cache.clear_expired()
        assert count >= 1

    def test_get_stats(self):
        history = PriceHistory(symbol="RELIANCE.NS", data=[])
        self.cache.set("stat_key1", history)
        self.cache.set("stat_key2", history)

        stats = self.cache.get_stats()
        assert stats["entries"] >= 2

    def test_cache_expiration(self):
        history = PriceHistory(
            symbol="RELIANCE.NS",
            data=[PriceData(symbol="RELIANCE.NS", date=date(2024, 1, 15), close=Decimal("2530.00"))],
        )
        cache_key = "expire_test"
        self.cache.set(cache_key, history, ttl_hours=-1)

        result = self.cache.get(cache_key)
        assert result is None
