import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from stock_analysis.data.cache import DataCache
from stock_analysis.data.fundamentals import Fundamentals
from stock_analysis.database import Database


class TestDataCache:
    @pytest.fixture
    def db(self):
        return Database(":memory:")

    @pytest.fixture
    def cache(self, db):
        return DataCache[Fundamentals](db, "test_cache", ttl_hours=24)

    def test_set_and_get(self, cache):
        fundamentals = Fundamentals(symbol="RELIANCE", pe_ratio=25.0, sector="Energy")
        cache.set("key1", "RELIANCE", fundamentals, source="yfinance")

        result = cache.get("key1")

        assert result is not None
        assert result["symbol"] == "RELIANCE"
        assert result["pe_ratio"] == 25.0
        assert result["sector"] == "Energy"
        assert result["source"] == "yfinance"

    def test_get_nonexistent(self, cache):
        result = cache.get("nonexistent")
        assert result is None

    def test_expired_entry_removed(self, cache):
        fundamentals = Fundamentals(symbol="RELIANCE", pe_ratio=25.0)
        cache.set("key1", "RELIANCE", fundamentals, source="yfinance")

        expired_time = (datetime.utcnow() - timedelta(hours=25)).isoformat()
        cache._db.execute(
            "UPDATE test_cache SET expires_at = ? WHERE cache_key = ?",
            (expired_time, "key1"),
        )

        result = cache.get("key1")
        assert result is None

    def test_delete(self, cache):
        fundamentals = Fundamentals(symbol="RELIANCE", pe_ratio=25.0)
        cache.set("key1", "RELIANCE", fundamentals)

        deleted = cache.delete("key1")
        assert deleted is True

        result = cache.get("key1")
        assert result is None

    def test_delete_nonexistent(self, cache):
        deleted = cache.delete("nonexistent")
        assert deleted is False

    def test_clear_symbol(self, cache):
        f1 = Fundamentals(symbol="RELIANCE", pe_ratio=25.0)
        f2 = Fundamentals(symbol="RELIANCE", pe_ratio=30.0)
        f3 = Fundamentals(symbol="TCS", pe_ratio=28.0)

        cache.set("key1", "RELIANCE", f1)
        cache.set("key2", "RELIANCE", f2)
        cache.set("key3", "TCS", f3)

        cleared = cache.clear_symbol("RELIANCE")
        assert cleared == 2

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is not None

    def test_clear_expired(self, cache):
        f1 = Fundamentals(symbol="RELIANCE", pe_ratio=25.0)
        f2 = Fundamentals(symbol="TCS", pe_ratio=28.0)

        cache.set("key1", "RELIANCE", f1)
        cache.set("key2", "TCS", f2)

        expired_time = (datetime.utcnow() - timedelta(hours=25)).isoformat()
        cache._db.execute("UPDATE test_cache SET expires_at = ? WHERE cache_key = ?", (expired_time, "key1"))

        cleared = cache.clear_expired()
        assert cleared == 1

        assert cache.get("key1") is None
        assert cache.get("key2") is not None

    def test_get_stats(self, cache):
        f1 = Fundamentals(symbol="RELIANCE", pe_ratio=25.0)
        f2 = Fundamentals(symbol="TCS", pe_ratio=28.0)

        cache.set("key1", "RELIANCE", f1)
        cache.set("key2", "TCS", f2)

        expired_time = (datetime.utcnow() - timedelta(hours=25)).isoformat()
        cache._db.execute("UPDATE test_cache SET expires_at = ? WHERE cache_key = ?", (expired_time, "key1"))

        stats = cache.get_stats()
        assert stats["total_entries"] == 2
        assert stats["expired_entries"] == 1
        assert stats["table_name"] == "test_cache"