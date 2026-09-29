import tempfile
from pathlib import Path

import pytest

from stock_analysis.database import Database, get_database, init_database, close_database
from stock_analysis.config import get_settings


class TestDatabase:
    def test_database_connect(self, temp_db_path):
        db = Database(temp_db_path)
        conn = db.connect()

        assert conn is not None
        cursor = conn.execute("SELECT 1 as test")
        row = cursor.fetchone()
        assert row["test"] == 1

        db.close()

    def test_database_wal_mode(self, temp_db_path):
        db = Database(temp_db_path)
        conn = db.connect()

        cursor = conn.execute("PRAGMA journal_mode")
        row = cursor.fetchone()
        assert row[0].lower() == "wal"

        db.close()

    def test_database_foreign_keys(self, temp_db_path):
        db = Database(temp_db_path)
        conn = db.connect()

        cursor = conn.execute("PRAGMA foreign_keys")
        row = cursor.fetchone()
        assert row[0] == 1

        db.close()

    def test_database_transaction_commit(self, temp_db_path):
        db = Database(temp_db_path)
        conn = db.connect()
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")

        with db.transaction() as txn:
            txn.execute("INSERT INTO test (value) VALUES (?)", ("committed",))

        cursor = conn.execute("SELECT value FROM test")
        row = cursor.fetchone()
        assert row["value"] == "committed"

        db.close()

    def test_database_transaction_rollback(self, temp_db_path):
        db = Database(temp_db_path)
        conn = db.connect()
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")

        with pytest.raises(ValueError):
            with db.transaction() as txn:
                txn.execute("INSERT INTO test (value) VALUES (?)", ("rolled_back",))
                raise ValueError("intentional rollback")

        cursor = conn.execute("SELECT COUNT(*) as cnt FROM test")
        row = cursor.fetchone()
        assert row["cnt"] == 0

        db.close()

    def test_database_execute_fetch(self, temp_db_path):
        db = Database(temp_db_path)
        conn = db.connect()
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")

        db.execute("INSERT INTO test (value) VALUES (?)", ("test1",))
        db.execute("INSERT INTO test (value) VALUES (?)", ("test2",))

        rows = db.fetchall("SELECT value FROM test ORDER BY id")
        assert len(rows) == 2
        assert rows[0]["value"] == "test1"
        assert rows[1]["value"] == "test2"

        row = db.fetchone("SELECT value FROM test WHERE id = ?", (1,))
        assert row["value"] == "test1"

        row = db.fetchone("SELECT value FROM test WHERE id = ?", (999,))
        assert row is None

        db.close()

    def test_get_database_singleton(self, temp_db_path, monkeypatch):
        monkeypatch.setenv("DATABASE_PATH", str(temp_db_path))
        from stock_analysis.config.settings import get_settings
        get_settings.cache_clear()

        db1 = get_database()
        db2 = get_database()

        assert db1 is db2
        assert db1.path == temp_db_path

        close_database()

    def test_init_database(self, temp_db_path, monkeypatch):
        monkeypatch.setenv("DATABASE_PATH", str(temp_db_path))
        from stock_analysis.config.settings import get_settings
        get_settings.cache_clear()

        db = init_database()
        assert db.path == temp_db_path
        assert db.connect() is not None

        close_database()

    def test_close_database(self, temp_db_path, monkeypatch):
        monkeypatch.setenv("DATABASE_PATH", str(temp_db_path))
        from stock_analysis.config.settings import get_settings
        get_settings.cache_clear()

        db = init_database()
        close_database()

        from stock_analysis.database.database import _db_instance
        assert _db_instance is None