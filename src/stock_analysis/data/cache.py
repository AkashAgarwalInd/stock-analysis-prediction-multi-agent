import json
from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Generic, Optional, TypeVar

from stock_analysis.database import Database
from stock_analysis.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class DataCache(Generic[T]):
    def __init__(
        self,
        db: Database,
        table_name: str,
        ttl_hours: int = 24,
        deserializer: Optional[Callable[[dict], T]] = None
    ):
        self._db = db
        self._table_name = table_name
        self._ttl_hours = ttl_hours
        self._deserializer = deserializer
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                cache_key TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                data_json TEXT NOT NULL,
                source TEXT,
                fetched_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._db.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_symbol ON {self._table_name}(symbol)
        """)
        self._db.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table_name}_expires ON {self._table_name}(expires_at)
        """)

    def _serialize(self, obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, datetime):
            return obj.isoformat()
        if is_dataclass(obj):
            result = {}
            for field in fields(obj):
                value = getattr(obj, field.name)
                result[field.name] = self._serialize(value)
            return result
        if isinstance(obj, list):
            return [self._serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        if hasattr(obj, '__dict__'):
            return self._serialize(obj.__dict__)
        return obj

    def _deserialize(self, data: Any) -> Any:
        if self._deserializer and isinstance(data, dict):
            return self._deserializer(data)
        return data

    def get(self, cache_key: str) -> Optional[T]:
        row = self._db.execute(
            f"SELECT data_json, expires_at FROM {self._table_name} WHERE cache_key = ?",
            (cache_key,)
        ).fetchone()

        if not row:
            return None

        expires_at = datetime.fromisoformat(row["expires_at"])
        if datetime.utcnow() > expires_at:
            self.delete(cache_key)
            return None

        try:
            data = json.loads(row["data_json"])
            return self._deserialize(data)
        except Exception as e:
            logger.warning("cache_deserialize_failed", table=self._table_name, cache_key=cache_key, error=str(e))
            self.delete(cache_key)
            return None

    def set(self, cache_key: str, symbol: str, data: T, source: str = "", fetched_at: Optional[datetime] = None) -> None:
        data_json = json.dumps(self._serialize(data))
        now = fetched_at or datetime.utcnow()
        expires_at = now + timedelta(hours=self._ttl_hours)

        self._db.execute(f"""
            INSERT OR REPLACE INTO {self._table_name}
            (cache_key, symbol, data_json, source, fetched_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            cache_key,
            symbol,
            data_json,
            source,
            now.isoformat(),
            expires_at.isoformat(),
        ))

    def delete(self, cache_key: str) -> bool:
        cursor = self._db.execute(f"DELETE FROM {self._table_name} WHERE cache_key = ?", (cache_key,))
        return cursor.rowcount > 0

    def clear_symbol(self, symbol: str) -> int:
        cursor = self._db.execute(f"DELETE FROM {self._table_name} WHERE symbol = ?", (symbol,))
        return cursor.rowcount

    def clear_expired(self) -> int:
        cursor = self._db.execute(
            f"DELETE FROM {self._table_name} WHERE expires_at < ?",
            (datetime.utcnow().isoformat(),)
        )
        return cursor.rowcount

    def get_stats(self) -> dict[str, Any]:
        row = self._db.execute(f"SELECT COUNT(*) as count FROM {self._table_name}").fetchone()
        expired = self._db.execute(
            f"SELECT COUNT(*) as count FROM {self._table_name} WHERE expires_at < ?",
            (datetime.utcnow().isoformat(),)
        ).fetchone()
        return {
            "total_entries": row["count"] if row else 0,
            "expired_entries": expired["count"] if expired else 0,
            "table_name": self._table_name,
        }
