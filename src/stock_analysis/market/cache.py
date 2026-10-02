import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

from stock_analysis.config import get_settings
from stock_analysis.database import Database
from stock_analysis.logging import get_logger
from stock_analysis.market.collector import PriceHistory

logger = get_logger(__name__)


class PriceCache:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_settings().database_path
        self._db = Database(self.db_path)
        self._init_cache_table()

    def _init_cache_table(self) -> None:
        self._db.connect()
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS price_cache (
                cache_key TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                data_json TEXT NOT NULL,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                expires_at TEXT
            )
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_cache_symbol ON price_cache(symbol)
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_cache_expires ON price_cache(expires_at)
        """)

    def _serialize_history(self, history: PriceHistory) -> dict:
        return {
            "symbol": history.symbol,
            "data": [d.to_dict() for d in history.data],
            "source": history.source,
            "fetched_at": history.fetched_at.isoformat(),
            "start_date": history.start_date.isoformat() if history.start_date else None,
            "end_date": history.end_date.isoformat() if history.end_date else None,
        }

    def _deserialize_history(self, data: dict) -> PriceHistory:
        from stock_analysis.market.collector import PriceData
        price_data = []
        for d in data.get("data", []):
            pd_obj = PriceData(
                symbol=d["symbol"],
                date=date.fromisoformat(d["date"]),
                open=Decimal(d["open"]) if d.get("open") else None,
                high=Decimal(d["high"]) if d.get("high") else None,
                low=Decimal(d["low"]) if d.get("low") else None,
                close=Decimal(d["close"]) if d.get("close") else None,
                adj_close=Decimal(d["adj_close"]) if d.get("adj_close") else None,
                volume=d.get("volume"),
                source=d.get("source", "yfinance"),
                fetched_at=datetime.fromisoformat(d["fetched_at"]),
            )
            price_data.append(pd_obj)

        return PriceHistory(
            symbol=data["symbol"],
            data=price_data,
            source=data.get("source", "yfinance"),
            fetched_at=datetime.fromisoformat(data["fetched_at"]),
            start_date=date.fromisoformat(data["start_date"]) if data.get("start_date") else None,
            end_date=date.fromisoformat(data["end_date"]) if data.get("end_date") else None,
        )

    def get(self, cache_key: str) -> Optional[PriceHistory]:
        row = self._db.fetchone(
            "SELECT data_json, expires_at FROM price_cache WHERE cache_key = ?",
            (cache_key,)
        )
        if not row:
            return None

        if row["expires_at"]:
            expires = datetime.fromisoformat(row["expires_at"])
            if datetime.utcnow() > expires:
                self.delete(cache_key)
                return None

        try:
            data = json.loads(row["data_json"])
            return self._deserialize_history(data)
        except Exception as e:
            logger.warning("cache_deserialize_failed", cache_key=cache_key, error=str(e))
            self.delete(cache_key)
            return None

    def set(self, cache_key: str, history: PriceHistory, ttl_hours: int = 24) -> None:
        data_json = json.dumps(self._serialize_history(history))

        from datetime import timedelta
        expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)

        self._db.execute("""
            INSERT OR REPLACE INTO price_cache
            (cache_key, symbol, data_json, source, fetched_at, start_date, end_date, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cache_key,
            history.symbol,
            data_json,
            history.source,
            history.fetched_at.isoformat(),
            history.start_date.isoformat() if history.start_date else None,
            history.end_date.isoformat() if history.end_date else None,
            expires_at.isoformat(),
        ))

    def delete(self, cache_key: str) -> bool:
        cursor = self._db.execute("DELETE FROM price_cache WHERE cache_key = ?", (cache_key,))
        return cursor.rowcount > 0

    def clear_expired(self) -> int:
        cursor = self._db.execute(
            "DELETE FROM price_cache WHERE expires_at < ?",
            (datetime.utcnow().isoformat(),)
        )
        return cursor.rowcount

    def clear_symbol(self, symbol: str) -> int:
        cursor = self._db.execute("DELETE FROM price_cache WHERE symbol = ?", (symbol.upper(),))
        return cursor.rowcount

    def get_stats(self) -> dict:
        row = self._db.fetchone("SELECT COUNT(*) as count FROM price_cache")
        return {"entries": row["count"] if row else 0}
