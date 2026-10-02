import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Union

from stock_analysis.config import get_settings
from stock_analysis.logging import get_logger

logger = get_logger(__name__)


class Database:
    def __init__(self, path: Union[str, Path], echo: bool = False):
        self.path = Path(path) if isinstance(path, str) else path
        self.echo = echo
        self._conn: sqlite3.Connection | None = None
        self._is_memory = str(self.path) == ":memory:"
        self._in_txn = False

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self._is_memory:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                self.path,
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.row_factory = sqlite3.Row
            if self.echo:
                self._conn.set_trace_callback(logger.debug)
            logger.info("database_connected", path=str(self.path))
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("database_closed", path=str(self.path))

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self.connect()
        self._in_txn = True
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._in_txn = False

    def _commit_if_write(self, conn: sqlite3.Connection) -> None:
        """Commit any implicit transaction opened by an INSERT/UPDATE/DELETE.

        Skipped inside ``transaction()`` so multi-statement units stay atomic.
        """
        if conn.in_transaction and not self._in_txn:
            conn.commit()

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        conn = self.connect()
        cursor = conn.execute(query, params)
        self._commit_if_write(conn)
        if self.echo:
            logger.debug("sql_executed", query=query, params=params)
        return cursor

    def executemany(self, query: str, params_list: list[tuple[Any, ...]]) -> sqlite3.Cursor:
        conn = self.connect()
        cursor = conn.executemany(query, params_list)
        self._commit_if_write(conn)
        if self.echo:
            logger.debug("sql_executemany", query=query, count=len(params_list))
        return cursor

    def fetchone(self, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.execute(query, params).fetchone()
        return row

    def fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return self.execute(query, params).fetchall()


_db_instance: Database | None = None


def get_database() -> Database:
    global _db_instance
    if _db_instance is None:
        settings = get_settings()
        _db_instance = Database(settings.database_path, settings.database_echo)
    return _db_instance


def init_database() -> Database:
    global _db_instance
    settings = get_settings()
    _db_instance = Database(settings.database_path, settings.database_echo)
    _db_instance.connect()
    return _db_instance


def close_database() -> None:
    global _db_instance
    if _db_instance is not None:
        _db_instance.close()
        _db_instance = None
