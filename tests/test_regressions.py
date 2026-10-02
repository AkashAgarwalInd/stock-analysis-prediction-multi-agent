"""Regression tests for bugs found in the whole-repo gap audit."""

from datetime import date, datetime
from decimal import Decimal

from stock_analysis.data.cache import DataCache
from stock_analysis.data.collector import _fundamentals_from_dict, _market_context_from_dict
from stock_analysis.data.fundamentals import Fundamentals
from stock_analysis.data.market_context import MarketContext
from stock_analysis.database import Database
from stock_analysis.guardrails import get_critical_gaps
from stock_analysis.market.cache import PriceCache
from stock_analysis.market.collector import PriceData, PriceHistory
from stock_analysis.market.resolver import SymbolResolver
from stock_analysis.schemas.analyst_reports import AnalystReport, DataGapSeverity


def test_database_writes_persist_across_connections(tmp_path):
    path = tmp_path / "persist.db"
    writer = Database(path)
    writer.execute("CREATE TABLE t (k TEXT)")
    writer.execute("INSERT INTO t VALUES (?)", ("a",))
    writer.executemany("INSERT INTO t VALUES (?)", [("b",), ("c",)])
    writer.close()

    reader = Database(path)
    assert [r["k"] for r in reader.fetchall("SELECT k FROM t ORDER BY k")] == ["a", "b", "c"]
    reader.close()


def test_transaction_rolls_back_on_error(tmp_path):
    db = Database(tmp_path / "txn.db")
    db.execute("CREATE TABLE t (k TEXT)")
    try:
        with db.transaction() as conn:
            conn.execute("INSERT INTO t VALUES ('x')")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert db.fetchall("SELECT * FROM t") == []


def test_price_cache_persists_decimals_across_instances(tmp_path):
    db_path = tmp_path / "prices.db"
    history = PriceHistory(
        symbol="TCS.NS",
        data=[
            PriceData(
                symbol="TCS.NS",
                date=date(2024, 1, 15),
                open=Decimal("3500.10"),
                close=Decimal("3530.25"),
                volume=10,
            )
        ],
        source="yfinance",
        fetched_at=datetime(2024, 1, 15, 10, 0, 0),
        start_date=date(2024, 1, 15),
        end_date=date(2024, 1, 15),
    )
    PriceCache(db_path=db_path).set("TCS.NS:2y", history)

    restored = PriceCache(db_path=db_path).get("TCS.NS:2y")

    assert restored is not None
    assert restored.data[0].close == Decimal("3530.25")
    assert isinstance(restored.data[0].open, Decimal)


def test_data_cache_round_trip_restores_datetimes(tmp_path):
    db = Database(tmp_path / "data.db")
    cache = DataCache(db, "fundamentals_cache", deserializer=_fundamentals_from_dict)
    fundamentals = Fundamentals(
        symbol="TCS", fetched_at=datetime(2024, 1, 2, 3, 4), earnings_date=datetime(2024, 2, 1)
    )
    cache.set("k", "TCS", fundamentals)

    restored = cache.get("k")

    assert restored.fetched_at == datetime(2024, 1, 2, 3, 4)
    assert restored.earnings_date == datetime(2024, 2, 1)


def test_market_context_keeps_sector_change_pct_through_cache(tmp_path):
    db = Database(tmp_path / "ctx.db")
    cache = DataCache(db, "market_context_cache", deserializer=_market_context_from_dict)
    cache.set("k", "TCS", MarketContext(symbol="TCS", sector_index_change_pct=1.5))

    assert cache.get("k").sector_index_change_pct == 1.5


def test_resolver_symbol_with_suffix_is_not_mangled():
    resolver = SymbolResolver(enable_yfinance_validation=False)
    assert resolver._to_yfinance_symbol("RELIANCE.NS") == "RELIANCE.NS"
    assert resolver._to_yfinance_symbol("tcs.bo") == "TCS.BO"
    assert resolver._to_yfinance_symbol("m&m") == "M&M.NS"
    assert resolver._to_yfinance_symbol("bajaj-auto") == "BAJAJAUTO.NS"


def _report(data_gaps):
    return AnalystReport(
        analyst="technical",
        stance="neutral",
        confidence=0.5,
        key_points=["Range-bound price action"],
        evidence=["RSI near 50"],
        risks=["Breakout failure could extend losses"],
        data_gaps=data_gaps,
    )


def test_plain_string_gap_mentioning_required_does_not_block():
    report = _report(["Missing required intraday volume data", "no data for options"])
    assert {g.severity for g in report.data_gaps} == {DataGapSeverity.MEDIUM}


def test_explicit_critical_gap_still_blocks():
    report = _report([{"description": "No price history", "severity": "critical"}])
    assert report.data_gaps[0].severity == DataGapSeverity.CRITICAL


def test_get_critical_gaps_extracts_only_critical():
    reports = [
        _report([{"description": "No price history", "severity": "critical"}]).model_dump(
            mode="json"
        ),
        _report(["Missing required intraday volume data"]).model_dump(mode="json"),
        None,
        {"analyst": "not-a-valid-report"},
    ]

    critical = get_critical_gaps(reports)

    assert critical == [{"analyst": "technical", "description": "No price history"}]
