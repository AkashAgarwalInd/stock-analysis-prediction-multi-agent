#!/usr/bin/env python3
"""
Demo script showing all implemented functionality.
Run: python demo.py
"""

import numpy as np
from stock_analysis import (
    # Phase 1: Config, Database, Logging, Schemas
    Settings,
    Database,
    configure_logging,
    get_logger,
    # Phase 2: Market Data Layer
    SymbolResolver,
    MarketPriceCollector,
    TradingCalendar,
    PriceCache,
    TechnicalIndicators,
    compute_all_indicators,
    # Phase 3: Fundamental, News, Market Context
    Fundamentals,
    FundamentalsCollector,
    NewsItem,
    NewsCollection,
    NewsCollector,
    MarketContext,
    MarketContextCollector,
    DataCache,
    CompleteData,
    DataCollector,
    # Phase 4: Quantitative Forecasting
    QuantBaseline,
    QuantForecaster,
    EWMAVolatility,
    VolatilityModel,
    HistoricalReturns,
    compute_momentum_drift,
)


def demo_phase1_config_logging():
    print("\n" + "="*60)
    print("PHASE 1: Config, Database, Logging, Schemas")
    print("="*60)

    # Config
    settings = Settings()
    print(f"✓ Settings loaded: env={settings.app_env}, log_level={settings.log_level}")

    # Logging
    configure_logging()
    logger = get_logger(__name__)
    logger.info("Demo logging works")

    # Database (in-memory for demo)
    db = Database(":memory:")
    db.connect()
    print(f"✓ Database connected: {db.path}")


def demo_phase2_market_data():
    print("\n" + "="*60)
    print("PHASE 2: Market Data Layer")
    print("="*60)

    # Symbol Resolver
    resolver = SymbolResolver()
    symbol = resolver.resolve("RELIANCE")
    print(f"✓ Resolved: {symbol.symbol} ({symbol.name}) - {symbol.exchange}")

    # Fuzzy match
    symbol2 = resolver.resolve("reliance industries")
    print(f"✓ Fuzzy match: {symbol2.symbol}")

    # Calendar
    from datetime import date
    cal = TradingCalendar()
    next_day = cal.next_trading_day(date.today())
    print(f"✓ Next trading day: {next_day}")

    # Technical Indicators
    import pandas as pd
    prices = np.array([100 + i * 0.5 + np.sin(i) for i in range(250)])
    df = pd.DataFrame({
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.random.randint(100000, 1000000, len(prices))
    })
    indicators_list = compute_all_indicators(df)
    if indicators_list:
        ind = indicators_list[-1]
        print(f"✓ Indicators computed: RSI={ind.rsi_14:.1f}, MACD={ind.macd:.3f}, BB_upper={ind.bb_upper:.1f}")
    else:
        print("⚠ Insufficient data for indicators (need 200+ rows)")


async def demo_phase3_data_layer():
    print("\n" + "="*60)
    print("PHASE 3: Fundamental, News, Market Context Data Layer")
    print("="*60)

    db = Database(":memory:")
    collector = DataCollector(db)

    # Fundamentals (uses yfinance - will work if network available)
    print("Fetching fundamentals for RELIANCE (cached if available)...")
    try:
        fund = collector.get_fundamentals("RELIANCE", use_cache=False)
        print(f"✓ Fundamentals: P/E={fund.pe_ratio}, ROE={fund.roe}, Sector={fund.sector}, MarketCap={fund.market_cap}")
    except Exception as e:
        print(f"⚠ Fundamentals (network): {e}")

    # News
    print("Fetching news for RELIANCE...")
    try:
        news = await collector.get_news("RELIANCE", use_cache=False)
        print(f"✓ News: {len(news.items)} articles, source={news.source}")
        for item in news.items[:2]:
            print(f"  - {item.title[:60]}... ({item.source})")
    except Exception as e:
        print(f"⚠ News (network): {e}")

    # Market Context
    print("Fetching market context for RELIANCE...")
    try:
        ctx = collector.get_market_context("RELIANCE", sector="Energy", use_cache=False)
        print(f"✓ Market Context: Nifty={ctx.nifty_50:.0f}, VIX={ctx.india_vix:.1f}, USD/INR={ctx.usd_inr:.2f}")
        print(f"  Relative strength vs Nifty: {ctx.relative_strength_vs_nifty:.2f}%")
    except Exception as e:
        print(f"⚠ Market Context (network): {e}")

    await collector.close()


def demo_phase4_quant_forecast():
    print("\n" + "="*60)
    print("PHASE 4: Quantitative Forecasting Engine")
    print("="*60)

    # Generate synthetic price data
    np.random.seed(42)
    base = 2500.0
    returns = np.random.normal(0.0005, 0.015, 252)  # 1 year daily
    prices = [base]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    prices = np.array(prices)

    print(f"Synthetic prices: {len(prices)} days, last={prices[-1]:.2f}")

    # Basic forecast
    forecaster = QuantForecaster(seed=42, n_paths=10000, horizon=5)
    baseline = forecaster.forecast(prices)

    print(f"\n✓ QuantBaseline:")
    print(f"  Last close:      {baseline.last_close:.2f}")
    print(f"  Horizon:         {baseline.horizon_trading_days} days")
    print(f"  P(up):           {baseline.prob_up:.1%}")
    print(f"  P(flat):         {baseline.prob_flat:.1%}")
    print(f"  P(down):         {baseline.prob_down:.1%}")
    print(f"  Expected return: {baseline.expected_return_pct:.2f}%")
    print(f"  P10 price:       {baseline.p10_price:.2f}")
    print(f"  P50 price:       {baseline.p50_price:.2f}")
    print(f"  P90 price:       {baseline.p90_price:.2f}")
    print(f"  Weekly vol:      {baseline.weekly_vol_pct:.2f}%")
    print(f"  Method:          {baseline.method}")

    # Deterministic seed test
    f1 = QuantForecaster(seed=123, n_paths=5000)
    f2 = QuantForecaster(seed=123, n_paths=5000)
    r1 = f1.forecast(prices)
    r2 = f2.forecast(prices)
    assert r1.p50_price == r2.p50_price, "Seeds should be deterministic"
    print(f"\n✓ Deterministic seed verified (same seed = same P50)")

    # With momentum drift
    baseline_drift = forecaster.forecast(prices, calibration_params={"momentum_drift": 0.002})
    print(f"\n✓ With momentum_drift=0.2%: expected_return={baseline_drift.expected_return_pct:.2f}%")

    # With regime
    baseline_regime = forecaster.forecast(prices, regime="trending")
    print(f"✓ With regime='trending': expected_return={baseline_regime.expected_return_pct:.2f}%")

    # Custom volatility model
    class ConstantVol(VolatilityModel):
        def estimate(self, returns): return 0.02
        def simulate_paths(self, returns, n_paths, horizon, drift, rng):
            return np.full((n_paths, horizon), drift + 0.02)

    forecaster_custom = QuantForecaster(n_paths=100, volatility_model=ConstantVol())
    baseline_custom = forecaster_custom.forecast(prices)
    print(f"✓ Custom volatility model works: weekly_vol={baseline_custom.weekly_vol_pct:.2f}%")


async def demo_integration():
    print("\n" + "="*60)
    print("INTEGRATION: Full Pipeline")
    print("="*60)

    db = Database(":memory:")
    collector = DataCollector(db)

    # Get all data for a symbol
    symbol = "RELIANCE"
    print(f"Fetching complete data for {symbol}...")

    try:
        complete = await collector.get_complete_async(symbol, sector="Energy")
        print(f"✓ CompleteData:")
        print(f"  Fundamentals: {'✓' if complete.fundamentals else '✗'}")
        print(f"  News:         {len(complete.news.items) if complete.news else 0} articles")
        print(f"  Market Ctx:   {'✓' if complete.market_context else '✗'}")
    except Exception as e:
        print(f"⚠ Integration (network): {e}")

    # Run quant forecast on fetched prices (or synthetic)
    print("\nRunning quant forecast on market data...")
    # Use synthetic for demo since we don't have real price history in collector
    np.random.seed(42)
    demo_prices = np.array([2500 * (1.001 ** i) * (1 + np.random.normal(0, 0.01)) for i in range(100)])
    baseline = QuantForecaster(seed=42).forecast(demo_prices)
    print(f"✓ Forecast complete: P50={baseline.p50_price:.2f}, Expected={baseline.expected_return_pct:.2f}%")

    await collector.close()


def main():
    print("="*60)
    print("STOCK ANALYSIS PREDICTION - MULTI-AGENT SYSTEM")
    print("Phases 1-4 Complete Demo")
    print("="*60)

    demo_phase1_config_logging()
    demo_phase2_market_data()
    import asyncio
    asyncio.run(demo_phase3_data_layer())
    demo_phase4_quant_forecast()
    asyncio.run(demo_integration())

    print("\n" + "="*60)
    print("ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nRun tests: pytest tests/ -v")
    print("Run CLI:   python -m stock_analysis.cli.main --help")


if __name__ == "__main__":
    main()