# Stock Analysis Prediction - Multi-Agent System

A modular, test-driven **multi-agent stock analysis and adaptive forecasting system for Indian NSE stocks**.

The system combines market, fundamental, news, and broader market-context data with quantitative forecasting and LLM-based analysis to produce **probabilistic next-5-trading-day forecasts**.

The long-term architecture is designed around a closed evaluation loop:

**Forecast → Immutable Snapshot → Actual Outcome → Evaluation → Postmortem → Controlled Calibration → Future Forecast**

The quantitative forecasting engine is currently implemented through Phase 4. The multi-agent analysis, LLM prediction, evaluation, and adaptive learning layers are planned for subsequent phases.

---

## Architecture Overview

```text
User Query
    ↓
Symbol Resolver
    ↓
Previous Forecast Review
    ↓
Market / Fundamental / News / Context Collectors
    ↓
Parallel Analysts
    │
    ├── Fundamental Analyst
    ├── Technical Analyst
    ├── News Analyst
    └── Market Context Analyst
    ↓
Quantitative Forecast Baseline
    ↓
LLM Predictor
    ↓
Critic
    ↓
Final Forecast
    ↓
Immutable Forecast Snapshot
    ↓
Later: Actual-vs-Forecast Evaluation
    ↓
Postmortem
    ↓
Controlled Calibration / Learning
    ↓
Future Forecast
```

### Current Implementation Status

```text
Phase 1: Foundation                         ✅ Complete
Phase 2: Market Data Layer                  ✅ Complete
Phase 3: Fundamental / News / Context Data  ✅ Complete
Phase 4: Quantitative Forecasting            ✅ Complete

Phase 5: LLM Analyst Nodes                  ⏳ Planned
Phase 6: Signal Synthesis                    ⏳ Planned
Phase 7: LangGraph Workflow                  ⏳ Planned
Phase 8: Evaluation / Backtest / Postmortem  ⏳ Planned
Phase 9: Calibration / Adaptive Learning     ⏳ Planned
```

---

## Key Design Principles

1. **Numbers come from Python/tools, never from the LLM**
   Financial calculations, indicators, probabilities, simulations, and other numerical computations are performed by deterministic Python components.

2. **LLMs interpret structured facts**
   LLMs reason over structured market, fundamental, news, technical, and quantitative outputs rather than raw uncontrolled data.

3. **Pydantic structured outputs**
   LLM outputs and important application boundaries use validated Pydantic schemas.

4. **No large DataFrames in workflow state**
   LangGraph state should remain compact and contain references or structured summaries rather than large datasets.

5. **Immutable forecasts**
   Once a forecast is created and stored, the original forecast must never be modified.

6. **Full auditability**
   The system should retain enough information to understand what data, signals, models, and reasoning contributed to a forecast.

7. **Measured evaluation**
   Forecasts are evaluated against the actual market outcome after the forecast horizon.

8. **Baseline comparison**
   LLM-adjusted forecasts are evaluated against the quantitative baseline rather than assuming that LLM reasoning automatically improves the forecast.

9. **Evidence-based analysis**
   Analysts should distinguish observed facts, quantitative signals, and interpretations. Unsupported claims should not become forecast inputs.

10. **Bounded learning**
    A single incorrect forecast or unusual market week should not cause uncontrolled changes to the forecasting system.

11. **Point-in-time integrity**
    Historical evaluation must use only information that would have been available at the time the original forecast was generated.

12. **Reproducibility**
    Quantitative simulations should support deterministic seeds and reproducible runs.

13. **Graceful degradation**
    Temporary failures or unavailable external data sources should not unnecessarily crash the entire analysis pipeline.

14. **Educational disclaimer**
    Forecast reports must clearly state that the system is for educational/research purposes and does not constitute financial advice.

---

## Current Implementation

### Phase 1: Foundation

The foundation provides:

* Pydantic Settings configuration
* Environment configuration
* SQLite database
* Alembic migrations
* Structured logging with structlog
* Pydantic schemas
* Google Gemini LLM factory
* CLI framework

### Phase 2: Market Data Layer

The market data layer provides:

* `SymbolResolver`

  * NSE symbol resolution
  * yfinance validation
* `MarketPriceCollector`

  * OHLCV market data
* `TradingCalendar`

  * NSE trading-day and holiday handling
* `PriceCache`

  * SQLite-backed cache with TTL
* `TechnicalIndicators`

  * 13 technical indicators including:

    * SMA
    * EMA
    * RSI
    * MACD
    * Bollinger Bands
    * ATR
    * Momentum
    * and others

### Phase 3: Fundamental, News & Market Context Data

The data layer provides three primary information sources.

#### Fundamentals

`FundamentalsCollector` retrieves available company-level information such as:

* P/E
* P/B
* ROE
* Debt/Equity
* Margins
* Growth
* Dividend information
* Market capitalization
* Sector
* Beta
* Earnings information
* Other optional company metrics

#### News

`NewsCollector` provides:

* Google News RSS
* 14-day lookback
* URL-based deduplication
* HTML stripping
* Content truncation

#### Market Context

`MarketContextCollector` provides broader market information including:

* Nifty 50
* Bank Nifty
* India VIX
* USD/INR
* Sector indices
* Relative strength versus Nifty
* Beta/context information

#### Data Caching

The data layer uses SQLite-backed caching with separate TTLs for different data types.

Current defaults include:

* Fundamentals: 24 hours
* News: 6 hours
* Market context: 4 hours

The system is designed for graceful degradation when an external data source is unavailable.

### Phase 4: Quantitative Forecasting Engine

The quantitative engine currently provides the statistical baseline used by later forecasting stages.

Key components:

* `QuantBaseline`
* `QuantForecaster`
* `EWMAVolatility`
* Historical simple and log returns
* Optional momentum drift
* Monte Carlo simulation

Default configuration:

* **10,000 Monte Carlo paths**
* **5 trading-day forecast horizon**
* **EWMA volatility λ = 0.94**
* Deterministic seeds for reproducibility

The quantitative baseline produces:

* P10 price
* P50 price
* P90 price
* Probability of upside
* Probability of flat outcome
* Probability of downside
* Expected return
* Weekly volatility

The volatility model is intentionally modular so that alternative models such as GARCH can be introduced later without redesigning the forecasting interface.

---

## Quick Start

### 1. Install Dependencies

```bash
# Clone and navigate to the project
cd stock-analysis-prediction-multi-agent

# Install the package
pip install -e .

# Or install dependencies manually
pip install yfinance feedparser httpx pandas numpy pydantic pydantic-settings structlog alembic python-dotenv
```

If the project is configured with `uv`, the recommended development installation is:

```bash
uv sync --dev
```

### 2. Configure Environment

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key

DATABASE_PATH=data/stock_analysis.db

LOG_LEVEL=INFO
LOG_FORMAT=json
```

Optional model configuration can be added as the LLM layer evolves:

```env
GEMINI_MODEL_PRIMARY=gemini-3.5-flash-lite
GEMINI_MODEL_CRITIC=gemini-3.5-flash-lite
GEMINI_MODEL_FALLBACK=gemini-3.5-flash-8b-lite
```

### 3. Run the Demo

```bash
python demo.py
```

The demo exercises the currently implemented Phase 1–4 functionality and uses real external data for the applicable data-collection components.

### 4. Run Tests

```bash
pytest tests/ -v
```

### 5. Run Individual Phase Tests

```bash
# Market data
pytest tests/test_resolver.py \
       tests/test_collector.py \
       tests/test_calendar.py \
       tests/test_price_cache.py \
       tests/test_indicators.py -v

# Fundamental / News / Market Context
pytest tests/test_fundamentals.py \
       tests/test_news.py \
       tests/test_market_context.py \
       tests/test_data_cache.py \
       tests/test_collector.py -v

# Quantitative forecasting
pytest tests/test_quant_forecast.py -v
```

---

## CLI Usage

```bash
# Show help
python -m stock_analysis.cli.main --help

# Resolve a symbol
python -m stock_analysis.cli.main resolve RELIANCE

# Get price data
python -m stock_analysis.cli.main price RELIANCE --days 30

# Get fundamentals
python -m stock_analysis.cli.main fundamentals RELIANCE

# Get news
python -m stock_analysis.cli.main news RELIANCE

# Get market context
python -m stock_analysis.cli.main context RELIANCE --sector Energy

# Run quantitative forecast
python -m stock_analysis.cli.main forecast RELIANCE --seed 42
```

---

## Project Structure

```text
stock_analysis/

├── cli/
│   └── main.py              # CLI commands
│
├── config/
│   └── settings.py          # Pydantic Settings
│
├── database/
│   ├── database.py          # SQLite wrapper
│   └── migrations.py        # Alembic migrations
│
├── data/                    # Phase 3: Data Layer
│   ├── cache.py             # Generic DataCache
│   ├── collector.py         # Unified DataCollector
│   ├── fundamentals.py      # FundamentalsCollector
│   ├── market_context.py    # MarketContextCollector
│   └── news.py              # NewsCollector
│
├── indicators/
│   └── technical.py         # Technical indicators
│
├── llm/
│   ├── factory.py           # LLM Factory
│   └── models.py            # LLM models
│
├── logging/
│   └── __init__.py          # Structured logging
│
├── market/                  # Phase 2: Market Data
│   ├── calendar.py          # TradingCalendar
│   ├── cache.py             # PriceCache
│   ├── collector.py         # MarketPriceCollector
│   └── resolver.py          # SymbolResolver
│
├── quant/                   # Phase 4: Quant Forecasting
│   └── forecast.py          # QuantForecaster / volatility models
│
├── schemas/                 # Pydantic schemas
│   ├── base.py
│   ├── symbol.py
│   ├── forecast.py
│   └── ...
│
└── __init__.py
```

---

## Example Usage

### Basic Quantitative Forecast

```python
from stock_analysis import QuantForecaster
import numpy as np

# Historical prices (e.g. 1 year of daily prices)
prices = np.array([...])

forecaster = QuantForecaster(
    seed=42,
    n_paths=10000,
    horizon=5,
)

baseline = forecaster.forecast(prices)

print(f"P(up)={baseline.prob_up:.1%}")
print(f"P(down)={baseline.prob_down:.1%}")
print(f"Expected return: {baseline.expected_return_pct:.2f}%")
print(
    f"P10={baseline.p10_price:.2f}, "
    f"P50={baseline.p50_price:.2f}, "
    f"P90={baseline.p90_price:.2f}"
)
print(f"Weekly vol: {baseline.weekly_vol_pct:.2f}%")
```

### Full Data Pipeline

```python
from stock_analysis import Database, DataCollector
import asyncio


async def main():
    db = Database("data/stock_analysis.db")
    collector = DataCollector(db)

    complete = await collector.get_complete_async(
        "RELIANCE",
        sector="Energy",
    )

    print(
        f"Fundamentals: "
        f"{complete.fundamentals.sector if complete.fundamentals else 'N/A'}"
    )

    print(
        f"News: "
        f"{len(complete.news.items) if complete.news else 0} articles"
    )

    print(
        f"Market: "
        f"Nifty={complete.market_context.nifty_50 "
        f"if complete.market_context else 'N/A'}"
    )

    await collector.close()


asyncio.run(main())
```

### Technical Indicators

```python
from stock_analysis import compute_all_indicators
import pandas as pd

df = pd.DataFrame({
    "open": [...],
    "high": [...],
    "low": [...],
    "close": [...],
    "volume": [...],
})

indicators = compute_all_indicators(df)

if indicators:
    latest = indicators[-1]

    print(f"RSI: {latest.rsi_14:.1f}")
    print(f"MACD: {latest.macd:.3f}")
    print(f"BB: {latest.bb_lower:.1f} - {latest.bb_upper:.1f}")
```

---

## Test Coverage

The current implementation includes dedicated tests for the major Phase 1–4 components.

| Module                 | Coverage Focus                                |
| ---------------------- | --------------------------------------------- |
| SymbolResolver         | Fuzzy matching, caching, validation           |
| MarketPriceCollector   | OHLCV, caching, errors                        |
| TradingCalendar        | Holidays, trading days                        |
| PriceCache             | TTL, expiry, clearing                         |
| TechnicalIndicators    | Technical indicator calculations              |
| FundamentalsCollector  | Fundamental fields, partial data, errors      |
| NewsCollector          | RSS, deduplication, old news, HTML stripping  |
| MarketContextCollector | Indices, sector mapping, errors               |
| DataCache              | Generic cache operations                      |
| DataCollector          | Unified collection and caching                |
| QuantForecaster        | Monte Carlo, probabilities, seeds, volatility |

The current test suite covers the implemented Phase 1–4 functionality.

---

## Quantitative Forecast Baseline

The quantitative forecast is deliberately treated as a **baseline**, not as the final prediction.

The future multi-agent system will compare LLM-driven analysis against this baseline to determine whether the additional reasoning provides measurable value.

The baseline currently uses:

```text
Historical Prices
      ↓
Historical Returns
      ↓
EWMA Volatility
      ↓
Optional Momentum Drift
      ↓
Monte Carlo Simulation
      ↓
Probabilistic Forecast
```

This separation allows the system to measure:

```text
Quantitative Baseline
        vs
LLM-Adjusted Forecast
        vs
Actual Market Outcome
```

This comparison is important for preventing the LLM layer from being treated as inherently superior simply because it provides a more sophisticated narrative.

---

## Adding Custom Volatility Models

The volatility interface is designed to support alternative models such as GARCH.

```python
from stock_analysis import VolatilityModel
import numpy as np


class GARCHVolatility(VolatilityModel):

    def estimate(self, returns: np.ndarray) -> float:
        # Implement GARCH(1,1) variance estimation
        pass

    def simulate_paths(
        self,
        returns,
        n_paths,
        horizon,
        drift,
        rng,
    ):
        # Implement GARCH path simulation
        pass


forecaster = QuantForecaster(
    volatility_model=GARCHVolatility()
)

baseline = forecaster.forecast(prices)
```

The purpose of this interface is to allow the quantitative forecasting layer to evolve independently from the higher-level analyst and LLM architecture.

---

## Future Multi-Agent Architecture

### Phase 5: LLM Analyst Nodes

Introduce specialized analysts operating over structured data:

* Fundamental Analyst
* Technical Analyst
* News Analyst
* Market Context Analyst

Each analyst should produce structured, evidence-backed outputs rather than unrestricted prose.

### Phase 6: Signal Synthesis

Combine:

* Quantitative baseline
* Fundamental signals
* Technical signals
* News signals
* Market-context signals

The objective is to produce a structured forecast input for the LLM predictor while preserving the quantitative baseline as an explicit reference point.

### Phase 7: LangGraph Workflow Orchestration

Introduce LangGraph for orchestration of the multi-agent workflow.

The workflow should support:

* Explicit state transitions
* Parallel analyst execution
* Structured state
* Failure handling
* Retry boundaries
* Checkpointing where appropriate
* Forecast generation
* Critic review
* Final forecast generation

### Phase 8: Evaluation, Backtesting & Postmortem

After the forecast horizon completes, compare the immutable forecast against actual market outcomes.

Evaluation should measure items such as:

* Direction accuracy
* Forecast interval coverage
* P10/P50/P90 calibration
* Probability calibration
* Expected-return error
* Baseline versus LLM-adjusted performance

The system should then generate a structured postmortem identifying where the forecast differed from the observed outcome.

### Phase 9: Controlled Calibration / Adaptive Learning

The postmortem should feed into a **bounded calibration process** rather than directly modifying the forecasting model after every prediction.

Potential calibration areas include:

* Signal weights
* Confidence calibration
* Probability adjustments
* Regime-specific parameters
* Analyst contribution
* Quantitative baseline adjustments

Changes should be:

* Evidence-based
* Versioned
* Auditable
* Bounded
* Evaluated against historical data before adoption

The system should preserve previous configurations so that calibration changes can be compared and rolled back.

---

## Forecast Immutability & Point-in-Time Integrity

A central requirement of the system is that a historical forecast represents exactly what the system knew when the forecast was generated.

Therefore:

```text
Forecast Created
      ↓
Snapshot Stored
      ↓
Snapshot Never Modified
      ↓
Future Market Data Arrives
      ↓
Actual Outcome Recorded Separately
      ↓
Evaluation
```

Historical evaluation must not accidentally use information that became available after the original forecast timestamp.

This is particularly important for:

* News
* Fundamental data
* Market context
* Price data
* Analyst inputs
* LLM-generated reasoning

Without point-in-time integrity, backtest and evaluation results can become misleading.

---

## Development

### Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=stock_analysis
```

Specific test:

```bash
pytest tests/test_config.py -v
```

### Code Quality

```bash
ruff check src tests

ruff format src tests

mypy src
```

---

## Database & Migrations

The project uses SQLite for the current persistence layer and Alembic for schema migrations.

Initialize the database:

```bash
stock-analysis init
```

Run migrations:

```bash
stock-analysis migrate
```

Check application status:

```bash
stock-analysis status
```

Show configuration:

```bash
stock-analysis config
```

---

## Roadmap

```text
Phase 1  ─ Foundation                              ✅
Phase 2  ─ Market Data Layer                      ✅
Phase 3  ─ Fundamental / News / Context Layer     ✅
Phase 4  ─ Quantitative Forecasting               ✅
Phase 5  ─ LLM Analyst Nodes                      ⏳
Phase 6  ─ Signal Synthesis                       ⏳
Phase 7  ─ LangGraph Workflow                    ⏳
Phase 8  ─ Evaluation / Backtesting / Postmortem ⏳
Phase 9  ─ Calibration / Adaptive Learning        ⏳
```

The goal is not simply to build an LLM-powered stock analysis application.

The goal is to build a **measurable forecasting system** where:

```text
Data
 ↓
Specialized Analysis
 ↓
Quantitative Baseline
 ↓
LLM Reasoning
 ↓
Critique
 ↓
Forecast
 ↓
Immutable Snapshot
 ↓
Actual Outcome
 ↓
Evaluation
 ↓
Postmortem
 ↓
Controlled Learning
 ↓
Improved Future Forecast
```

Each additional layer should be measurable against the previous layer rather than being added purely for architectural complexity.

---

## License

MIT License - see `LICENSE` file for details.

---

## Disclaimer

**This software is for educational and research purposes only. It does not constitute financial advice. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.**
