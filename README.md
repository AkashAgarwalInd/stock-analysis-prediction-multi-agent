# Multi-Agent Indian Stock Analyst + Adaptive Weekly Forecaster

A sophisticated system for analyzing Indian NSE stocks and producing probabilistic next-5-trading-day forecasts using a multi-agent architecture with LLM-enhanced quantitative baselines.

## Architecture Overview

```
User Query
    ↓
Symbol Resolver
    ↓
Previous Forecast Review
    ↓
Market/Fundamental/News/Context Collectors
    ↓
Parallel Analysts
    ↓
Quantitative Forecast Baseline
    ↓
LLM Predictor
    ↓
Critic
    ↓
Final Report
    ↓
Immutable Forecast Snapshot
    ↓
Later: Actual-vs-Forecast Evaluation
    ↓
Postmortem
    ↓
Controlled Calibration/Learning
    ↓
Future Forecast
```

## Key Principles

1. **Numbers from Python/tools, never from LLM** - All calculations done in Python
2. **LLMs interpret structured facts** - LLMs only reason over structured data
3. **Pydantic structured output** - Every LLM output validated via Pydantic
4. **No large DataFrames in state** - Keep LangGraph state minimal
5. **Immutable forecasts** - Once created, forecasts never change
6. **Full auditability** - Complete audit trail for every decision
7. **Measured evaluation** - Compare forecasts against actual prices
8. **Baseline comparison** - LLM-adjusted vs quantitative baseline
9. **No unsubstantiated claims** - Evidence-based only
10. **Bounded learning** - One bad week doesn't change the model
11. **Point-in-time integrity** - Strict historical data integrity
12. **Educational disclaimer** - Every report includes non-financial-advice notice

## Quick Start

### Installation

```bash
# Clone and navigate
cd stock-analysis-prediction-multi-agent

# Install with uv (recommended) or pip
uv sync --dev
# or
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your Gemini API key
```

### Configuration

Required environment variables in `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

Optional overrides:
- `GEMINI_MODEL_PRIMARY` - Main forecasting model (default: gemini-3.5-flash-lite)
- `GEMINI_MODEL_CRITIC` - Critic model (default: gemini-3.5-flash-lite)
- `GEMINI_MODEL_FALLBACK` - Fallback model (default: gemini-3.5-flash-8b-lite)
- `DATABASE_PATH` - SQLite database location (default: data/stock_analysis.db)
- `LOG_LEVEL` - Logging level (default: INFO)

### Commands

```bash
# Initialize database and run migrations
stock-analysis init

# Run migrations
stock-analysis migrate

# Show application status
stock-analysis status

# Show configuration
stock-analysis config
```

## Project Structure

```
src/stock_analysis/
├── cli/           # Typer CLI commands
├── config/        # Pydantic-settings configuration
├── database/      # SQLite + Alembic migrations
├── llm/           # Gemini LLM factory
├── logging/       # Structured logging (structlog)
├── schemas/       # Pydantic models
└── __init__.py
```

## Development

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src/stock_analysis

# Specific test file
pytest tests/test_config.py -v
```

### Code Quality

```bash
# Lint
ruff check src tests

# Format
ruff format src tests

# Type check
mypy src
```

## Current Phase

**Phase 1: Foundation** ✅
- Configuration management
- Environment configuration
- Structured logging
- SQLite database with migrations
- Basic Pydantic schemas
- Gemini LLM factory
- CLI skeleton

## Future Phases

- **Phase 2**: Symbol resolution, NSE data collection
- **Phase 3**: Market/fundamental/news collectors
- **Phase 4**: Parallel analysts
- **Phase 5**: Quantitative forecast baseline
- **Phase 6**: LLM predictor + Critic
- **Phase 7**: Final report + Immutable snapshot
- **Phase 8**: Evaluation + Postmortem
- **Phase 9**: Calibration/Learning

## License

MIT License - see LICENSE file for details.

## Disclaimer

**This software is for educational and research purposes only. It does not constitute financial advice. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.**