# Multi-Agent Indian Stock Analyst + Adaptive Weekly Forecaster

A research system that forecasts the **next five NSE trading days** for an Indian stock as a
probability distribution, then checks itself: every forecast is stored immutably, scored
against what actually happened, diagnosed, and used, within strict bounds, to adjust later
forecasts. It is built on LangGraph, Gemini, SQLite and a Monte Carlo quant baseline.

> **Educational and research use only.** Nothing here is investment advice. Forecasts are
> probabilistic estimates that may be wrong. Never make financial decisions based on this output.

```text
Forecast → immutable snapshot → actual outcome → scoring → postmortem → bounded calibration → next forecast
```

## What a run shows

`stock-agents analyze RELIANCE` prints one report containing:

1. **Last forecast vs actual**: the previous forecast's quant baseline and final forecast next
   to the actual close, the errors, the up/flat/down Brier loss, LLM value added and the postmortem cause.
2. **System adaptation**: the calibration in force, what changed and why, and its effect on
   this forecast's P10–P90 range.
3. **Current analysis**: the market regime and four independent analysts (technical,
   fundamental, sentiment, context).
4. **Quant baseline**: the EWMA-volatility Monte Carlo distribution (P10/P50/P90, P(up/flat/down)).
5. **Final forecast**: the baseline after bounded, evidence-backed LLM adjustments that a
   deterministic critic has checked.
6. **Risks, invalidation triggers, track record and benchmark comparison** (naive flat,
   uncalibrated quant, calibrated quant and final forecast), followed by the disclaimer.

Every number in the report comes from code, never from the LLM, and the report writer refuses
any number that cannot be traced back to the stored snapshot.

## Quick start

```bash
uv sync --dev                 # or: pip install -e ".[dev]"
cp .env.example .env          # add GEMINI_API_KEY for the LLM path (optional)
stock-agents init             # create data/stock_analysis.db and run migrations

python demo.py --no-llm       # full demo on RELIANCE, INFY and HDFCBANK, no API key needed
```

The Plan.md §76 acceptance flow runs as written against the app database:

```bash
stock-agents backtest --ticker RELIANCE.NS --weeks 12   # 12 weeks, point-in-time, scored and learned
stock-agents evaluate --ticker RELIANCE.NS              # track record vs benchmarks
stock-agents calibration --ticker RELIANCE.NS           # calibration versions
stock-agents analyze RELIANCE                           # today's forecast, reviewed and calibrated
```

A backtest only appends history after what its database already holds (any ticker), so it
works on a fresh app database. To run another one later, pass `--database
data/backtests/<name>.db` to `backtest` and the same `--database` to the other commands; the
error message suggests a path.

The demo runs the same flow on three stocks, each in its own database under
`data/demo/<time>/<TICKER>/`, checks that every analysis report shows the 15 items of §76 and
prints a summary.
Use `python demo.py --tickers TCS ITC --weeks 8`, `--llm` and `--show-reports` to vary it.

Without `GEMINI_API_KEY` the LLM path still runs: the analysts degrade and the quant baseline
ships. `--no-llm` skips the LLM entirely.

## CLI

`stock-agents` and `stock-analysis` are the same command.

| Command | What it does |
|---|---|
| `init` / `migrate` | Create the database / apply Alembic migrations |
| `analyze QUERY [--database P] [--no-llm] [--no-review] [-o report.md]` | Live forecast for a ticker or company name (`"tata motors"`) |
| `backtest --ticker T --weeks N [--end D] [--no-llm] [--database P] [-o F]` | Weekly forecasts simulated point-in-time, each scored and learned from in order (default: into the app database) |
| `review [--ticker T \| --all] [--no-llm]` | Score matured forecasts, write postmortems, update lessons and calibration |
| `history --ticker T [--database P]` | Stored forecasts with their outcomes |
| `evaluate [--ticker T] [--database P]` | Track record (8/26 weeks, all time) against the benchmarks, plus reliability |
| `calibration [--ticker T] [--database P]` | Calibration versions and predicted-vs-observed probability tables |
| `scorecard [--ticker T] [--database P]` | Analyst hit rates, quality by regime, decision-engine evaluation |
| `usage [--ticker T] [--days N] [--database P]` | LLM calls, tokens and estimated cost from the call log |
| `status` / `config` | Settings in effect (the API key is masked) |

## How it works

```text
review_matured → memory_loader → 4 analysts (parallel) → guardrails → decision_engine
  → quant_baseline (calibrated; the uncalibrated shadow is kept) → adjustment_gate
  → predictor ⇄ critic (bounded revisions) → final_forecast → forecast_snapshot → memory_writer
```

- **Data**: yfinance prices, fundamentals and market context, plus Google News RSS. All sources
  are cached in SQLite and rate limited. A failing source degrades the run instead of stopping it.
- **Quant baseline** (`quant/`): EWMA volatility (λ=0.94) Monte Carlo with a fixed seed. The
  exact close series is stored, so any baseline can be reproduced bit for bit.
- **LLM layer** (`llm/`): Gemini through `LLMFactory.generate_structured`. Every call returns a
  Pydantic model. LLMs only interpret supplied facts and may shift the baseline's probabilities
  by at most `PREDICTOR_MAX_PROB_SHIFT`, which a deterministic critic enforces.
- **Snapshots** (`snapshots/`, `database/forecast_store.py`): immutable and hash-checked on every
  load. Corrections are new versions, and evaluation always scores the original.
- **Review graph** (`langgraph/review_graph.py`): `find_matured → score → postmortem →
  learning → calibration → scorecards → track_record`. It runs before every forecast and on
  `review`.
- **Learning** (`learning/`): postmortems separate what was knowable at forecast time from
  hindsight. Lessons need repeated, non-overlapping evidence. Calibration (a volatility
  multiplier and a P50 bias) is shrunk, bounded, versioned with its evidence, and only starts
  after 8 independent scored forecasts.
- **Evaluation**: Brier score, pinball loss, 80% coverage, PIT, sharpness, reliability tables
  and LLM/calibration value added with confidence intervals. No verdict is given below the
  minimum sample sizes.

## Point-in-time integrity

- A backtest runs week by week on a simulated clock: forecast at the as-of close, score at the
  target's data time, learn, then move to the next week. Forecasts see only
  `HistoricalMarketData.as_of(d)`, and evaluation raises `LookaheadError` beyond the clock.
- Memory, lessons, calibration, scorecards and the track record are all read with an `as_of`
  cutoff, so records are stamped with the simulated time (tests in `tests/phase10` and
  `tests/phase18/test_leakage.py`).
- Fundamentals and news are not point-in-time, so the backtest disables those analysts. LLM
  backtests may still carry pretraining knowledge of past dates, and the report says so.
- The NSE holiday calendar (`market/calendar.py`) has been checked against Nifty 50 sessions
  up to 2026-10-02. Later dates follow the published list and must be maintained yearly.

## Reliability and safety (Phase 18)

| Concern | What the code does |
|---|---|
| Retries | Every Gemini and yfinance request has a timeout. Gemini calls retry 429/5xx/timeouts with exponential backoff and jitter, then fall back to `GEMINI_MODEL_FALLBACK`. yfinance and RSS calls retry with backoff. |
| Rate limits | A token bucket per service (`llm`, `yfinance`, `news`), configured in settings |
| Structured output | One repair re-prompt with the validation errors, then the caller's fallback (degraded analyst or quant baseline) |
| Prompt injection | News text is sanitized (markup, invisible characters and instruction-like phrases removed) and passed only inside `<<<BEGIN/END UNTRUSTED DATA>>>` delimiters with a "data only" instruction. No node has tool access. |
| Cost | Each call's tokens, latency and estimated cost go to `llm_calls`, linked to the forecast. Each run has a call budget (`LLM_MAX_CALLS_PER_RUN`). See `stock-agents usage`. |
| Logging | structlog to stderr (`LOG_FORMAT=json` for JSON). Each node logs `node_completed` with `run_id`, `ticker`, `node`, `latency_ms` and `status`. |

See [docs/operations.md](docs/operations.md) for the settings, the call log and troubleshooting.

## Project layout

```text
src/stock_analysis/
  market/ data/        collectors, symbol resolver, NSE calendar, caches
  indicators/ quant/   technical indicators, Monte Carlo baseline
  llm/                 Gemini factory, usage tracking
  langgraph/           forecast graph, review graph, run_forecast
  snapshots/           snapshot builder, report writer, reproduction
  review/              outcome scoring and metrics
  learning/            postmortems, lessons, calibration, scorecards, track record
  backtest/            point-in-time historical simulation
  analysis/            the analyze command
  database/            SQLite stores (Alembic migrations in alembic/versions)
  reliability.py       retries and rate limits
  untrusted.py         prompt-injection protection
  demo.py              the multi-stock demo (python demo.py)
tests/                 phase tests (tests/phaseN), helpers, fixtures
```

## Development

```bash
pytest                          # full suite (offline; network calls are faked)
pytest tests/phase18/           # one phase
ruff check src tests            # lint (some style rules are deferred in pyproject.toml)
alembic upgrade head            # migrations
```

`Plan.md` is the design spec and its phase list is the build order. Phases 1–18 are implemented.

## Limitations

- Free data sources: yfinance can lag, rate limit or carry placeholder bars. Outcomes that the
  data cannot support are marked invalid rather than scored.
- Google News RSS covers only about the last 14 days, so historical news is not available for backtests.
- Probabilities depend on the volatility model. A short track record does not show skill, and
  the reports refuse to draw conclusions below the configured sample sizes.

## License

MIT, see `LICENSE`.

## Disclaimer

This output is generated by an automated system for educational and demonstration purposes
only. It is not investment advice, and forecasts are probabilistic estimates that may be wrong.
Do not make financial decisions based on this output.
