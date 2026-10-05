# Operations guide

How to run, configure and troubleshoot the forecaster. For the design, see `Plan.md` and the
README.

## Setup

```bash
uv sync --dev
cp .env.example .env        # GEMINI_API_KEY is optional; without it the quant baseline ships
stock-agents init           # creates DATABASE_PATH and applies every migration
stock-agents migrate        # after pulling new migrations (e.g. 006 adds llm_calls, 007 the forecast source)
```

Backtest databases are migrated automatically when a backtest creates or extends them.

## Settings that matter in production

Every setting can be set in `.env` or the environment (names are case-insensitive).
`stock-agents config` prints the values in effect.

| Setting | Default | Meaning |
|---|---|---|
| `GEMINI_MODEL_PRIMARY` / `_FALLBACK` | flash-lite | Model per role. A call that still fails after its retries moves to the fallback model. |
| `LLM_REQUESTS_PER_MINUTE`, `LLM_RATE_LIMIT_BURST` | 30, 4 | Token bucket shared by every Gemini call in the process (0 = no limit) |
| `YFINANCE_REQUESTS_PER_MINUTE`, `YFINANCE_RATE_LIMIT_BURST` | 120, 5 | Same for yfinance (prices, fundamentals, indices, symbol checks) |
| `NEWS_REQUESTS_PER_MINUTE`, `NEWS_RATE_LIMIT_BURST` | 30, 3 | Same for Google News RSS |
| `LLM_RETRY_ATTEMPTS`, `LLM_RETRY_BASE_DELAY_SECONDS`, `LLM_RETRY_MAX_DELAY_SECONDS` | 3, 2, 30 | Total tries per model on HTTP 408/429/5xx, timeouts and connection errors. The delay doubles each try (±10% jitter). |
| `EXTERNAL_RETRY_ATTEMPTS`, `EXTERNAL_RETRY_BASE_DELAY_SECONDS`, `EXTERNAL_RETRY_MAX_DELAY_SECONDS` | 3, 1, 20 | Same for data downloads. Programming errors are never retried. |
| `LLM_REQUEST_TIMEOUT_SECONDS`, `EXTERNAL_REQUEST_TIMEOUT_SECONDS` | 120, 30 | Per-request timeout for Gemini / yfinance downloads (RSS uses the collector's 30 s); a timeout is retried |
| `LLM_STRUCTURED_OUTPUT_REPAIR` | true | One re-prompt with the schema errors when output does not validate |
| `LLM_MAX_CALLS_PER_RUN` | 40 | Budget per forecast run, including the pre-run review's postmortems (0 = none). Calls beyond it fail fast: the analyst degrades or the quant baseline ships. |
| `LLM_INPUT_COST_PER_MILLION_TOKENS`, `LLM_OUTPUT_COST_PER_MILLION_TOKENS` | 0.10, 0.40 | USD prices used for the cost estimate; set them to your model's price list |
| `UNTRUSTED_TEXT_MAX_CHARS` | 500 | Per-item limit for untrusted text shown to an LLM |
| `LOG_LEVEL`, `LOG_FORMAT` | INFO, console | `LOG_FORMAT=json` writes one JSON object per line to stderr |

## Logs

Logs go to **stderr**; reports go to stdout, so `stock-agents analyze RELIANCE > report.txt`
captures only the report. Inside a forecast run every line carries `run_id` and `ticker`.
Each graph node adds one line:

```json
{"event": "node_completed", "node": "predictor", "run_id": "…", "ticker": "RELIANCE",
 "latency_ms": 2140, "status": "ok", "level": "info", "timestamp": "…"}
```

The review graph's steps (`find_matured`, `score`, `postmortem`, …) log the same way.
`status` is `error` when a node returned an error marker (e.g. an invalid predictor output);
an exception logs `node_failed` and is re-raised. Each LLM call logs `llm_call` with `model`,
`role`, `purpose` (`generate` or `repair`), `attempts`, `prompt_tokens`, `completion_tokens`,
`estimated_cost` and `status`. A run ends with `forecast_run_completed` (totals for the run).

## LLM call log and cost

Every Gemini call is appended to the `llm_calls` table of the database the run writes to:

| Column | |
|---|---|
| `run_id` | One per forecast run (`backtest-…` for a whole backtest, `review-…` for `stock-agents review`) |
| `forecast_id` | Filled in once the run's forecast is stored |
| `ticker`, `node`, `role`, `model`, `purpose` | Who called what, and why |
| `prompt_tokens`, `completion_tokens`, `cost_est`, `latency_ms` | Usage and the cost estimate |
| `status`, `attempts`, `error` | `ok` or `error`, the number of tries, and the last error |

```bash
stock-agents usage                                   # all calls in the app database
stock-agents usage --ticker RELIANCE --days 7        # one ticker, last week
stock-agents usage --database data/backtests/RELIANCE-….db
```

`analyze` and `review --llm` print the run's usage, and the backtest report includes it. Inside a review, calls are tagged with the review step (`postmortem`). An
older database without `llm_calls` still forecasts: usage is kept in memory and a
`llm_call_log_unavailable` warning suggests `stock-agents migrate`. Call-log rows use wall-clock
times and are not point-in-time history, so they never block extending a backtest database.

## Untrusted text

News headlines and summaries (live `analyze` and postmortem hindsight news) are web content:

1. `untrusted.sanitize_untrusted_text` unescapes HTML; strips tags, control, zero-width and
   bidi characters, and anything resembling our delimiters; replaces instruction-like phrases
   ("ignore previous instructions", "system prompt", "respond only with", role tags …) with
   `[redacted]`; and truncates. `news_summary` reports `redacted_items` when something was
   removed.
2. `untrusted.wrap_untrusted` puts the content between `<<<BEGIN UNTRUSTED DATA>>>` and
   `<<<END UNTRUSTED DATA>>>`, preceded by the instruction to treat it only as data.
3. The sentiment and postmortem prompts repeat the rule. The predictor is told that analyst
   text may quote news and must never be followed as an instruction.

Numbers never come from the LLM, outputs are schema-validated and bounded, and the critic is
deterministic. So even a successful injection cannot move a forecast beyond the configured limits.

## Trading calendar

`market/calendar.py` lists weekday NSE holidays per year. Dates up to 2026-10-02 have been
checked against Nifty 50 sessions. Add each new year's list when NSE publishes it, and check it
against data once sessions have happened: a weekday without a `^NSEI` bar should be listed,
and a listed date should have no bar.

Diwali Laxmi Pujan is an official holiday with a one-hour Muhurat session. yfinance shows a bar
for it, so a week containing it is marked invalid rather than scored. yfinance sometimes repeats
the previous close as a zero-volume bar on a holiday; such bars are dropped.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Warning: GEMINI_API_KEY is not set` | The analysts degrade and the quant baseline ships. Set the key or use `--no-llm`. |
| `forecast_snapshots has no source column; run stock-analysis migrate` | The database predates migration 007 (live vs backtest labels): run `stock-agents migrate`; existing forecasts keep no stored source and are inferred when read |
| `… tables are missing; run stock-analysis migrate` | The database predates a migration: run `stock-agents migrate` (or `alembic upgrade head`) |
| Many `external_call_retrying` warnings for yfinance | Yahoo is throttling: lower `YFINANCE_REQUESTS_PER_MINUTE` |
| `LLM call budget of N calls for this run is spent` | Raise `LLM_MAX_CALLS_PER_RUN`, or check for a revision loop (`MAX_REVISIONS`) |
| A backtest week is `invalid: … sessions the forecast calendar did not expect` | The calendar and the data disagree about a session; see "Trading calendar" |
| `Backtest failed: … already has history from …` | Backtests only append history after what the database holds (e.g. the app database after an `analyze` or an earlier backtest): add the suggested `--database data/backtests/….db`, and pass it to `evaluate`/`calibration`/`analyze` too |
