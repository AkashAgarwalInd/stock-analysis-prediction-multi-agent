# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Multi-agent stock analysis and probabilistic next-5-trading-day forecaster for Indian NSE stocks (LangGraph + Gemini + SQLite + Monte Carlo quant baseline). Educational/research only — output must never be framed as buy/sell/hold advice.

`Plan.md` is the authoritative spec (≈4300 lines). Its phase list (section "Phase 1 — Foundation" … "Phase 18 — Hardening", near the end of the file) is the build order: complete a phase with passing tests before starting the next, and ask before deviating from the specified architecture. Note the README uses an older, different phase numbering — trust `Plan.md`.

## Commands

```bash
uv sync --dev                       # install (or: pip install -e ".[dev]")
pytest                              # all tests (config in pyproject: -v --tb=short, asyncio_mode=auto)
pytest tests/phase5/                # one phase's tests
pytest tests/test_quant_forecast.py::TestClassName::test_name   # single test
ruff check src tests && ruff format src tests
mypy src                            # strict mode
python demo.py                      # end-to-end demo of data + quant layers (hits real yfinance/RSS)
alembic upgrade head                # DB migrations (alembic/versions/)
python -m stock_analysis.cli.main --help   # Typer CLI: init, migrate, review, status, config
```

The README lists CLI commands like `resolve`, `price`, `forecast` — these are not implemented in `cli/main.py`.

`ruff check src tests` passes, but only because ~250 style findings in `src/` are deferred via `ignore`/`per-file-ignores` (see the comments in `pyproject.toml`) — "clean" here does not mean the codebase satisfies the selected rule set. `ruff format` would reformat 23 files, so it is not part of the gate yet either. `mypy src` runs to completion but still reports ~110 strict-mode errors (mostly bare `dict`/`list` type args); don't treat it as a gate yet.

`tests/conftest.py` sets `APP_ENV=testing` and a dummy `GEMINI_API_KEY`, and clears the `get_settings()` lru_cache around every test. `tests/test_gemini_smoke.py` is skipped wholesale (google.generativeai / pkg_resources issue). Older phase tests live flat in `tests/`; Phase 5+ tests live in `tests/phaseN/` (`tests/phase7` holds both the memory tests and the snapshot tests the user called "Phase 7"; `tests/phase9` is the outcome scorer). Forecast fixtures (`migrated_db`, `store`, `memory_store`, `outcome_store`, `snapshot`, `adjusted_state`, ...) live in `tests/conftest.py`, with helpers in `tests/forecast_helpers.py` (fake LLMs) and `tests/outcome_helpers.py` (synthetic actual prices); import them as `tests.forecast_helpers`.

## Architecture

Everything is under `src/stock_analysis/`. Data flows in layers:

1. **Data collection (deterministic, cached)** — `market/` (SymbolResolver, MarketPriceCollector, TradingCalendar, PriceCache) and `data/` (Fundamentals/News/MarketContext collectors behind a unified async `DataCollector`). All caches are SQLite-backed with per-type TTLs; collectors degrade gracefully when a source fails rather than raising.
2. **Indicators & quant** — `indicators/technical.py` (`compute_all_indicators`) and `quant/forecast.py` (`QuantForecaster` → `QuantBaseline` with P10/P50/P90, prob_up/flat/down, weekly vol). Volatility is pluggable via the `VolatilityModel` ABC (default `EWMAVolatility`, λ=0.94); simulations take a seed for reproducibility.
3. **LLM layer** — `llm/factory.py` `LLMFactory` / `get_llm_factory()`; call `generate_structured(role=..., prompt=..., response_schema=<PydanticModel>)`. The JSON schema is appended to the prompt (not passed as Gemini's `response_schema`, which rejects free-form dict fields); API errors retry once on the FALLBACK model, while schema violations raise `ValidationError`. Temperature/max tokens default to settings. Model IDs come from settings (`GEMINI_MODEL_PRIMARY/CRITIC/FALLBACK`), never hardcoded.
4. **LangGraph workflow** — `langgraph/workflow.py` `build_workflow()` / `compile_graph(llm_factory=None)`. State is the Pydantic `GraphState` (`schemas/graph_state.py`, `extra="forbid"` — any new node output key needs a field there). Topology:
   `memory_loader` → `mark_collectors` → 4 parallel analysts (technical, fundamental, sentiment, context) → `guardrails` → `decision_engine` → `quant_baseline` → `adjustment_gate` → `predictor` → `critic` → `revision` → (back to `predictor` while the critic fails, else) `final_forecast` → `join` → `forecast_snapshot` → `memory_writer` → END.
   Collector data is pre-loaded into state as compact summaries before the graph runs. All nodes are module-level functions (or `make_*` factories when an LLM is injected) so they can be unit tested directly. The Phase 6 loop is bounded by `MAX_REVISIONS` / `PREDICTOR_MAX_PROB_SHIFT` (settings, defaults 2 / 0.15); after the budget is spent the quant baseline ships with `fallback_to_quant=True`. The critic is deterministic and judges the *raw* predictor dict, because the strict `PredictorResult` schema rejects bad output before a critic could see it. Phase 6 schemas (`PredictorResult`, `CriticResult`, `FinalForecast`) are in `schemas/forecast_pipeline.py`.
   **Phase 7 snapshots** (Plan.md calls this Phase 8): `forecast_snapshot` builds an immutable `ForecastSnapshot` (`schemas/snapshot.py`, via `snapshots/builder.py`) and renders the report (`snapshots/report.py`). It is persisted only when a `ForecastSnapshotStore` (`database/forecast_store.py`) is passed to `compile_graph`; production code should call `langgraph/runner.py` `run_forecast`, which always persists and fails if a completed forecast or its price history was not stored. With a store, the quant node also saves the exact close series (`price_history_snapshots`), so `snapshots/reproduce.py` `reproduce_quant_baseline` can re-run a stored baseline bit-for-bit.
   **Plan.md Phase 7 memory**: `database/memory_store.py` `MemoryStore` (same SQLite DB, migration `003`; not LangGraph Store) holds per-forecast `ForecastInsight`s, lessons with an append-only evidence log (status candidate/active/retired is derived; `LESSON_ACTIVATION_MIN_EVIDENCE`=3, 2 for earnings-type categories), and the `TrackRecord` (`outcome_metrics` stays `None` until outcomes are scored). `memory_loader` puts a `MemoryContext` in state; `build_predictor_prompt` adds a PRIOR CONTEXT section only when it has content; the context is recorded in the snapshot's `data_inputs["memory"]` and shown in the report. Memory failures never block a forecast. All memory reads take an `as_of`/`before` cutoff for point-in-time use.
   **Plan.md Phase 9 outcome scoring** (`review/`, migration `004`): `OutcomeReviewer.review_matured(now)` finds matured original forecasts without an outcome, fetches actual prices for exactly the `[as_of, target]` window (`review/prices.py`, yfinance with dividend/split columns) plus Nifty 50, and calls the pure `score_forecast` (`review/outcome_scorer.py`, metrics in `review/metrics.py`). Matured = target session closed + `OUTCOME_DATA_DELAY_MINUTES`. Actual return comes from one consistent fetched series and is applied to the snapshot's `last_close`, so later dividend/split re-adjustment of history cancels out; unexplained daily moves beyond `OUTCOME_MAX_DAILY_MOVE_PCT` are invalid. The headline loss is the up/flat/down Brier score (`llm_value_added = baseline_loss - final_loss`); quantile (pinball) value added is stored too. Scored/invalid outcomes are stored append-only in `forecast_outcomes`/`outcome_daily`; unresolved ones are not stored and are retried. `stock-analysis review` prints the "Last forecast vs actual" section (`review/report.py`).
5. **Guardrails** — `guardrails.py` `run_preflight_guardrails(state)` runs between analysts and the decision engine (missing reports, CRITICAL `DataGap`s, coverage, consensus confidence). On failure the graph returns a degraded `MIXED` regime with confidence 0.0 plus `guardrail_violations` in state, instead of raising.
6. **Decision engine** — `DecisionEngineInterface` in `schemas/analyst_reports.py` with a deterministic `RulesDecisionEngine`; outputs are bounded enums (`MarketRegime`, `RiskCategory`, `AnalystStance`, `AdjustmentGateDecision`), not free text.

### Schema gotchas

- `AnalystReport` validators enforce stance/confidence consistency (STRONG_* needs ≥0.7, BULLISH/BEARISH ≥0.5), length limits on key_points (200) / evidence (300), and reject generic risks like "market risk". Test fixtures must satisfy these.
- A plain-string gap is CRITICAL only if it contains the word "critical"; use a `DataGap`/dict with `severity="critical"` to block the run deliberately.
- `data_gaps` accepts strings, dicts, or `DataGap` objects (with `DataGapSeverity`) for backward compatibility; use `AnalystReport`'s `data_gaps` validator (it already normalizes to `DataGap`) when consuming them.
- Analyst/decision enums, `DataGap`, `AnalystReport` and `DecisionResult` are defined once in `schemas/analyst_reports.py`; `schemas/graph_state.py` re-exports them.
- `GraphState.decision` / `adjustment_gate_decision` are `DecisionResult` models inside nodes (use attributes, not `.get`), but nodes must return them as dicts (`model_dump(mode="json")`).
- Persistence: `database/database.py` SQLite wrapper + Alembic; settings via Pydantic Settings reading `.env` (see `.env.example`).
- Snapshot tables (migration `002`) reject UPDATE/DELETE via triggers; correct a forecast with `ForecastSnapshotStore.record_correction` (new version, original `made_at` kept). Each load re-checks a hash of the stored values. Evaluation must score originals (`list_for_ticker` returns version 1 by default).
- The report writer is deterministic and refuses numbers not traceable to the snapshot (`find_unsupported_numbers`). Bump `versions.py` `PREDICTOR_PROMPT_VERSION` when `build_predictor_prompt` changes; analyst prompt files and response schemas are versioned by content hash automatically.

## Design rules (from Plan.md / README)

- Numbers (indicators, probabilities, prices) come from Python code, never from the LLM; LLMs only interpret supplied structured facts.
- Every LLM call returns a Pydantic structured output — no regex parsing of free text.
- Keep LangGraph state compact: no DataFrames or article bodies in state; store in SQLite/cache and pass summaries/keys.
- Forecast snapshots are immutable once stored; evaluation must be point-in-time (no lookahead).
- LLM-adjusted forecasts are always compared against the quant baseline; calibration changes must be bounded and auditable.
- Use structlog (`stock_analysis.logging`), not `print` (ruff `T20` enforces this); no bare `except`; type hints and docstrings on public functions.
