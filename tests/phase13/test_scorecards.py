"""Plan.md Phase 13: analyst, regime and decision scorecards ("done when" historical scores
can be displayed and insufficient samples do not trigger weighting)."""

import math
import statistics
from datetime import UTC, date, datetime, timedelta

import pytest
from typer.testing import CliRunner

from stock_analysis.config.settings import get_settings
from stock_analysis.database import LearningStore
from stock_analysis.learning import (
    analyst_scorecards,
    build_scorecards,
    decision_evaluations,
    decision_outcomes_for,
    group_scores,
    hit_rate,
    render_scorecards,
)
from stock_analysis.review import score_forecast
from stock_analysis.schemas.analyst_reports import MarketRegime
from stock_analysis.schemas.learning import DecisionOutcome
from stock_analysis.schemas.outcome import OutcomeStatus
from stock_analysis.schemas.scorecard import ScoredForecast
from tests.learning_helpers import WEEKDAYS, evaluated_at, save_scored, snapshot_as_of, week
from tests.outcome_helpers import path_series

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _scored(i, *, ticker="RELIANCE", regime="trending_up", sector="Energy", hits=None, **kw):
    values = {
        "direction_correct": True,
        "in_80pct_band": True,
        "signed_error_pct": 1.0,
        "abs_error_pct": 1.0,
        "brier": 0.5,
        **kw,
    }
    return ScoredForecast(
        forecast_id=f"f{i}",
        ticker=ticker,
        sector=sector,
        market_regime=regime,
        as_of_date=date(2026, 1, 5) + timedelta(weeks=i),
        target_date=date(2026, 1, 12) + timedelta(weeks=i),
        evaluated_at=T0 + timedelta(weeks=i),
        analyst_hits=hits or {},
        **values,
    )


def _decision(i, value, *, decision="allow_adjustment", correct=True, adjusted=True):
    return DecisionOutcome(
        forecast_id=f"f{i}",
        decision_type="forecast_adjustment_gate",
        ticker="RELIANCE",
        as_of_date=date(2026, 1, 5),
        target_date=date(2026, 1, 12),
        decision_engine="adjustment_gate",
        decision_model="rules",
        decision_model_version="0.1.0",
        decision=decision,
        confidence=0.7,
        adjustment_applied=adjusted,
        fallback_to_quant=not adjusted,
        calibration_version=0,
        direction_correct=correct,
        in_80pct_band=True,
        final_loss=0.5,
        baseline_loss=0.5 + value,
        llm_value_added=value,
        llm_value_added_pinball=value / 2,
        recorded_at=T0,
    )


class TestHitRate:
    @pytest.mark.parametrize(
        ("hits", "n", "low", "high"),
        [
            (8, 14, 0.3259, 0.7862),
            (0, 5, 0.0, 0.4345),
            (5, 5, 0.5655, 1.0),
            (10, 20, 0.2993, 0.7007),
        ],
    )
    def test_wilson_interval(self, hits, n, low, high):
        h = hit_rate(hits, n)
        assert h.rate == pytest.approx(hits / n)
        assert (h.ci_low, h.ci_high) == (
            pytest.approx(low, abs=1e-4),
            pytest.approx(high, abs=1e-4),
        )

    def test_no_samples(self):
        assert hit_rate(0, 0).rate is None


class TestAnalystScorecards:
    def test_counts_only_analysts_that_made_a_call(self):
        forecasts = [
            _scored(0, hits={"technical": True, "context": False}),
            _scored(1, hits={"technical": False, "context": False}, regime="sideways"),
            _scored(2, hits={"technical": True}, ticker="TCS", sector="IT"),
        ]
        cards = {c.analyst: c for c in analyst_scorecards(forecasts)}

        tech = cards["technical"]
        assert (tech.pooled.hits, tech.pooled.n) == (2, 3)
        assert {k: (v.hits, v.n) for k, v in tech.by_ticker.items()} == {
            "RELIANCE": (1, 2),
            "TCS": (1, 1),
        }
        assert {k: (v.hits, v.n) for k, v in tech.by_regime.items()} == {
            "sideways": (0, 1),
            "trending_up": (2, 2),
        }
        assert {k: (v.hits, v.n) for k, v in tech.by_sector.items()} == {
            "Energy": (1, 2),
            "IT": (1, 1),
        }
        assert (cards["context"].pooled.hits, cards["context"].pooled.n) == (0, 2)
        # disabled / degraded analysts have no entry in analyst_hits, so no record
        assert cards["sentiment"].pooled.n == 0 and cards["sentiment"].pooled.rate is None

    def test_insufficient_samples_never_make_weighting_eligible(self, monkeypatch):
        monkeypatch.setenv("ANALYST_WEIGHTING_ENABLED", "true")
        get_settings.cache_clear()
        few = [_scored(i, hits={"technical": True}) for i in range(19)]
        card = analyst_scorecards(few)[0]
        assert not card.weighting_eligible
        assert "only 19 calls" in card.weighting_note
        many = [_scored(i, hits={"technical": True}) for i in range(20)]
        assert analyst_scorecards(many)[0].weighting_eligible

    def test_weighting_flag_off_by_default(self):
        many = [_scored(i, hits={"technical": True}) for i in range(40)]
        card = analyst_scorecards(many)[0]
        assert not card.weighting_eligible
        assert "disabled" in card.weighting_note


class TestGroupScores:
    def test_hand_computed_regime_scores(self):
        forecasts = [
            _scored(0, signed_error_pct=2.0, abs_error_pct=2.0, brier=0.2),
            _scored(
                1,
                signed_error_pct=-4.0,
                abs_error_pct=4.0,
                brier=0.6,
                direction_correct=False,
                in_80pct_band=False,
            ),
            _scored(2, regime="sideways", signed_error_pct=1.0, abs_error_pct=1.0, brier=0.4),
        ]
        scores = {g.group: g for g in group_scores(forecasts, lambda f: f.market_regime)}
        up = scores["trending_up"]
        assert up.n == 2
        assert (up.direction.hits, up.direction.n) == (1, 2)
        assert up.mean_abs_error_pct == pytest.approx(3.0)
        assert up.mean_signed_error_pct == pytest.approx(-1.0)
        assert up.brier == pytest.approx(0.4)
        assert up.coverage_80pct == pytest.approx(0.5)
        assert scores["sideways"].n == 1


class TestDecisionEvaluation:
    def test_insufficient_samples_draw_no_finding(self):
        (ev,) = decision_evaluations([_decision(i, 0.1) for i in range(19)])
        assert ev.finding == "insufficient evidence: 19 forecasts, at least 20 needed"
        assert ev.mean_llm_value_added == pytest.approx(0.1)

    def test_consistent_improvement_is_reported(self):
        outcomes = [_decision(i, 0.05 + 0.01 * (i % 3)) for i in range(24)]
        (ev,) = decision_evaluations(outcomes)
        assert ev.ci_low > 0
        assert ev.share_improved == 1.0
        assert ev.finding.startswith("forecasts under this decision improved")

    def test_noisy_effect_is_not_a_finding(self):
        outcomes = [_decision(i, 0.1 if i % 2 else -0.1) for i in range(24)]
        (ev,) = decision_evaluations(outcomes)
        assert ev.ci_low < 0 < ev.ci_high
        assert ev.finding == "no measurable effect on forecast quality"

    def test_consistent_harm_is_reported(self):
        (ev,) = decision_evaluations([_decision(i, -0.05 - 0.01 * (i % 2)) for i in range(20)])
        assert ev.finding.startswith("forecasts under this decision were worse")

    def test_interval_uses_student_t(self):
        values = [0.01 * (i % 5) for i in range(20)]
        (ev,) = decision_evaluations([_decision(i, v) for i, v in enumerate(values)])
        half = 2.093 * statistics.stdev(values) / math.sqrt(20)  # t(0.975, 19 df)
        assert ev.ci_high - ev.mean_llm_value_added == pytest.approx(half, rel=1e-3)

    def test_no_adjustment_is_not_reported_as_no_effect(self):
        outcomes = [_decision(i, 0.0, decision="no_adjustment", adjusted=False) for i in range(25)]
        (ev,) = decision_evaluations(outcomes)
        assert ev.n_adjusted == 0
        assert ev.finding.startswith("no forecast was adjusted under this decision")

    def test_decisions_are_grouped_by_value(self):
        outcomes = [_decision(0, 0.1), _decision(1, 0.0, decision="no_adjustment", correct=False)]
        evs = {e.decision: e for e in decision_evaluations(outcomes)}
        assert set(evs) == {"allow_adjustment", "no_adjustment"}
        assert evs["no_adjustment"].direction.hits == 0


@pytest.fixture
def learning_store(migrated_db):
    return LearningStore(migrated_db)


@pytest.fixture
def history(adjusted_state, store, outcome_store, learning_store):
    """Three scored weeks (one in a sideways regime, one with a bearish technical
    analyst and the sector known) plus one invalid week."""
    sideways = adjusted_state.decision.model_copy(update={"result": MarketRegime.SIDEWAYS.value})
    bearish = {**adjusted_state.technical_report, "stance": "bearish"}
    variants = [
        {},
        {"decision": sideways},
        {"technical_report": bearish, "fundamentals_summary": {"sector": "Energy"}},
    ]
    snaps = []
    for n, update in enumerate(variants):
        snap = snapshot_as_of(adjusted_state, week(n), **update)
        outcome = save_scored(store, outcome_store, snap, 3.0, amplitude=0.6)
        learning_store.record_decision_outcomes(
            decision_outcomes_for(snap, outcome, cause=None, now=evaluated_at(snap))
        )
        snaps.append(snap)
    invalid = snapshot_as_of(adjusted_state, week(3))
    store.save(invalid)
    jump = score_forecast(  # an unexplained 45% one-day drop: evaluated as invalid
        invalid,
        path_series(invalid, [0, -45, -45, -45, -45]),
        None,
        now=evaluated_at(invalid),
        calendar=WEEKDAYS,
    )
    assert jump.status == OutcomeStatus.INVALID
    outcome_store.save(jump)
    return snaps


class TestBuildScorecards:
    def test_from_stored_history(self, history, outcome_store, learning_store):
        cards = build_scorecards(
            outcome_store, learning_store, as_of=evaluated_at(history[-1]) + timedelta(days=30)
        )
        assert cards.n_forecasts == 3  # the invalid week is not scored
        regimes = {g.group: g.n for g in cards.by_regime}
        assert regimes == {"sideways": 1, "trending_up": 2}
        technical = next(a for a in cards.analysts if a.analyst == "technical")
        assert (technical.pooled.hits, technical.pooled.n) == (2, 3)  # bearish call missed
        assert set(technical.by_sector) == {"Energy", "unknown"}
        gate = [d for d in cards.decisions if d.decision_type == "forecast_adjustment_gate"]
        assert [(d.decision, d.n) for d in gate] == [("allow_adjustment", 3)]

    def test_point_in_time(self, history, outcome_store, learning_store):
        cutoff = evaluated_at(history[1])
        cards = build_scorecards(outcome_store, learning_store, as_of=cutoff)
        assert cards.n_forecasts == 2
        per_type: dict[str, int] = {}
        for d in cards.decisions:
            per_type[d.decision_type] = per_type.get(d.decision_type, 0) + d.n
        assert set(per_type.values()) == {2}  # only the two weeks evaluated by the cutoff
        early = build_scorecards(outcome_store, learning_store, as_of=cutoff - timedelta(days=30))
        assert early.n_forecasts == 0 and early.decisions == []

    def test_overlapping_forecasts_count_once(
        self, history, adjusted_state, store, outcome_store, learning_store
    ):
        overlap = snapshot_as_of(adjusted_state, week(0) + timedelta(days=2))
        outcome = save_scored(store, outcome_store, overlap, 3.0, amplitude=0.6)
        learning_store.record_decision_outcomes(
            decision_outcomes_for(overlap, outcome, cause=None, now=evaluated_at(overlap))
        )
        cards = build_scorecards(
            outcome_store, learning_store, as_of=evaluated_at(history[-1]) + timedelta(days=30)
        )
        assert (cards.n_forecasts, cards.overlapping_excluded) == (3, 1)
        gate = [d for d in cards.decisions if d.decision_type == "forecast_adjustment_gate"]
        assert sum(d.n for d in gate) == 3  # its decisions are left out too
        assert "1 more left out" in render_scorecards(cards)

    def test_ticker_filter(self, history, outcome_store, learning_store):
        later = evaluated_at(history[-1])
        assert (
            build_scorecards(outcome_store, learning_store, as_of=later, ticker="TCS").n_forecasts
            == 0
        )

    def test_rendered(self, history, outcome_store, learning_store):
        text = render_scorecards(
            build_scorecards(outcome_store, learning_store, as_of=evaluated_at(history[-1]))
        )
        assert "- **technical**: 2/3 (67%, 95% CI" in text
        assert "| sideways | 1 |" in text
        assert "forecast_adjustment_gate: `allow_adjustment` | 3 |" in text
        assert "insufficient evidence: 3 forecasts, at least 20 needed" in text
        assert text.count("Weighting: automatic analyst weighting is disabled") == 1


def test_cli_scorecard(history, migrated_db):
    from stock_analysis.cli import app

    path = str(migrated_db.path)
    result = CliRunner().invoke(app, ["scorecard", "--database", path])
    assert result.exit_code == 0, result.output
    assert "Scored forecasts: 3" in result.output
    assert "### Decision-engine decisions" in result.output
    missing = CliRunner().invoke(app, ["scorecard", "--database", path + ".nope"])
    assert missing.exit_code == 1 and "No database at" in missing.output


def test_cli_scorecard_accepts_exchange_suffix(history, migrated_db):
    from stock_analysis.cli import app

    args = ["scorecard", "--database", str(migrated_db.path), "--ticker", "reliance.ns"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Scorecards for RELIANCE" in result.output
    assert "Scored forecasts: 3" in result.output


def test_cli_scorecard_unmigrated_database(tmp_path):
    import sqlite3

    from stock_analysis.cli import app

    path = tmp_path / "old.db"
    sqlite3.connect(path).close()
    result = CliRunner().invoke(app, ["scorecard", "--database", str(path)])
    assert result.exit_code == 1
    assert "stock-analysis migrate" in result.output


def test_cli_scorecard_empty(migrated_db):
    from stock_analysis.cli import app

    result = CliRunner().invoke(app, ["scorecard", "--database", str(migrated_db.path)])
    assert result.exit_code == 0
    assert "No scored forecasts yet." in result.output
