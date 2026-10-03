"""Plan.md Phase 11: postmortem, cause classification and the hindsight guard."""

from datetime import timedelta

import pytest

from stock_analysis.learning import (
    build_postmortem_facts,
    build_postmortem_prompt,
    is_expected_noise,
    run_postmortem,
)
from stock_analysis.learning.postmortem import PostmortemError
from stock_analysis.schemas.learning import (
    CauseCategory,
    FactTiming,
    HindsightNewsItem,
    PostmortemMethod,
)
from tests.learning_helpers import (
    FakePostmortemLLM,
    evaluated_at,
    fact_id,
    made_at,
    score,
    snapshot_as_of,
    week,
)

NOISE = {"final_pct": 0.5, "amplitude": 0.6}  # inside the 80% band, normal volatility
VOLATILE_MISS = {"final_pct": 6.0, "amplitude": 2.5}  # outside the band, vol ratio ~3.75


@pytest.fixture
def snap(adjusted_state):
    return snapshot_as_of(adjusted_state, week(0))


def _lesson(
    refs,
    text="Realised weekly volatility for this ticker can exceed the EWMA estimate "
    "when all analysts agree on a bullish trend",
    category="volatility_underestimated",
    scope="ticker",
):
    return {"text": text, "scope": scope, "category": category, "evidence_refs": refs}


def _diagnosis(cause="volatility_underestimated", lessons=(), knowable=(), hindsight=(), **extra):
    return {
        "primary_cause": cause,
        "explanation": "Daily swings were far larger than the forecast's volatility.",
        "knowable_at_forecast_time": list(knowable),
        "only_in_hindsight": list(hindsight),
        "lessons": list(lessons),
        "confidence": "medium",
        **extra,
    }


def _run(snap, llm, **scenario):
    outcome = score(snap, **scenario)
    return outcome, run_postmortem(snap, outcome, llm_factory=llm, now=evaluated_at(snap))


class TestNoise:
    def test_normal_noise_creates_no_lesson_and_skips_the_llm(self, snap):
        llm = FakePostmortemLLM(_diagnosis(lessons=[_lesson(["F3"])]))
        outcome, pm = _run(snap, llm, **NOISE)

        assert is_expected_noise(outcome)
        assert pm.primary_cause == CauseCategory.WITHIN_EXPECTED_NOISE
        assert pm.method == PostmortemMethod.DETERMINISTIC
        assert pm.lessons == []
        assert llm.prompts == []  # no LLM call for ordinary noise

    def test_llm_noise_verdict_never_carries_a_lesson(self, snap):
        # inside the band but volatile: the LLM is consulted and calls it noise
        llm = FakePostmortemLLM(
            _diagnosis(cause="within_expected_noise", lessons=[_lesson(["F3"])])
        )
        outcome, pm = _run(snap, llm, final_pct=1.0, amplitude=3.0)
        assert outcome.in_80pct_band and not is_expected_noise(outcome)
        assert pm.primary_cause == CauseCategory.WITHIN_EXPECTED_NOISE
        assert pm.lessons == []
        assert any("within expected noise" in n for n in pm.guard_notes)

    def test_noise_cause_rejected_outside_the_band(self, snap):
        llm = FakePostmortemLLM(_diagnosis(cause="within_expected_noise"))
        _, pm = _run(snap, llm, **VOLATILE_MISS)
        assert pm.method == PostmortemMethod.DETERMINISTIC_FALLBACK
        assert pm.primary_cause == CauseCategory.VOLATILITY_UNDERESTIMATED
        assert "outside the 80% band" in pm.guard_notes[0]


class TestDiagnosis:
    def test_candidate_lesson_from_forecast_time_evidence(self, snap):
        llm = FakePostmortemLLM(
            _diagnosis(
                lessons=[_lesson(["F3", "F11"])],
                knowable=[{"statement": "Quant volatility was 2.71% weekly", "fact_ids": ["F3"]}],
                hindsight=[
                    {"statement": "Realised volatility was much higher", "fact_ids": ["H4"]}
                ],
            )
        )
        _, pm = _run(snap, llm, **VOLATILE_MISS)

        assert pm.method == PostmortemMethod.LLM
        assert pm.primary_cause == CauseCategory.VOLATILITY_UNDERESTIMATED
        assert [(c.scope, c.category, c.ticker) for c in pm.lessons] == [
            ("ticker", "volatility_underestimated", "RELIANCE")
        ]
        assert pm.guard_notes == []
        assert pm.prompt_version and pm.llm_model

    def test_at_most_three_lessons(self, snap):
        lessons = [
            _lesson(["F3"], category=c)
            for c in ("volatility_underestimated", "trend_misread", "analyst_error", "data_issue")
        ]
        _, pm = _run(snap, FakePostmortemLLM(_diagnosis(lessons=lessons)), **VOLATILE_MISS)
        # the schema refuses a fourth lesson, so the whole answer falls back to rules
        assert pm.method == PostmortemMethod.DETERMINISTIC_FALLBACK
        assert pm.lessons == []
        assert "ValidationError" in pm.guard_notes[0]

    @pytest.mark.parametrize(
        ("cause", "why"),
        [
            ("market_wide_move", "Nifty 50 moved only"),
            ("earnings_or_corporate_event", "no corporate-action or news fact"),
            ("regulatory_or_news_shock", "no news fact"),
            ("volatility_overestimated", "not well below"),
            ("analyst_error", "no analyst made a wrong"),
            ("data_issue", "no data-quality fact"),
        ],
    )
    def test_hallucinated_cause_is_rejected(self, snap, cause, why):
        llm = FakePostmortemLLM(_diagnosis(cause=cause, lessons=[_lesson(["F3"])]))
        _, pm = _run(snap, llm, **VOLATILE_MISS)
        assert pm.method == PostmortemMethod.DETERMINISTIC_FALLBACK
        assert pm.primary_cause == CauseCategory.VOLATILITY_UNDERESTIMATED
        assert pm.lessons == []
        assert why in pm.guard_notes[0]

    def test_market_wide_move_accepted_when_nifty_moved_with_the_error(self, snap):
        llm = FakePostmortemLLM(_diagnosis(cause="market_wide_move"))
        _, pm = _run(snap, llm, final_pct=6.0, amplitude=0.6, nifty_pct=4.0)
        assert pm.method == PostmortemMethod.LLM
        assert pm.primary_cause == CauseCategory.MARKET_WIDE_MOVE

    def test_news_cause_needs_a_cited_news_fact(self, snap):
        news = [
            HindsightNewsItem(
                published_at=made_at(snap.as_of_date) + timedelta(days=2),
                headline="Regulator approves new tariff structure",
            )
        ]
        outcome = score(snap, **VOLATILE_MISS)
        facts = build_postmortem_facts(snap, outcome, news)
        news_id = fact_id(facts, "news")
        llm = FakePostmortemLLM(
            _diagnosis(
                cause="regulatory_or_news_shock",
                hindsight=[{"statement": "A tariff ruling moved the stock", "fact_ids": [news_id]}],
            )
        )
        pm = run_postmortem(snap, outcome, llm_factory=llm, now=evaluated_at(snap), news=news)
        assert pm.method == PostmortemMethod.LLM
        assert pm.primary_cause == CauseCategory.REGULATORY_OR_NEWS_SHOCK


class TestHindsightGuard:
    def test_facts_are_split_by_when_they_were_knowable(self, snap):
        outcome = score(snap, **VOLATILE_MISS)
        facts = build_postmortem_facts(snap, outcome)
        actual = f"{outcome.actual_close_on_forecast_basis:.2f}"

        assert all(f.id.startswith("F") for f in facts.forecast_time)
        assert all(f.timing == FactTiming.FORECAST_TIME for f in facts.forecast_time)
        assert all(f.id.startswith("H") for f in facts.hindsight)
        assert not any(actual in f.text for f in facts.forecast_time)
        assert any(actual in f.text for f in facts.hindsight)

    def test_prompt_keeps_outcome_out_of_the_forecast_time_section(self, snap):
        outcome = score(snap, **VOLATILE_MISS)
        prompt = build_postmortem_prompt(snap, outcome, build_postmortem_facts(snap, outcome))
        forecast_time, hindsight = prompt.split("FORECAST-TIME FACTS")[1].split("HINDSIGHT FACTS")
        actual = f"{outcome.actual_close_on_forecast_basis:.2f}"
        assert actual not in forecast_time and actual in hindsight
        assert "Do not change any computed metric" in prompt

    def test_news_is_hindsight_only_inside_the_window(self, snap):
        made = made_at(snap.as_of_date)
        news = [
            HindsightNewsItem(published_at=made - timedelta(hours=1), headline="Before forecast"),
            HindsightNewsItem(published_at=made + timedelta(days=1), headline="During window"),
            HindsightNewsItem(published_at=made + timedelta(days=30), headline="Much later"),
        ]
        facts = build_postmortem_facts(snap, score(snap, **VOLATILE_MISS), news)
        texts = [f.text for f in (*facts.forecast_time, *facts.hindsight)]
        assert any("During window" in t for t in texts)
        assert not any("Before forecast" in t or "Much later" in t for t in texts)
        assert all(f.timing == FactTiming.HINDSIGHT for f in facts.hindsight)

    def test_knowable_claim_citing_hindsight_is_moved(self, snap):
        llm = FakePostmortemLLM(
            _diagnosis(
                knowable=[
                    {"statement": "Volatility was obviously going to spike", "fact_ids": ["H4"]},
                    {"statement": "Analysts were unanimous", "fact_ids": ["F8", "F9"]},
                    {"statement": "An invented fact", "fact_ids": ["F99"]},
                ]
            )
        )
        _, pm = _run(snap, llm, **VOLATILE_MISS)
        assert [s.statement for s in pm.knowable_at_forecast_time] == ["Analysts were unanimous"]
        assert "Volatility was obviously going to spike" in [
            s.statement for s in pm.only_in_hindsight
        ]
        assert any("Moved to hindsight" in n for n in pm.guard_notes)
        assert any("unknown facts ['F99']" in n for n in pm.guard_notes)

    def test_lessons_cannot_rest_on_hindsight(self, snap):
        outcome = score(snap, **VOLATILE_MISS)
        actual = f"{outcome.actual_close_on_forecast_basis:.2f}"
        llm = FakePostmortemLLM(
            _diagnosis(
                lessons=[
                    _lesson(
                        ["F3", "H4"], text="Weekly volatility was underestimated in this window"
                    ),
                    _lesson(
                        ["F3"],
                        text=f"Expect closes near {actual} when analysts all turn bullish",
                        category="trend_misread",
                    ),
                    _lesson(
                        ["F42"],
                        text="Some lesson resting on a made-up fact entirely",
                        category="analyst_error",
                    ),
                ]
            )
        )
        pm = run_postmortem(snap, outcome, llm_factory=llm, now=evaluated_at(snap))
        assert pm.method == PostmortemMethod.LLM
        assert pm.lessons == []
        notes = " ".join(pm.guard_notes)
        assert "cites hindsight facts ['H4']" in notes
        assert f"quotes numbers known only in hindsight ['{actual}']" in notes
        assert "cites unknown facts ['F42']" in notes


class TestSafety:
    @pytest.mark.parametrize(
        "llm",
        [
            FakePostmortemLLM(error=RuntimeError("LLM unavailable")),
            FakePostmortemLLM(_diagnosis(actual_return_pct=0.0)),  # tries to restate a metric
            FakePostmortemLLM({"primary_cause": "bad luck"}),
        ],
    )
    def test_invalid_response_falls_back_safely(self, snap, llm):
        outcome, pm = _run(snap, llm, **VOLATILE_MISS)
        assert pm.method == PostmortemMethod.DETERMINISTIC_FALLBACK
        assert pm.primary_cause == CauseCategory.VOLATILITY_UNDERESTIMATED
        assert pm.confidence == "low"
        assert pm.lessons == []
        assert pm.guard_notes

    def test_scores_are_not_part_of_the_postmortem(self, snap):
        outcome, pm = _run(snap, FakePostmortemLLM(_diagnosis()), **VOLATILE_MISS)
        assert "actual_return_pct" not in pm.model_dump()
        # the scored outcome is a frozen model the postmortem only reads
        assert score(snap, **VOLATILE_MISS).model_dump(exclude={"evaluated_at"}) == (
            outcome.model_dump(exclude={"evaluated_at"})
        )

    def test_unscored_forecast_is_refused(self, snap):
        invalid = score(snap, 0.0, amplitude=0.0).model_copy(
            update={"status": "invalid", "invalid_reason": "test"}
        )
        with pytest.raises(PostmortemError):
            run_postmortem(snap, invalid, now=evaluated_at(snap))

    def test_without_llm_uses_rules(self, snap):
        _, pm = _run(snap, None, **VOLATILE_MISS)
        assert pm.method == PostmortemMethod.DETERMINISTIC_FALLBACK
        assert pm.guard_notes == ["No LLM configured"]


def test_knowable_claim_quoting_hindsight_numbers_is_moved(snap):
    outcome = score(snap, **VOLATILE_MISS)
    actual = f"{outcome.actual_close_on_forecast_basis:.2f}"
    llm = FakePostmortemLLM(
        _diagnosis(
            knowable=[
                {"statement": f"A close near {actual} was foreseeable", "fact_ids": ["F3"]},
                {"statement": "Quant weekly volatility was 2.71%", "fact_ids": ["F3"]},
            ]
        )
    )
    pm = run_postmortem(snap, outcome, llm_factory=llm, now=evaluated_at(snap))
    assert [s.statement for s in pm.knowable_at_forecast_time] == [
        "Quant weekly volatility was 2.71%"
    ]
    assert any("quotes numbers known only in hindsight" in n for n in pm.guard_notes)
