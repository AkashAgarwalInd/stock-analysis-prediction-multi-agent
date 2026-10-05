"""Plan.md Phases 11-12: postmortems, lessons and bounded calibration."""

from stock_analysis.learning.calibration import (
    compare_benchmarks,
    estimate_calibration,
    observation_from,
    update_calibration,
)
from stock_analysis.learning.context import add_learning_context
from stock_analysis.learning.cycle import (
    HindsightNewsSource,
    LearningCycle,
    LearningReport,
    decision_outcomes_for,
)
from stock_analysis.learning.evaluation import (
    build_evaluation,
    build_track_record,
    evaluate_forecast,
    load_evaluated_forecasts,
    probability_calibration,
    reliability_table,
)
from stock_analysis.learning.lessons import LessonAction, apply_lessons, retire_stale_lessons
from stock_analysis.learning.news import RssHindsightNewsSource
from stock_analysis.learning.postmortem import (
    build_postmortem_facts,
    build_postmortem_prompt,
    classify_deterministically,
    is_expected_noise,
    run_postmortem,
)
from stock_analysis.learning.report import (
    render_adaptation,
    render_calibration_history,
    render_lesson_actions,
    render_postmortem,
    render_probability_calibration,
    render_scorecards,
    render_track_record,
    render_track_records,
)
from stock_analysis.learning.scorecards import (
    analyst_scorecards,
    build_scorecards,
    decision_evaluations,
    group_scores,
    hit_rate,
)

__all__ = [
    "HindsightNewsSource",
    "LearningCycle",
    "LearningReport",
    "LessonAction",
    "RssHindsightNewsSource",
    "add_learning_context",
    "analyst_scorecards",
    "apply_lessons",
    "build_evaluation",
    "build_scorecards",
    "build_track_record",
    "build_postmortem_facts",
    "build_postmortem_prompt",
    "classify_deterministically",
    "compare_benchmarks",
    "decision_evaluations",
    "decision_outcomes_for",
    "estimate_calibration",
    "evaluate_forecast",
    "group_scores",
    "hit_rate",
    "is_expected_noise",
    "load_evaluated_forecasts",
    "observation_from",
    "probability_calibration",
    "reliability_table",
    "render_adaptation",
    "render_calibration_history",
    "render_lesson_actions",
    "render_postmortem",
    "render_probability_calibration",
    "render_scorecards",
    "render_track_record",
    "render_track_records",
    "retire_stale_lessons",
    "run_postmortem",
    "update_calibration",
]
