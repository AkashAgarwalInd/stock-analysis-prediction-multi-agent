"""Plan.md Phases 11-12: postmortems, lessons and bounded calibration."""

from stock_analysis.learning.calibration import (
    compare_benchmarks,
    estimate_calibration,
    observation_from,
    update_calibration,
)
from stock_analysis.learning.cycle import (
    HindsightNewsSource,
    LearningCycle,
    LearningReport,
    decision_outcomes_for,
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
    render_lesson_actions,
    render_postmortem,
)

__all__ = [
    "HindsightNewsSource",
    "LearningCycle",
    "LearningReport",
    "LessonAction",
    "RssHindsightNewsSource",
    "apply_lessons",
    "build_postmortem_facts",
    "build_postmortem_prompt",
    "classify_deterministically",
    "compare_benchmarks",
    "decision_outcomes_for",
    "estimate_calibration",
    "is_expected_noise",
    "observation_from",
    "render_adaptation",
    "render_lesson_actions",
    "render_postmortem",
    "retire_stale_lessons",
    "run_postmortem",
    "update_calibration",
]
