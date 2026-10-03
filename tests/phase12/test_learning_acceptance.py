"""Plan.md §71: bad forecasts -> diagnosis -> evidence -> calibration -> changed forecast,
while one bad week changes nothing."""

import pytest

from stock_analysis.database import LearningStore
from stock_analysis.learning import LearningCycle
from tests.forecast_helpers import FakeLLM, run_graph
from tests.learning_helpers import evaluated_at, save_scored, snapshot_as_of, week


@pytest.fixture
def learning_store(migrated_db):
    return LearningStore(migrated_db)


def _bad_weeks(adjusted_state, store, outcome_store, weeks):
    snaps = []
    for n in weeks:
        snap = snapshot_as_of(adjusted_state, week(n))
        save_scored(store, outcome_store, snap, 3.0 if n % 2 else -3.0, amplitude=0.6)
        snaps.append(snap)
    return snaps


def test_bad_weeks_change_future_forecasts_but_one_does_not(
    adjusted_state, initial_state, price_history, store, outcome_store, memory_store, learning_store
):
    cycle = LearningCycle(store, outcome_store, memory_store, learning_store)

    first = _bad_weeks(adjusted_state, store, outcome_store, [0])
    report = cycle.run(evaluated_at(first[-1]))
    assert len(report.postmortems) == 1 and not report.calibration_updates[0].changed
    unchanged = run_graph(FakeLLM(), initial_state, store, memory_store, learning_store)
    assert unchanged.calibration is None
    assert unchanged.quant_baseline == adjusted_state.quant_baseline

    later = _bad_weeks(adjusted_state, store, outcome_store, range(1, 8))
    report = cycle.run(evaluated_at(later[-1]))
    assert len(report.postmortems) == 7
    (update,) = report.calibration_updates
    assert update.changed and update.stored.n_samples == 8

    adapted = run_graph(FakeLLM(), initial_state, store, memory_store, learning_store)
    assert adapted.calibration["version"] == 1
    assert adapted.quant_baseline_uncalibrated == adjusted_state.quant_baseline
    assert adapted.quant_baseline["p90_price"] > adjusted_state.quant_baseline["p90_price"]
    assert adapted.quant_baseline["p10_price"] < adjusted_state.quant_baseline["p10_price"]
    assert store.get(adapted.forecast_id).calibration_version == 1
