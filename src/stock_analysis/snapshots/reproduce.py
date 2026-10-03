"""Re-run a stored snapshot's quant baseline from its persisted price history."""

from __future__ import annotations

import numpy as np

from stock_analysis.database.forecast_store import ForecastSnapshotStore
from stock_analysis.quant import DailyQuantiles, QuantBaseline, QuantForecaster
from stock_analysis.schemas.snapshot import ForecastSnapshot
from stock_analysis.versions import QUANT_N_PATHS, QUANT_SEED, get_model_versions


class ReproductionError(Exception):
    """The snapshot's baseline cannot be re-run with the stored data and current code."""


def reproduce_quant_baseline(
    snapshot: ForecastSnapshot, store: ForecastSnapshotStore, *, calibrated: bool = True
) -> tuple[QuantBaseline, list[DailyQuantiles]]:
    """Recompute the quant baseline and daily path for ``snapshot``.

    With ``calibrated=False`` the uncalibrated shadow baseline is recomputed instead.

    Raises:
        ReproductionError: the price history was not stored, or the quant model
            configuration has changed since the snapshot was made.
    """
    sha = (snapshot.data_inputs.get("price") or {}).get("history_sha256")
    if not sha:
        raise ReproductionError(f"Snapshot {snapshot.forecast_id} has no price history fingerprint")
    history = store.get_price_history(sha)
    if history is None:
        raise ReproductionError(f"Price history {sha} for {snapshot.forecast_id} was not stored")

    method = str(snapshot.quant_baseline.get("method", "unknown"))
    current = get_model_versions(method)["quant"]
    recorded = snapshot.model_versions.get("quant")
    if current != recorded:
        raise ReproductionError(
            f"Quant model changed: snapshot used {recorded!r}, code is {current!r}"
        )

    forecaster = QuantForecaster(
        seed=QUANT_SEED, n_paths=QUANT_N_PATHS, horizon=snapshot.horizon_trading_days
    )
    params = (
        snapshot.calibration.model_dump(exclude={"version"})
        if calibrated and snapshot.calibration is not None
        else None
    )
    return forecaster.forecast_with_daily_path(np.array(history.close_values()), params)
