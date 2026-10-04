"""Point-in-time analyst inputs built from a ``PointInTimeView``.

Only the technical and market-context analysts can be fed historically: both
are computed here from prices up to the as-of date. Fundamentals and news are
not reliably point-in-time (yfinance fundamentals are today's values; RSS news
cannot be reconstructed), so the backtest disables those analysts instead of
silently giving them future information (Plan.md §33-34).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from stock_analysis.backtest.data import INDIA_VIX_SYMBOL, QUANT_HISTORY_DAYS, PointInTimeView
from stock_analysis.indicators.technical import latest_indicator_summary
from stock_analysis.review.prices import NIFTY_50_SYMBOL

DISABLED_IN_BACKTEST = {
    "fundamental": (
        "fundamentals from the data source are today's values, not the as-of date's "
        "(not point-in-time)"
    ),
    "sentiment": "historical news cannot be reconstructed point-in-time (Plan.md §34)",
}
_BETA_WINDOW = 60
_DECIMALS = 4


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, _DECIMALS)
    return value


def technical_summary(view: PointInTimeView, symbol: str) -> Optional[dict[str, Any]]:
    """The latest indicator values as of the view's date (None if too little history)."""
    return latest_indicator_summary(view.frame(symbol, days=QUANT_HISTORY_DAYS), _DECIMALS)


def _pct(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


def _closes(view: PointInTimeView, symbol: str) -> list[float]:
    return [b.close for b in view.bars(symbol, days=QUANT_HISTORY_DAYS)]


def market_context_summary(view: PointInTimeView, symbol: str) -> dict[str, Any]:
    """Nifty 50 / India VIX levels and the stock's beta, from prices up to the as-of date."""
    summary: dict[str, Any] = {"as_of": view.as_of_date.isoformat(), "source": "historical"}
    nifty = _closes(view, NIFTY_50_SYMBOL)
    if len(nifty) >= 21:
        summary.update(
            nifty_50=nifty[-1],
            nifty_50_change_pct=_pct(nifty[-1], nifty[-2]),
            nifty_50_return_5d_pct=_pct(nifty[-1], nifty[-6]),
            nifty_50_return_20d_pct=_pct(nifty[-1], nifty[-21]),
        )
    vix = _closes(view, INDIA_VIX_SYMBOL)
    if len(vix) >= 2:
        summary.update(india_vix=vix[-1], india_vix_change=vix[-1] - vix[-2])

    stock_bars = {b.date: b.close for b in view.bars(symbol, days=QUANT_HISTORY_DAYS)}
    nifty_bars = {b.date: b.close for b in view.bars(NIFTY_50_SYMBOL, days=QUANT_HISTORY_DAYS)}
    common = sorted(set(stock_bars) & set(nifty_bars))[-(_BETA_WINDOW + 1) :]
    if len(common) > 20:
        s = np.diff(np.log([stock_bars[d] for d in common]))
        m = np.diff(np.log([nifty_bars[d] for d in common]))
        if np.var(m) > 0:
            summary["beta"] = float(np.cov(s, m)[0, 1] / np.var(m, ddof=1))
            summary["correlation_nifty"] = float(np.corrcoef(s, m)[0, 1])
        summary["relative_strength_vs_nifty_20d"] = _pct(
            stock_bars[common[-1]], stock_bars[common[-21]]
        ) - _pct(nifty_bars[common[-1]], nifty_bars[common[-21]])
    return {k: _round(v) for k, v in summary.items()}
