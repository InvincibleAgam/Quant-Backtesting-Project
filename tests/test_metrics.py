"""Performance metric tests."""

from __future__ import annotations

import pandas as pd

from quantbt.metrics import compute_performance_metrics


def test_metric_calculations() -> None:
    index = pd.bdate_range("2024-01-01", periods=4, name="timestamp")
    equity_curve = pd.DataFrame(
        {
            "equity": [100.0, 110.0, 105.0, 115.0],
            "returns": [0.0, 0.1, -0.0454545455, 0.0952380952],
            "turnover": [0.0, 0.1, 0.05, 0.05],
            "gross_exposure": [0.0, 1.0, 1.0, 1.0],
            "drawdown": [0.0, 0.0, -0.0454545455, 0.0],
        },
        index=index,
    )
    trades = pd.DataFrame(
        {
            "pnl": [10.0, -5.0, 20.0],
            "return_pct": [0.1, -0.05, 0.2],
        }
    )

    metrics = compute_performance_metrics(equity_curve, trades, annualization=252)

    assert round(metrics["cumulative_return"], 4) == 0.15
    assert round(metrics["maximum_drawdown"], 4) == -0.0455
    assert round(metrics["win_rate"], 4) == 0.6667
    assert round(metrics["profit_factor"], 4) == 6.0
    assert metrics["number_of_trades"] == 3.0
