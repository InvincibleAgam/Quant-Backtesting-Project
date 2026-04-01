"""Execution simulator tests."""

from __future__ import annotations

import pandas as pd

from quantbt.config import ExecutionConfig
from quantbt.execution import ExecutionSimulator
from quantbt.types import Order


def test_execution_costs_and_slippage_are_applied() -> None:
    config = ExecutionConfig(
        price_source="next_open",
        commission_bps=5.0,
        slippage_bps=10.0,
        spread_bps=20.0,
        volume_share_slippage_bps=0.0,
        volume_limit=1.0,
    )
    simulator = ExecutionSimulator(config)
    order = Order(timestamp=pd.Timestamp("2024-01-01"), asset="TEST", quantity=100, reason="unit_test", signal=1.0)
    bar = pd.Series({"open": 100.0, "close": 101.0, "volume": 1_000.0})
    outcome = simulator.execute_order(order, bar, pd.Timestamp("2024-01-02"))

    assert outcome.fill is not None
    assert outcome.unfilled_quantity == 0
    assert outcome.fill.fill_price == 100.2
    assert round(outcome.fill.commission, 4) == 5.01
    assert outcome.fill.slippage_cost == 10.0
    assert outcome.fill.spread_cost == 10.0
