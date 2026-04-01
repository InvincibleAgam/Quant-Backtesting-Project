"""Portfolio accounting tests."""

from __future__ import annotations

import pandas as pd

from quantbt.portfolio import PositionState
from quantbt.types import Fill


def test_position_updates_and_realized_pnl() -> None:
    position = PositionState(asset="TEST")
    buy_fill = Fill(
        timestamp=pd.Timestamp("2024-01-01"),
        asset="TEST",
        quantity=10,
        fill_price=100.0,
        commission=0.0,
        slippage_cost=0.0,
        spread_cost=0.0,
        notional=1_000.0,
        reason="entry",
    )
    sell_fill_1 = Fill(
        timestamp=pd.Timestamp("2024-01-02"),
        asset="TEST",
        quantity=-5,
        fill_price=110.0,
        commission=0.0,
        slippage_cost=0.0,
        spread_cost=0.0,
        notional=-550.0,
        reason="trim",
    )
    sell_fill_2 = Fill(
        timestamp=pd.Timestamp("2024-01-03"),
        asset="TEST",
        quantity=-5,
        fill_price=120.0,
        commission=0.0,
        slippage_cost=0.0,
        spread_cost=0.0,
        notional=-600.0,
        reason="exit",
    )

    position.apply_fill(buy_fill, bar_index=0)
    trades_1 = position.apply_fill(sell_fill_1, bar_index=1)
    trades_2 = position.apply_fill(sell_fill_2, bar_index=2)

    assert position.quantity == 0
    assert position.realized_pnl == 150.0
    assert len(trades_1) == 1
    assert len(trades_2) == 1
    assert trades_1[0].pnl == 50.0
    assert trades_2[0].pnl == 100.0
