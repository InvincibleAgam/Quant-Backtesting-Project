"""Backtest engine integration tests."""

from __future__ import annotations

import pandas as pd

from quantbt.backtester import BacktestEngine
from quantbt.config import BacktestConfig, ExecutionConfig
from quantbt.strategies import MomentumBreakoutStrategy


def test_orders_fill_on_next_bar_not_signal_bar(single_asset_market_data) -> None:
    strategy = MomentumBreakoutStrategy(lookback=2, exit_lookback=1, long_only=True)
    execution_config = ExecutionConfig(
        price_source="next_open",
        commission_bps=0.0,
        slippage_bps=0.0,
        spread_bps=0.0,
        volume_share_slippage_bps=0.0,
        volume_limit=1.0,
        allow_short=False,
    )
    engine = BacktestEngine(BacktestConfig(initial_cash=100_000.0), execution_config)
    result = engine.run(single_asset_market_data, strategy)

    signal_frame = result.signals.xs("TEST", level="asset")
    first_signal_time = signal_frame[signal_frame["signal"] > 0.0].index[0]
    first_fill_time = pd.to_datetime(result.fills.iloc[0]["timestamp"])

    assert first_fill_time > first_signal_time
    assert first_fill_time == single_asset_market_data.timestamps[single_asset_market_data.timestamps.get_loc(first_signal_time) + 1]
