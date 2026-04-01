"""Strategy signal generation tests."""

from __future__ import annotations

from quantbt.strategies import MovingAverageCrossoverStrategy


def test_moving_average_signals_are_deterministic(single_asset_market_data) -> None:
    strategy = MovingAverageCrossoverStrategy(fast_window=2, slow_window=3, long_only=False)
    signals = strategy.generate_signals(single_asset_market_data)
    asset_signals = signals.xs("TEST", level="asset")["signal"]

    assert asset_signals.iloc[0] == 0.0
    assert asset_signals.iloc[1] == 0.0
    assert asset_signals.iloc[2] == 1.0
    assert asset_signals.iloc[4] == 1.0
    assert asset_signals.iloc[-1] == -1.0


def test_signal_generation_has_no_future_leakage(single_asset_market_data) -> None:
    strategy = MovingAverageCrossoverStrategy(fast_window=2, slow_window=3, long_only=False)
    full_signals = strategy.generate_signals(single_asset_market_data).xs("TEST", level="asset")["signal"]
    truncated_data = single_asset_market_data.between(
        single_asset_market_data.timestamps.min(),
        single_asset_market_data.timestamps[-2],
    )
    truncated_signals = strategy.generate_signals(truncated_data).xs("TEST", level="asset")["signal"]

    assert full_signals.iloc[:-1].tolist() == truncated_signals.tolist()
