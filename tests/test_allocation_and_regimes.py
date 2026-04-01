"""Tests for allocation constraints and regime segmentation."""

from __future__ import annotations

import pandas as pd

from quantbt.analysis.regimes import analyze_regime_performance, infer_market_regimes
from quantbt.config import ExecutionConfig
from quantbt.data import generate_sample_ohlcv
from quantbt.execution import PositionSizer


def test_position_sizer_respects_asset_cap_and_active_position_limit() -> None:
    config = ExecutionConfig(
        allow_short=False,
        allocation_scheme="inverse_volatility",
        max_gross_leverage=1.0,
        max_asset_weight=0.34,
        max_active_positions=2,
        min_cash_buffer=0.0,
        volatility_lookback=5,
    )
    sizer = PositionSizer(config)
    signals = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0})
    prices = pd.Series({"A": 100.0, "B": 100.0, "C": 100.0})
    price_history = pd.DataFrame(
        {
            "A": [100, 101, 102, 103, 104, 105],
            "B": [100, 100.5, 101, 101.5, 102, 102.5],
            "C": [100, 110, 90, 115, 85, 120],
        },
        index=pd.bdate_range("2024-01-01", periods=6),
    )

    targets = sizer.target_shares(signals, prices, equity=1_000.0, price_history=price_history)

    active_assets = [asset for asset, quantity in targets.items() if quantity > 0]
    assert len(active_assets) == 2
    assert max(targets.values()) <= 3


def test_regime_segmentation_returns_summary() -> None:
    market_data = generate_sample_ohlcv(assets=("A", "B"), periods=140, seed=9)
    regime_frame = infer_market_regimes(market_data, lookback=20)

    index = regime_frame.index
    strategy_returns = pd.Series(0.001, index=index)
    strategy_curve = pd.DataFrame({"returns": strategy_returns}, index=index)
    strategy_curve["equity"] = (1.0 + strategy_curve["returns"]).cumprod()
    strategy_curve["drawdown"] = strategy_curve["equity"] / strategy_curve["equity"].cummax() - 1.0

    benchmark_returns = pd.Series(0.0005, index=index)
    benchmark_curve = pd.DataFrame({"returns": benchmark_returns}, index=index)
    benchmark_curve["equity"] = (1.0 + benchmark_curve["returns"]).cumprod()
    benchmark_curve["drawdown"] = benchmark_curve["equity"] / benchmark_curve["equity"].cummax() - 1.0

    summary = analyze_regime_performance(strategy_curve, benchmark_curve, regime_frame)

    assert not summary.empty
    assert "regime" in summary.columns
    assert "strategy_sharpe_ratio" in summary.columns
