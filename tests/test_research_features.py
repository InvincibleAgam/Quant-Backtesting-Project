"""Tests for walk-forward and bootstrap research utilities."""

from __future__ import annotations

from quantbt.analysis.robustness import bootstrap_return_paths
from quantbt.backtester import WalkForwardRunner
from quantbt.config import BacktestConfig, ExecutionConfig
from quantbt.data import generate_sample_ohlcv
from quantbt.strategies import MovingAverageCrossoverStrategy


def test_bootstrap_return_paths_produces_statistics() -> None:
    market_data = generate_sample_ohlcv(assets=("TEST",), periods=40, seed=11)
    returns = market_data.asset_frame("TEST")["close"].pct_change().fillna(0.0)

    samples, stats, paths = bootstrap_return_paths(
        returns=returns,
        iterations=25,
        block_size=3,
        initial_cash=100_000.0,
        seed=11,
    )

    assert len(samples) == 25
    assert not stats.empty
    assert stats.iloc[0]["iterations"] == 25.0
    assert paths.shape[0] == len(returns)
    assert paths.shape[1] == 25


def test_walk_forward_runner_produces_segments_and_curve() -> None:
    market_data = generate_sample_ohlcv(assets=("TEST",), periods=80, seed=5)
    runner = WalkForwardRunner(
        backtest_config=BacktestConfig(initial_cash=100_000.0),
        execution_config=ExecutionConfig(
            commission_bps=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
            volume_share_slippage_bps=0.0,
            volume_limit=1.0,
            allow_short=False,
        ),
    )

    segments, curve, metrics = runner.run(
        market_data=market_data,
        strategy_class=MovingAverageCrossoverStrategy,
        param_grid=[
            {"fast_window": 5, "slow_window": 20},
            {"fast_window": 10, "slow_window": 30},
        ],
        train_bars=30,
        test_bars=10,
        step_bars=10,
        long_only=True,
    )

    assert not segments.empty
    assert "param_fast_window" in segments.columns
    assert not curve.empty
    assert not metrics.empty
    assert "annualized_return" in metrics.columns
