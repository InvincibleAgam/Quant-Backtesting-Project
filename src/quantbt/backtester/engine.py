"""Event-style backtesting engine."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from quantbt.config import BacktestConfig, ExecutionConfig
from quantbt.execution import ExecutionSimulator, PositionSizer
from quantbt.metrics import compute_performance_metrics
from quantbt.portfolio import PortfolioState
from quantbt.strategies.base import BaseStrategy
from quantbt.types import BacktestResult, MarketData, Order
from quantbt.utils import annualization_factor, build_equal_weight_benchmark


class BacktestEngine:
    """Consume signals, schedule orders, fill them, and track portfolio state."""

    def __init__(self, backtest_config: BacktestConfig, execution_config: ExecutionConfig) -> None:
        self.backtest_config = backtest_config
        self.execution_config = execution_config
        self.execution_simulator = ExecutionSimulator(execution_config)
        self.position_sizer = PositionSizer(execution_config)

    def run(self, market_data: MarketData, strategy: BaseStrategy) -> BacktestResult:
        """Run a full backtest for a strategy over the supplied market data."""

        signals = strategy.generate_signals(market_data)
        portfolio = PortfolioState(initial_cash=self.backtest_config.initial_cash)
        pending_orders: dict[pd.Timestamp, list[Order]] = defaultdict(list)
        timestamps = list(market_data.timestamps)
        timestamp_to_next = {
            timestamps[idx]: timestamps[idx + 1] for idx in range(len(timestamps) - 1)
        }

        for bar_index, timestamp in enumerate(timestamps):
            bars = market_data.bars.xs(timestamp, level="timestamp").copy()
            turnover_notional = 0.0

            for order in pending_orders.pop(timestamp, []):
                if order.asset not in bars.index:
                    next_timestamp = timestamp_to_next.get(timestamp)
                    if next_timestamp is not None:
                        pending_orders[next_timestamp].append(order)
                    continue
                outcome = self.execution_simulator.execute_order(order, bars.loc[order.asset], timestamp)
                if outcome.fill is not None:
                    turnover_notional += abs(outcome.fill.notional)
                    portfolio.apply_fill(outcome.fill, bar_index)
                if outcome.unfilled_quantity != 0:
                    next_timestamp = timestamp_to_next.get(timestamp)
                    if next_timestamp is not None:
                        pending_orders[next_timestamp].append(
                            Order(
                                timestamp=timestamp,
                                asset=order.asset,
                                quantity=outcome.unfilled_quantity,
                                reason=f"{order.reason}_residual",
                                signal=order.signal,
                            )
                        )

            portfolio.snapshot(timestamp, bars, turnover_notional)
            next_timestamp = timestamp_to_next.get(timestamp)
            if next_timestamp is None:
                continue

            signal_slice = signals.xs(timestamp, level="timestamp")["signal"]
            prices = bars["close"]
            price_history = market_data.bars.loc[(slice(None, timestamp), slice(None)), "close"].unstack("asset")
            equity = portfolio.equity_curve().iloc[-1]["equity"]
            target_positions = self.position_sizer.target_shares(
                signal_slice,
                prices,
                float(equity),
                price_history=price_history,
            )
            current_positions = portfolio.position_quantities()

            for asset in market_data.assets:
                current_quantity = current_positions.get(asset, 0)
                target_quantity = target_positions.get(asset, 0)
                delta = target_quantity - current_quantity
                if delta == 0:
                    continue
                pending_orders[next_timestamp].append(
                    Order(
                        timestamp=timestamp,
                        asset=asset,
                        quantity=delta,
                        reason="signal_rebalance",
                        signal=float(signal_slice.get(asset, 0.0)),
                    )
                )

        equity_curve = portfolio.equity_curve()
        trades = portfolio.trades_frame()
        fills = portfolio.fills_frame()
        annualization = annualization_factor(equity_curve.index, self.backtest_config.annual_trading_days)
        metrics = compute_performance_metrics(equity_curve, trades, annualization)
        benchmark_curve = build_equal_weight_benchmark(market_data.bars, self.backtest_config.initial_cash)
        benchmark_metrics = compute_performance_metrics(benchmark_curve, pd.DataFrame(), annualization)
        return BacktestResult(
            strategy_name=strategy.name,
            parameters=strategy.describe_parameters(),
            equity_curve=equity_curve,
            trades=trades,
            fills=fills,
            signals=signals,
            metrics=metrics,
            benchmark_curve=benchmark_curve,
            benchmark_metrics=benchmark_metrics,
        )
