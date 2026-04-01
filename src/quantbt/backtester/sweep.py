"""Parameter sweep and model selection helpers."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from quantbt.backtester.engine import BacktestEngine
from quantbt.config import BacktestConfig, ExecutionConfig
from quantbt.strategies.base import BaseStrategy
from quantbt.types import MarketData


class ParameterSweepRunner:
    """Evaluate parameter grids and rank them by in-sample metrics."""

    def __init__(self, backtest_config: BacktestConfig, execution_config: ExecutionConfig) -> None:
        self.backtest_config = backtest_config
        self.execution_config = execution_config

    def run(
        self,
        market_data: MarketData,
        strategy_class: type[BaseStrategy],
        param_grid: list[dict[str, object]],
        metric_name: str = "sharpe_ratio",
        long_only: bool = False,
    ) -> pd.DataFrame:
        """Run the full grid on in-sample data."""

        results: list[dict[str, object]] = []
        for params in param_grid:
            strategy = strategy_class(long_only=long_only, **params)
            engine = BacktestEngine(
                backtest_config=replace(self.backtest_config),
                execution_config=replace(self.execution_config),
            )
            result = engine.run(market_data, strategy)
            row = {
                "strategy": strategy.name,
                metric_name: result.metrics.get(metric_name, 0.0),
                "annualized_return": result.metrics.get("annualized_return", 0.0),
                "maximum_drawdown": result.metrics.get("maximum_drawdown", 0.0),
                "number_of_trades": result.metrics.get("number_of_trades", 0.0),
            }
            row.update(params)
            results.append(row)
        sweep = pd.DataFrame(results).sort_values(metric_name, ascending=False).reset_index(drop=True)
        return sweep
