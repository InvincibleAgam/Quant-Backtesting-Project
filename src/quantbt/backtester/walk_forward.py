"""Walk-forward optimization utilities."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from quantbt.backtester.engine import BacktestEngine
from quantbt.backtester.sweep import ParameterSweepRunner
from quantbt.config import BacktestConfig, ExecutionConfig
from quantbt.metrics import compute_performance_metrics
from quantbt.strategies.base import BaseStrategy
from quantbt.types import MarketData
from quantbt.utils import annualization_factor


def _extract_best_parameters(sweep: pd.DataFrame) -> dict[str, object]:
    excluded = {"strategy", "sharpe_ratio", "annualized_return", "maximum_drawdown", "number_of_trades"}
    row = sweep.iloc[0].to_dict()
    return {key: value for key, value in row.items() if key not in excluded}


class WalkForwardRunner:
    """Run repeated train/test cycles across rolling or stepping windows."""

    def __init__(self, backtest_config: BacktestConfig, execution_config: ExecutionConfig) -> None:
        self.backtest_config = backtest_config
        self.execution_config = execution_config

    def run(
        self,
        market_data: MarketData,
        strategy_class: type[BaseStrategy],
        param_grid: list[dict[str, object]],
        train_bars: int,
        test_bars: int,
        step_bars: int | None = None,
        metric_name: str = "sharpe_ratio",
        long_only: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run walk-forward selection and testing over the full sample."""

        timestamps = market_data.timestamps.sort_values()
        if len(timestamps) < train_bars + test_bars:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        step = step_bars or test_bars
        sweep_runner = ParameterSweepRunner(self.backtest_config, self.execution_config)
        segment_rows: list[dict[str, object]] = []
        equity_segments: list[pd.DataFrame] = []

        start_idx = 0
        segment_id = 0
        while start_idx + train_bars + test_bars <= len(timestamps):
            train_start = timestamps[start_idx]
            train_end = timestamps[start_idx + train_bars - 1]
            test_start = timestamps[start_idx + train_bars]
            test_end = timestamps[start_idx + train_bars + test_bars - 1]

            train_data = market_data.between(train_start, train_end)
            test_data = market_data.between(test_start, test_end)

            sweep = sweep_runner.run(
                market_data=train_data,
                strategy_class=strategy_class,
                param_grid=param_grid,
                metric_name=metric_name,
                long_only=long_only,
            )
            best_parameters = _extract_best_parameters(sweep)
            strategy = strategy_class(long_only=long_only, **best_parameters)
            train_result = BacktestEngine(
                backtest_config=replace(self.backtest_config),
                execution_config=replace(self.execution_config),
            ).run(train_data, strategy)
            test_result = BacktestEngine(
                backtest_config=replace(self.backtest_config),
                execution_config=replace(self.execution_config),
            ).run(test_data, strategy)

            segment_row: dict[str, object] = {
                "segment_id": segment_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "selected_metric": sweep.iloc[0][metric_name] if not sweep.empty else 0.0,
                "train_sharpe_ratio": train_result.metrics["sharpe_ratio"],
                "test_sharpe_ratio": test_result.metrics["sharpe_ratio"],
                "train_annualized_return": train_result.metrics["annualized_return"],
                "test_annualized_return": test_result.metrics["annualized_return"],
                "test_maximum_drawdown": test_result.metrics["maximum_drawdown"],
                "test_number_of_trades": test_result.metrics["number_of_trades"],
            }
            segment_row.update({f"param_{key}": value for key, value in best_parameters.items()})
            segment_rows.append(segment_row)

            segment_equity = test_result.equity_curve[["equity", "returns", "drawdown"]].copy()
            segment_equity["segment_id"] = segment_id
            equity_segments.append(segment_equity)

            start_idx += step
            segment_id += 1

        segments = pd.DataFrame(segment_rows)
        if not equity_segments:
            return segments, pd.DataFrame(), pd.DataFrame()

        stitched = pd.concat(equity_segments).sort_index()
        compounded = self.backtest_config.initial_cash * (1.0 + stitched["returns"].fillna(0.0)).cumprod()
        stitched["stitched_equity"] = compounded
        stitched["stitched_drawdown"] = stitched["stitched_equity"] / stitched["stitched_equity"].cummax() - 1.0
        stitched_curve = pd.DataFrame(
            {
                "equity": stitched["stitched_equity"],
                "returns": stitched["returns"].fillna(0.0),
                "drawdown": stitched["stitched_drawdown"],
            }
        )
        metrics = compute_performance_metrics(
            stitched_curve,
            trades=pd.DataFrame(),
            annualization=annualization_factor(stitched_curve.index, self.backtest_config.annual_trading_days),
        )
        metrics_frame = pd.DataFrame([metrics])
        return segments, stitched_curve, metrics_frame
