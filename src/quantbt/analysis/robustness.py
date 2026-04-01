"""Robustness checks and summary helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from quantbt.backtester import BacktestEngine
from quantbt.config import BacktestConfig, ExecutionConfig
from quantbt.metrics import compute_performance_metrics
from quantbt.strategies.base import BaseStrategy
from quantbt.types import BacktestResult, MarketData
from quantbt.utils import annualization_factor


def _format_frame_for_text(frame: pd.DataFrame) -> str:
    """Round numeric columns while leaving datetimes untouched for text reports."""

    formatted = frame.copy()
    numeric_columns = formatted.select_dtypes(include=["number"]).columns
    if len(numeric_columns) > 0:
        formatted.loc[:, numeric_columns] = formatted.loc[:, numeric_columns].round(4)
    return formatted.to_string(index=False)


def slippage_sensitivity(
    market_data: MarketData,
    strategy: BaseStrategy,
    backtest_config: BacktestConfig,
    execution_config: ExecutionConfig,
    scenarios_bps: tuple[float, ...],
) -> pd.DataFrame:
    """Evaluate performance under alternative slippage assumptions."""

    rows: list[dict[str, float]] = []
    for slippage in scenarios_bps:
        scenario_config = replace(execution_config, slippage_bps=slippage)
        result = BacktestEngine(backtest_config, scenario_config).run(market_data, strategy)
        rows.append(
            {
                "slippage_bps": slippage,
                "annualized_return": result.metrics["annualized_return"],
                "sharpe_ratio": result.metrics["sharpe_ratio"],
                "maximum_drawdown": result.metrics["maximum_drawdown"],
                "number_of_trades": result.metrics["number_of_trades"],
            }
        )
    return pd.DataFrame(rows)


def bootstrap_return_paths(
    returns: pd.Series,
    iterations: int = 500,
    block_size: int = 5,
    initial_cash: float = 1_000_000.0,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a moving-block bootstrap over portfolio returns."""

    clean_returns = returns.fillna(0.0)
    if clean_returns.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(seed)
    n_obs = len(clean_returns)
    block = max(1, min(block_size, n_obs))
    annualization = annualization_factor(clean_returns.index)
    rows: list[dict[str, float]] = []
    paths: list[pd.Series] = []

    values = clean_returns.to_numpy()
    index = clean_returns.index
    for iteration in range(iterations):
        sampled: list[float] = []
        while len(sampled) < n_obs:
            start = int(rng.integers(0, max(n_obs - block + 1, 1)))
            sampled.extend(values[start : start + block].tolist())
        sampled_returns = pd.Series(sampled[:n_obs], index=index)
        equity = initial_cash * (1.0 + sampled_returns).cumprod()
        curve = pd.DataFrame({"equity": equity, "returns": sampled_returns})
        curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0
        metrics = compute_performance_metrics(curve, trades=pd.DataFrame(), annualization=annualization)
        rows.append(
            {
                "iteration": float(iteration),
                "terminal_equity": float(curve["equity"].iloc[-1]),
                "cumulative_return": metrics["cumulative_return"],
                "annualized_return": metrics["annualized_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "maximum_drawdown": metrics["maximum_drawdown"],
            }
        )
        paths.append(equity.rename(iteration))

    summary = pd.DataFrame(rows)
    summary_stats = pd.DataFrame(
        [
            {
                "iterations": float(iterations),
                "block_size": float(block),
                "median_terminal_equity": float(summary["terminal_equity"].median()),
                "p05_terminal_equity": float(summary["terminal_equity"].quantile(0.05)),
                "p95_terminal_equity": float(summary["terminal_equity"].quantile(0.95)),
                "median_sharpe_ratio": float(summary["sharpe_ratio"].median()),
                "p05_sharpe_ratio": float(summary["sharpe_ratio"].quantile(0.05)),
                "p95_sharpe_ratio": float(summary["sharpe_ratio"].quantile(0.95)),
                "probability_positive_return": float((summary["cumulative_return"] > 0.0).mean()),
                "probability_sharpe_above_one": float((summary["sharpe_ratio"] > 1.0).mean()),
            }
        ]
    )
    paths_frame = pd.concat(paths, axis=1)
    paths_frame.columns = [f"path_{int(column)}" for column in paths_frame.columns]
    return summary, summary_stats, paths_frame


def write_research_summary(
    output_path: str | Path,
    strategy: BaseStrategy,
    in_sample: BacktestResult,
    out_of_sample: BacktestResult,
    sweep: pd.DataFrame,
    robustness: pd.DataFrame,
    cost_free_metrics: dict[str, float] | None = None,
    walk_forward_segments: pd.DataFrame | None = None,
    walk_forward_metrics: pd.DataFrame | None = None,
    bootstrap_stats: pd.DataFrame | None = None,
    regime_summary: pd.DataFrame | None = None,
) -> Path:
    """Write a concise strategy summary for interview-style presentation."""

    best_row = sweep.iloc[0].to_dict() if not sweep.empty else {}
    survives_costs = (
        cost_free_metrics is None
        or out_of_sample.metrics["annualized_return"] >= 0.0
        or out_of_sample.metrics["annualized_return"] > cost_free_metrics["annualized_return"] * 0.25
    )
    summary_lines = [
        f"Strategy: {strategy.name}",
        f"Selected parameters: {strategy.describe_parameters()}",
        f"Best in-sample grid row: {best_row}",
        "",
        "In-sample metrics:",
        str(pd.Series(in_sample.metrics).round(4)),
        "",
        "Out-of-sample metrics:",
        str(pd.Series(out_of_sample.metrics).round(4)),
        "",
        "Out-of-sample benchmark metrics:",
        str(pd.Series(out_of_sample.benchmark_metrics).round(4)),
        "",
        "Slippage sensitivity:",
        robustness.round(4).to_string(index=False),
    ]
    if cost_free_metrics is not None:
        summary_lines.extend(
            [
                "",
                "Out-of-sample metrics without costs:",
                str(pd.Series(cost_free_metrics).round(4)),
                f"Results survive modeled costs: {'yes' if survives_costs else 'no'}",
            ]
        )
    if walk_forward_segments is not None and not walk_forward_segments.empty:
        summary_lines.extend(
            [
                "",
                "Walk-forward segment summary:",
                _format_frame_for_text(walk_forward_segments),
            ]
        )
    if walk_forward_metrics is not None and not walk_forward_metrics.empty:
        summary_lines.extend(
            [
                "",
                "Walk-forward stitched metrics:",
                _format_frame_for_text(walk_forward_metrics),
            ]
        )
    if bootstrap_stats is not None and not bootstrap_stats.empty:
        summary_lines.extend(
            [
                "",
                "Bootstrap robustness summary:",
                _format_frame_for_text(bootstrap_stats),
            ]
        )
    if regime_summary is not None and not regime_summary.empty:
        summary_lines.extend(
            [
                "",
                "Regime segmentation summary:",
                _format_frame_for_text(regime_summary),
            ]
        )
    path = Path(output_path)
    path.write_text("\n".join(summary_lines))
    return path
