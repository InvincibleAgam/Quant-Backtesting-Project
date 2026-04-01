"""Plotting and report generation."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantbt.analysis.regimes import plot_regime_performance
from quantbt.types import BacktestResult


def _monthly_returns_heatmap_data(equity_curve: pd.DataFrame) -> pd.DataFrame:
    monthly = equity_curve["returns"].resample("ME").apply(lambda values: (1.0 + values).prod() - 1.0)
    data = monthly.to_frame("return")
    data["year"] = data.index.year
    data["month"] = data.index.strftime("%b")
    heatmap = data.pivot(index="year", columns="month", values="return")
    ordered_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return heatmap.reindex(columns=ordered_months)


def plot_equity_curve(result: BacktestResult, output_dir: Path) -> Path:
    """Save equity curve and benchmark plot."""

    figure, axis = plt.subplots(figsize=(12, 6))
    result.equity_curve["equity"].plot(ax=axis, label=result.strategy_name, linewidth=2.0)
    result.benchmark_curve["equity"].plot(ax=axis, label="benchmark", linestyle="--", alpha=0.8)
    axis.set_title(f"{result.strategy_name} Equity Curve")
    axis.set_ylabel("Portfolio Value")
    axis.legend()
    axis.grid(alpha=0.25)
    path = output_dir / "equity_curve.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_drawdown_curve(result: BacktestResult, output_dir: Path) -> Path:
    """Save drawdown plot."""

    figure, axis = plt.subplots(figsize=(12, 4))
    result.equity_curve["drawdown"].plot(ax=axis, color="firebrick", linewidth=1.8)
    axis.fill_between(result.equity_curve.index, result.equity_curve["drawdown"], 0.0, color="salmon", alpha=0.4)
    axis.set_title(f"{result.strategy_name} Drawdown")
    axis.set_ylabel("Drawdown")
    axis.grid(alpha=0.25)
    path = output_dir / "drawdown.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_trade_distribution(result: BacktestResult, output_dir: Path) -> Path | None:
    """Save trade PnL histogram if trades exist."""

    if result.trades.empty:
        return None
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.hist(result.trades["pnl"], bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    axis.set_title(f"{result.strategy_name} Trade PnL Distribution")
    axis.set_xlabel("PnL")
    axis.set_ylabel("Count")
    axis.grid(alpha=0.2)
    path = output_dir / "trade_distribution.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_rolling_risk(result: BacktestResult, output_dir: Path, window: int = 63) -> Path:
    """Save rolling Sharpe and volatility charts."""

    returns = result.equity_curve["returns"].fillna(0.0)
    rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std(ddof=0)
    rolling_sharpe = rolling_sharpe * np.sqrt(252)
    rolling_vol = returns.rolling(window).std(ddof=0) * np.sqrt(252)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    rolling_sharpe.plot(ax=axes[0], color="darkgreen")
    axes[0].set_title(f"{result.strategy_name} Rolling Sharpe ({window} bars)")
    axes[0].grid(alpha=0.25)
    rolling_vol.plot(ax=axes[1], color="darkorange")
    axes[1].set_title(f"{result.strategy_name} Rolling Volatility ({window} bars)")
    axes[1].grid(alpha=0.25)
    path = output_dir / "rolling_risk.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_monthly_heatmap(result: BacktestResult, output_dir: Path) -> Path:
    """Save a monthly return heatmap."""

    heatmap = _monthly_returns_heatmap_data(result.equity_curve)
    figure, axis = plt.subplots(figsize=(12, 4))
    image = axis.imshow(heatmap.fillna(0.0).values, aspect="auto", cmap="RdYlGn")
    axis.set_title(f"{result.strategy_name} Monthly Returns")
    axis.set_xticks(range(len(heatmap.columns)))
    axis.set_xticklabels(heatmap.columns)
    axis.set_yticks(range(len(heatmap.index)))
    axis.set_yticklabels(heatmap.index)
    for row_idx, year in enumerate(heatmap.index):
        for col_idx, month in enumerate(heatmap.columns):
            value = heatmap.loc[year, month]
            label = "" if pd.isna(value) else f"{value:.1%}"
            axis.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, shrink=0.8)
    path = output_dir / "monthly_heatmap.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_parameter_sweep(sweep: pd.DataFrame, output_dir: Path, metric_name: str = "sharpe_ratio") -> Path | None:
    """Save a grid-search visualization for one- or two-parameter sweeps."""

    parameter_columns = [column for column in sweep.columns if column not in {"strategy", metric_name, "annualized_return", "maximum_drawdown", "number_of_trades"}]
    if not parameter_columns:
        return None

    path = output_dir / "parameter_sweep.png"
    if len(parameter_columns) == 1:
        figure, axis = plt.subplots(figsize=(8, 4))
        x = parameter_columns[0]
        axis.plot(sweep[x], sweep[metric_name], marker="o", linewidth=2.0)
        axis.set_xlabel(x)
        axis.set_ylabel(metric_name)
        axis.set_title("Parameter Sweep")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return path

    x, y = parameter_columns[:2]
    pivot = sweep.pivot_table(index=y, columns=x, values=metric_name, aggfunc="mean")
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(pivot.values, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(pivot.columns)))
    axis.set_xticklabels(pivot.columns)
    axis.set_yticks(range(len(pivot.index)))
    axis.set_yticklabels(pivot.index)
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    axis.set_title("Parameter Sweep")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_walk_forward_curve(walk_forward_curve: pd.DataFrame, output_dir: Path) -> Path | None:
    """Save the stitched walk-forward equity curve."""

    if walk_forward_curve.empty:
        return None
    figure, axis = plt.subplots(figsize=(12, 5))
    walk_forward_curve["equity"].plot(ax=axis, color="navy", linewidth=2.0)
    axis.set_title("Walk-Forward Stitched Out-of-Sample Equity")
    axis.set_ylabel("Portfolio Value")
    axis.grid(alpha=0.25)
    path = output_dir / "walk_forward_equity.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_walk_forward_parameters(walk_forward_segments: pd.DataFrame, output_dir: Path) -> Path | None:
    """Plot parameter choices and out-of-sample Sharpe across walk-forward segments."""

    if walk_forward_segments.empty:
        return None
    parameter_columns = [column for column in walk_forward_segments.columns if column.startswith("param_")]
    figure, axes = plt.subplots(1 + len(parameter_columns), 1, figsize=(12, 3 * (1 + len(parameter_columns))), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    walk_forward_segments["test_sharpe_ratio"].plot(ax=axes[0], marker="o", color="darkgreen")
    axes[0].set_title("Walk-Forward Test Sharpe by Segment")
    axes[0].grid(alpha=0.25)
    for idx, column in enumerate(parameter_columns, start=1):
        walk_forward_segments[column].plot(ax=axes[idx], marker="o", linewidth=1.8)
        axes[idx].set_title(column.replace("param_", "Selected "))
        axes[idx].grid(alpha=0.25)
    path = output_dir / "walk_forward_parameters.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_bootstrap_distributions(bootstrap_samples: pd.DataFrame, output_dir: Path) -> Path | None:
    """Plot bootstrap terminal equity and Sharpe distributions."""

    if bootstrap_samples.empty:
        return None
    figure, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].hist(bootstrap_samples["terminal_equity"], bins=30, color="slateblue", edgecolor="black", alpha=0.8)
    axes[0].set_title("Bootstrap Terminal Equity Distribution")
    axes[0].grid(alpha=0.2)
    axes[1].hist(bootstrap_samples["sharpe_ratio"], bins=30, color="teal", edgecolor="black", alpha=0.8)
    axes[1].set_title("Bootstrap Sharpe Distribution")
    axes[1].grid(alpha=0.2)
    path = output_dir / "bootstrap_distributions.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def create_strategy_report(
    result: BacktestResult,
    output_dir: str | Path,
    sweep: pd.DataFrame | None = None,
    robustness: pd.DataFrame | None = None,
    walk_forward_segments: pd.DataFrame | None = None,
    walk_forward_curve: pd.DataFrame | None = None,
    walk_forward_metrics: pd.DataFrame | None = None,
    bootstrap_samples: pd.DataFrame | None = None,
    bootstrap_stats: pd.DataFrame | None = None,
    bootstrap_paths: pd.DataFrame | None = None,
    regime_summary: pd.DataFrame | None = None,
    regime_assignments: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Write plots and CSV artifacts for one backtest result."""

    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(report_dir / "equity_curve.csv")
    result.fills.to_csv(report_dir / "fills.csv", index=False)
    result.trades.to_csv(report_dir / "trades.csv", index=False)
    if sweep is not None:
        sweep.to_csv(report_dir / "parameter_sweep.csv", index=False)
    if robustness is not None:
        robustness.to_csv(report_dir / "robustness.csv", index=False)
    if walk_forward_segments is not None and not walk_forward_segments.empty:
        walk_forward_segments.to_csv(report_dir / "walk_forward_segments.csv", index=False)
    if walk_forward_curve is not None and not walk_forward_curve.empty:
        walk_forward_curve.to_csv(report_dir / "walk_forward_equity.csv")
    if walk_forward_metrics is not None and not walk_forward_metrics.empty:
        walk_forward_metrics.to_csv(report_dir / "walk_forward_metrics.csv", index=False)
    if bootstrap_samples is not None and not bootstrap_samples.empty:
        bootstrap_samples.to_csv(report_dir / "bootstrap_samples.csv", index=False)
    if bootstrap_stats is not None and not bootstrap_stats.empty:
        bootstrap_stats.to_csv(report_dir / "bootstrap_stats.csv", index=False)
    if bootstrap_paths is not None and not bootstrap_paths.empty:
        bootstrap_paths.to_csv(report_dir / "bootstrap_paths.csv")
    if regime_summary is not None and not regime_summary.empty:
        regime_summary.to_csv(report_dir / "regime_summary.csv", index=False)
    if regime_assignments is not None and not regime_assignments.empty:
        regime_assignments.to_csv(report_dir / "regime_assignments.csv")
    pd.DataFrame([result.metrics]).to_csv(report_dir / "metrics.csv", index=False)
    pd.DataFrame([result.benchmark_metrics]).to_csv(report_dir / "benchmark_metrics.csv", index=False)

    outputs = {
        "equity_curve": str(plot_equity_curve(result, report_dir)),
        "drawdown": str(plot_drawdown_curve(result, report_dir)),
        "rolling_risk": str(plot_rolling_risk(result, report_dir)),
        "monthly_heatmap": str(plot_monthly_heatmap(result, report_dir)),
    }
    trade_dist = plot_trade_distribution(result, report_dir)
    if trade_dist is not None:
        outputs["trade_distribution"] = str(trade_dist)
    sweep_plot = plot_parameter_sweep(sweep, report_dir) if sweep is not None else None
    if sweep_plot is not None:
        outputs["parameter_sweep"] = str(sweep_plot)
    walk_forward_plot = plot_walk_forward_curve(walk_forward_curve, report_dir) if walk_forward_curve is not None else None
    if walk_forward_plot is not None:
        outputs["walk_forward_equity"] = str(walk_forward_plot)
    walk_forward_params_plot = (
        plot_walk_forward_parameters(walk_forward_segments, report_dir)
        if walk_forward_segments is not None
        else None
    )
    if walk_forward_params_plot is not None:
        outputs["walk_forward_parameters"] = str(walk_forward_params_plot)
    bootstrap_plot = plot_bootstrap_distributions(bootstrap_samples, report_dir) if bootstrap_samples is not None else None
    if bootstrap_plot is not None:
        outputs["bootstrap_distributions"] = str(bootstrap_plot)
    regime_plot = plot_regime_performance(regime_summary, report_dir) if regime_summary is not None else None
    if regime_plot is not None:
        outputs["regime_performance"] = str(regime_plot)
    return outputs
