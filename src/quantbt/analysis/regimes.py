"""Market regime inference and segmented performance analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from quantbt.metrics import compute_performance_metrics
from quantbt.types import MarketData
from quantbt.utils import annualization_factor


def infer_market_regimes(
    market_data: MarketData,
    lookback: int = 63,
) -> pd.DataFrame:
    """Infer bull/bear and high/low volatility states from equal-weight market returns."""

    close_wide = market_data.bars["close"].unstack("asset").sort_index().ffill()
    market_level = close_wide.mean(axis=1)
    market_returns = market_level.pct_change().fillna(0.0)
    realized_vol = market_returns.rolling(lookback).std(ddof=0)
    trailing_mean = market_level.rolling(lookback).mean()
    trend_state = pd.Series("bear", index=market_level.index)
    trend_state.loc[market_level >= trailing_mean] = "bull"
    vol_cutoff = realized_vol.expanding(min_periods=max(lookback, 2)).median()
    vol_state = pd.Series("low_vol", index=market_level.index)
    vol_state.loc[realized_vol >= vol_cutoff] = "high_vol"
    regime = trend_state + "_" + vol_state
    return pd.DataFrame(
        {
            "market_level": market_level,
            "market_return": market_returns,
            "realized_volatility": realized_vol,
            "trend_state": trend_state,
            "volatility_state": vol_state,
            "regime": regime,
        }
    ).dropna(subset=["realized_volatility", "market_level"])


def analyze_regime_performance(
    equity_curve: pd.DataFrame,
    benchmark_curve: pd.DataFrame,
    regime_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize strategy and benchmark performance by inferred regime."""

    if regime_frame.empty or equity_curve.empty or benchmark_curve.empty:
        return pd.DataFrame()
    merged = regime_frame.join(
        equity_curve[["equity", "returns"]],
        how="inner",
    ).join(
        benchmark_curve[["equity", "returns"]].rename(
            columns={"equity": "benchmark_equity", "returns": "benchmark_returns"}
        ),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | str]] = []
    annualization = annualization_factor(merged.index)

    for regime, group in merged.groupby("regime"):
        strategy_curve = pd.DataFrame(
            {
                "returns": group["returns"].fillna(0.0),
            },
            index=group.index,
        )
        strategy_curve["equity"] = 1.0 * (1.0 + strategy_curve["returns"]).cumprod()
        strategy_curve["drawdown"] = strategy_curve["equity"] / strategy_curve["equity"].cummax() - 1.0

        benchmark_local = pd.DataFrame(
            {
                "returns": group["benchmark_returns"].fillna(0.0),
            },
            index=group.index,
        )
        benchmark_local["equity"] = 1.0 * (1.0 + benchmark_local["returns"]).cumprod()
        benchmark_local["drawdown"] = benchmark_local["equity"] / benchmark_local["equity"].cummax() - 1.0

        strategy_metrics = compute_performance_metrics(strategy_curve, pd.DataFrame(), annualization)
        benchmark_metrics = compute_performance_metrics(benchmark_local, pd.DataFrame(), annualization)
        rows.append(
            {
                "regime": regime,
                "observations": float(len(group)),
                "strategy_cumulative_return": strategy_metrics["cumulative_return"],
                "strategy_annualized_return": strategy_metrics["annualized_return"],
                "strategy_sharpe_ratio": strategy_metrics["sharpe_ratio"],
                "strategy_maximum_drawdown": strategy_metrics["maximum_drawdown"],
                "benchmark_cumulative_return": benchmark_metrics["cumulative_return"],
                "benchmark_annualized_return": benchmark_metrics["annualized_return"],
                "benchmark_sharpe_ratio": benchmark_metrics["sharpe_ratio"],
                "benchmark_maximum_drawdown": benchmark_metrics["maximum_drawdown"],
                "average_market_return": float(group["market_return"].mean()),
                "average_market_volatility": float(group["realized_volatility"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def plot_regime_performance(regime_summary: pd.DataFrame, output_dir: str | Path) -> Path | None:
    """Plot strategy versus benchmark performance by regime."""

    if regime_summary.empty:
        return None
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = range(len(regime_summary))
    axes[0].bar(x, regime_summary["strategy_annualized_return"], width=0.4, label="strategy", alpha=0.85)
    axes[0].bar(
        [value + 0.4 for value in x],
        regime_summary["benchmark_annualized_return"],
        width=0.4,
        label="benchmark",
        alpha=0.75,
    )
    axes[0].set_title("Annualized Return by Regime")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].bar(x, regime_summary["strategy_sharpe_ratio"], width=0.4, label="strategy", alpha=0.85)
    axes[1].bar(
        [value + 0.4 for value in x],
        regime_summary["benchmark_sharpe_ratio"],
        width=0.4,
        label="benchmark",
        alpha=0.75,
    )
    axes[1].set_title("Sharpe Ratio by Regime")
    axes[1].grid(alpha=0.2)
    axes[1].set_xticks([value + 0.2 for value in x])
    axes[1].set_xticklabels(regime_summary["regime"], rotation=20)
    path = report_dir / "regime_performance.png"
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
