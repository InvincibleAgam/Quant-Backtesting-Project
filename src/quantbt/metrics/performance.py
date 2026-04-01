"""Portfolio performance and risk metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or np.isnan(denominator):
        return 0.0
    return numerator / denominator


def compute_performance_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    annualization: int = 252,
) -> dict[str, float]:
    """Compute quant-style return and risk metrics."""

    if equity_curve.empty:
        raise ValueError("equity_curve must not be empty")

    returns = equity_curve["returns"].fillna(0.0)
    cumulative_return = equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0] - 1.0
    n_periods = max(len(returns), 1)
    annualized_return = (1.0 + cumulative_return) ** (annualization / n_periods) - 1.0
    annualized_volatility = returns.std(ddof=0) * math.sqrt(annualization)
    downside = returns.clip(upper=0.0)
    downside_volatility = downside.std(ddof=0) * math.sqrt(annualization)
    sharpe_ratio = _safe_divide(returns.mean() * annualization, annualized_volatility)
    sortino_ratio = _safe_divide(returns.mean() * annualization, downside_volatility)
    drawdown = equity_curve["drawdown"] if "drawdown" in equity_curve else equity_curve["equity"] / equity_curve["equity"].cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar_ratio = _safe_divide(annualized_return, abs(max_drawdown))

    if trades.empty:
        win_rate = 0.0
        profit_factor = 0.0
        average_trade_return = 0.0
        number_of_trades = 0.0
    else:
        wins = trades[trades["pnl"] > 0.0]["pnl"]
        losses = trades[trades["pnl"] < 0.0]["pnl"]
        win_rate = float((trades["pnl"] > 0.0).mean())
        profit_factor = _safe_divide(float(wins.sum()), float(abs(losses.sum())))
        average_trade_return = float(trades["return_pct"].mean())
        number_of_trades = float(len(trades))

    metrics = {
        "cumulative_return": float(cumulative_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "maximum_drawdown": max_drawdown,
        "calmar_ratio": float(calmar_ratio),
        "win_rate": win_rate,
        "profit_factor": float(profit_factor),
        "average_trade_return": float(average_trade_return),
        "turnover": float(equity_curve["turnover"].sum()) if "turnover" in equity_curve else 0.0,
        "average_gross_exposure": float(equity_curve["gross_exposure"].mean()) if "gross_exposure" in equity_curve else 0.0,
        "number_of_trades": number_of_trades,
    }
    return metrics
