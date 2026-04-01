"""Shared utility functions."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd


def annualization_factor(index: pd.Index, default: int = 252) -> int:
    """Infer annualization from timestamps when possible."""

    if len(index) < 2:
        return default
    deltas = pd.Series(index).diff().dropna()
    median_delta = deltas.median()
    if median_delta <= pd.Timedelta("2D"):
        return 252
    if median_delta <= pd.Timedelta("10D"):
        return 52
    return 12


def to_plain_dict(value: Any) -> dict[str, Any]:
    """Convert dataclass-like objects to a regular dictionary."""

    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError("value must be a dataclass or dictionary")


def train_test_split_index(index: pd.Index, train_fraction: float) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the last in-sample timestamp and first out-of-sample timestamp."""

    timestamps = pd.Index(index.unique()).sort_values()
    split_idx = max(1, int(len(timestamps) * train_fraction))
    split_idx = min(split_idx, len(timestamps) - 1)
    train_end = timestamps[split_idx - 1]
    test_start = timestamps[split_idx]
    return train_end, test_start


def build_equal_weight_benchmark(market_data: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    """Construct an equal-weight buy-and-hold benchmark from close prices."""

    close_wide = market_data["close"].unstack("asset").dropna(how="all")
    close_wide = close_wide.ffill().dropna(how="all")
    if close_wide.empty:
        raise ValueError("benchmark cannot be built from empty close data")
    returns = close_wide.pct_change().fillna(0.0)
    portfolio_returns = returns.mean(axis=1).fillna(0.0)
    equity = initial_cash * (1.0 + portfolio_returns).cumprod()
    benchmark = pd.DataFrame(
        {
            "equity": equity,
            "returns": equity.pct_change().fillna(0.0),
        }
    )
    drawdown = benchmark["equity"] / benchmark["equity"].cummax() - 1.0
    benchmark["drawdown"] = drawdown
    return benchmark


def set_random_seed(seed: int) -> None:
    """Seed NumPy for deterministic operations."""

    np.random.seed(seed)
