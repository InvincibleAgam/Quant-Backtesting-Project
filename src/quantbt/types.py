"""Shared domain models and typed containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class MarketData:
    """Canonical OHLCV container using a MultiIndex of timestamp and asset."""

    bars: pd.DataFrame

    def __post_init__(self) -> None:
        required_columns = {"open", "high", "low", "close", "volume"}
        missing_columns = required_columns.difference(self.bars.columns)
        if missing_columns:
            raise ValueError(f"bars missing required columns: {sorted(missing_columns)}")
        if not isinstance(self.bars.index, pd.MultiIndex):
            raise TypeError("bars index must be a MultiIndex of timestamp and asset")
        if list(self.bars.index.names) != ["timestamp", "asset"]:
            raise ValueError("bars index names must be ['timestamp', 'asset']")
        if not self.bars.index.is_monotonic_increasing:
            raise ValueError("bars must be sorted by timestamp and asset")

    @property
    def assets(self) -> list[str]:
        return list(self.bars.index.get_level_values("asset").unique())

    @property
    def timestamps(self) -> pd.Index:
        return self.bars.index.get_level_values("timestamp").unique()

    def asset_frame(self, asset: str) -> pd.DataFrame:
        """Return one asset view indexed by timestamp."""

        return self.bars.xs(asset, level="asset").copy()

    def slice_until(self, timestamp: pd.Timestamp) -> "MarketData":
        """Return all bars up to and including the supplied timestamp."""

        subset = self.bars.loc[(slice(None, timestamp), slice(None)), :]
        return MarketData(subset.copy())

    def between(self, start: pd.Timestamp, end: pd.Timestamp) -> "MarketData":
        """Return a time-sliced view of the data."""

        subset = self.bars.loc[(slice(start, end), slice(None)), :]
        return MarketData(subset.copy())

    def resample(self, rule: str) -> "MarketData":
        """Resample each asset independently."""

        frames: list[pd.DataFrame] = []
        for asset in self.assets:
            frame = self.asset_frame(asset)
            resampled = pd.DataFrame(
                {
                    "open": frame["open"].resample(rule).first(),
                    "high": frame["high"].resample(rule).max(),
                    "low": frame["low"].resample(rule).min(),
                    "close": frame["close"].resample(rule).last(),
                    "volume": frame["volume"].resample(rule).sum(),
                }
            ).dropna()
            resampled["asset"] = asset
            frames.append(resampled.reset_index().set_index(["timestamp", "asset"]))
        combined = pd.concat(frames).sort_index()
        return MarketData(combined)


@dataclass(slots=True)
class MarketEvent:
    """A market data event for a single timestamp."""

    timestamp: pd.Timestamp
    bars: pd.DataFrame


@dataclass(slots=True)
class SignalEvent:
    """A strategy output at a point in time."""

    timestamp: pd.Timestamp
    asset: str
    signal: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Order:
    """Market order scheduled for future execution."""

    timestamp: pd.Timestamp
    asset: str
    quantity: int
    reason: str
    signal: float

    @property
    def side(self) -> int:
        return 1 if self.quantity > 0 else -1


@dataclass(slots=True)
class Fill:
    """Executed order fill."""

    timestamp: pd.Timestamp
    asset: str
    quantity: int
    fill_price: float
    commission: float
    slippage_cost: float
    spread_cost: float
    notional: float
    reason: str


@dataclass(slots=True)
class TradeRecord:
    """Closed trade record for performance analysis."""

    asset: str
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    return_pct: float
    holding_period_bars: int


@dataclass(slots=True)
class BacktestResult:
    """Full simulation artifacts."""

    strategy_name: str
    parameters: dict[str, Any]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    fills: pd.DataFrame
    signals: pd.DataFrame
    metrics: dict[str, float]
    benchmark_curve: pd.DataFrame
    benchmark_metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
