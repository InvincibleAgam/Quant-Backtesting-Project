"""Base strategy abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any

import pandas as pd

from quantbt.types import MarketData


def stack_signals(signals_by_asset: dict[str, pd.Series]) -> pd.DataFrame:
    """Combine asset-level signal series into canonical MultiIndex format."""

    frames: list[pd.DataFrame] = []
    for asset, series in signals_by_asset.items():
        local = pd.DataFrame({"signal": series.astype(float)})
        local["asset"] = asset
        local.index.name = "timestamp"
        frames.append(local.reset_index().set_index(["timestamp", "asset"]))
    return pd.concat(frames).sort_index()


@dataclass(slots=True)
class BaseStrategy(ABC):
    """Abstract strategy API."""

    long_only: bool = False
    name: str = field(init=False, default="base")

    @abstractmethod
    def generate_signals(self, market_data: MarketData) -> pd.DataFrame:
        """Return a MultiIndex DataFrame with a `signal` column."""

    @classmethod
    def default_parameter_grid(cls) -> list[dict[str, Any]]:
        """Default grid used for in-sample parameter sweeps."""

        return []

    def describe_parameters(self) -> dict[str, Any]:
        """Return constructor parameters for reporting."""

        params = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "name"
        }
        return params
